# train_ensemble_final.py
"""
MASSIVE ENSEMBLE EMOTION CLASSIFIER
- Trains 5 Experts Sequentially.
- SAVES models to disk after training.
- FIXED: SigLIP hidden_size access error.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from accelerate import Accelerator
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModel, 
    AutoImageProcessor, Wav2Vec2FeatureExtractor,
    HubertModel, WavLMModel, SiglipVisionModel,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ================= CONFIG =================
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "outputs", "checkpoints_ensemble")
os.makedirs(OUT_DIR, exist_ok=True)

# --- THE ROSTER OF EXPERTS ---
MODEL_ZOO = [
    ("text",    "roberta-base",                      "RoBERTa"),
    ("text",    "Qwen/Qwen2.5-1.5B-Instruct",        "Qwen"),
    ("audio",   "facebook/hubert-base-ls960",        "HuBERT"),
    ("audio",   "microsoft/wavlm-base-plus",         "WavLM"),
    ("video",   "google/siglip-base-patch16-224",    "SigLIP") 
]

TRAIN_CSV = os.path.join(ROOT, "outputs", "train_merged.csv")
DEV_CSV   = os.path.join(ROOT, "outputs", "dev_merged.csv")
MELD_RAW_DIR = os.path.join(ROOT, "data", "MELD", "MELD.Raw")
FRAME_CACHE_DIR = os.path.join(ROOT, "data", "MELD", "frames_cache")

BATCH_SIZE_LARGE = 16 
BATCH_SIZE_LLM   = 1  
ACCUM_STEPS = 8
EPOCHS = 3 
LR = 2e-5
MAX_LEN = 128
FRAMES = 4
SAMPLE_RATE = 16000

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
label2id = {l: i for i, l in enumerate(LABELS)}

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ================= DATASET (RAM CACHED) =================
class RAMDataset(Dataset):
    def __init__(self, split_name):
        filename = os.path.join(ROOT, f"cached_hyper_{split_name.lower()}.pt")
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Cache file {filename} not found! Run 'generate_hyper_cache_turbo.py' first.")
            
        print(f"[{split_name}] Loading cache from disk...")
        try:
            self.data = torch.load(filename, weights_only=False)
        except:
            self.data = torch.load(filename)
            
        self.targets = [x["label"] for x in self.data]

    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        frames = item["frames"]
        
        # Pad frames if needed
        frames_list = []
        if len(frames) > 0:
            while len(frames_list) < FRAMES:
                frames_list.extend(frames)
            frames_list = frames_list[:FRAMES]
        else:
            frames_list = [np.zeros((224, 224, 3), dtype=np.uint8)] * FRAMES
            
        return {
            "text": item["text"],
            "video": frames_list, 
            "audio": item["audio"],
            "label": item["label"]
        }

class UniversalCollator:
    def __init__(self, processor, mode):
        self.proc = processor
        self.mode = mode
        
    def __call__(self, batch):
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        
        if self.mode == "text":
            texts = [b["text"] for b in batch]
            if self.proc.pad_token is None: self.proc.pad_token = self.proc.eos_token
            out = self.proc(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            return {"input_ids": out.input_ids, "attention_mask": out.attention_mask, "labels": labels}
            
        elif self.mode == "audio":
            audios = [b["audio"] for b in batch]
            out = self.proc(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
            return {"input_values": out.input_values, "labels": labels}
            
        elif self.mode == "video":
            vis_raw = [img for b in batch for img in b["video"]]
            out = self.proc(images=vis_raw, return_tensors="pt")
            return {"pixel_values": out.pixel_values, "labels": labels}

# ================= EXPERT MODEL =================
class ExpertModel(nn.Module):
    def __init__(self, model_name, mode):
        super().__init__()
        self.mode = mode
        
        if mode == "text":
            if "Qwen" in model_name:
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
                self.backbone = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb)
                self.backbone = prepare_model_for_kbit_training(self.backbone)
                peft = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], task_type="CAUSAL_LM")
                self.backbone = get_peft_model(self.backbone, peft)
                self.dim = self.backbone.config.hidden_size
            else:
                self.backbone = AutoModel.from_pretrained(model_name)
                self.dim = self.backbone.config.hidden_size
        
        elif mode == "audio":
            if "hubert" in model_name: self.backbone = HubertModel.from_pretrained(model_name)
            else: self.backbone = WavLMModel.from_pretrained(model_name)
            self.backbone.feature_extractor._freeze_parameters()
            self.dim = self.backbone.config.hidden_size
            
        elif mode == "video":
            if "siglip" in model_name:
                self.backbone = SiglipVisionModel.from_pretrained(model_name)
                # FIX: SiglipVisionModel uses .hidden_size directly, NOT .vision_config.hidden_size
                self.dim = self.backbone.config.hidden_size 
            else:
                self.backbone = AutoModel.from_pretrained(model_name)
                self.dim = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(LABELS))
        )

    def forward(self, batch):
        if self.mode == "text":
            if hasattr(self.backbone, "get_input_embeddings"): 
                outputs = self.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True)
                pooled = outputs.hidden_states[-1][:, -1, :] 
            else: 
                outputs = self.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                pooled = outputs.last_hidden_state[:, 0, :]
                
        elif self.mode == "audio":
            outputs = self.backbone(input_values=batch["input_values"])
            pooled = outputs.last_hidden_state.mean(dim=1)
            
        elif self.mode == "video":
            outputs = self.backbone(pixel_values=batch["pixel_values"])
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feats = outputs.pooler_output
            else:
                feats = outputs.last_hidden_state.mean(dim=1)
            
            B = batch["labels"].size(0)
            feats = feats.view(B, -1, self.dim) 
            pooled = feats.mean(dim=1)

        return self.classifier(pooled)

# ================= TRAINING LOOP =================
def train_expert(m_type, m_name, friendly_name, train_ds, dev_ds):
    save_file = os.path.join(OUT_DIR, f"logits_{friendly_name}.npy")
    model_file = os.path.join(OUT_DIR, f"model_{friendly_name}.pt")
    
    if os.path.exists(save_file) and os.path.exists(model_file):
        logger.info(f"✅ Skipping {friendly_name} (Already done).")
        return

    logger.info(f"\n🚀 Training Expert: {friendly_name} ({m_name})")
    accelerator = Accelerator(gradient_accumulation_steps=ACCUM_STEPS, mixed_precision="fp16")
    
    # Init Processor
    try:
        if m_type == "text": processor = AutoTokenizer.from_pretrained(m_name, use_fast=True)
        elif m_type == "audio": processor = Wav2Vec2FeatureExtractor.from_pretrained(m_name)
        elif m_type == "video": processor = AutoImageProcessor.from_pretrained(m_name)
    except:
        # Retry without fast if failed
        if m_type == "text": processor = AutoTokenizer.from_pretrained(m_name)
        elif m_type == "audio": processor = Wav2Vec2FeatureExtractor.from_pretrained(m_name)
        elif m_type == "video": processor = AutoImageProcessor.from_pretrained(m_name)

    curr_batch = 1 if "Qwen" in m_name else BATCH_SIZE_LARGE

    weights = 1.0 / (np.bincount(train_ds.targets) + 1e-6)
    sampler = WeightedRandomSampler([weights[t] for t in train_ds.targets], len(train_ds.targets))
    
    collator = UniversalCollator(processor, m_type)
    train_dl = DataLoader(train_ds, batch_size=curr_batch, sampler=sampler, collate_fn=collator, num_workers=0)
    dev_dl = DataLoader(dev_ds, batch_size=curr_batch, shuffle=False, collate_fn=collator, num_workers=0)

    model = ExpertModel(m_name, m_type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    model, optimizer, train_dl, dev_dl = accelerator.prepare(model, optimizer, train_dl, dev_dl)

    for epoch in range(EPOCHS):
        model.train()
        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(model):
                logits = model(batch)
                loss = loss_fn(logits, batch["labels"])
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            if step % 50 == 0 and accelerator.is_main_process:
                print(f"[{friendly_name}] Epoch {epoch+1} Loss: {loss.item():.4f}")

    # Save Model & Logits
    model.eval()
    if accelerator.is_main_process:
        print(f"💾 Saving {friendly_name} model to {model_file}...")
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), model_file)

    all_logits = []
    with torch.no_grad():
        for batch in tqdm(dev_dl, desc=f"Inferencing {friendly_name}"):
            logits = model(batch)
            # Normalization before saving
            probs = F.softmax(logits, dim=-1)
            all_logits.append(accelerator.gather(probs).cpu().numpy())
    
    final_logits = np.concatenate(all_logits)
    if len(final_logits) > len(dev_ds): final_logits = final_logits[:len(dev_ds)]
        
    np.save(save_file, final_logits)
    logger.info(f"✅ Saved logits to {save_file}")

    del model, optimizer, train_dl, dev_dl
    accelerator.free_memory()
    torch.cuda.empty_cache()
    gc.collect()

def main():
    train_ds = RAMDataset("TRAIN")
    dev_ds = RAMDataset("DEV")

    for m_type, m_name, friendly in MODEL_ZOO:
        try:
            train_expert(m_type, m_name, friendly, train_ds, dev_ds)
        except Exception as e:
            print(f"❌ Failed {friendly}: {e}")
            import traceback
            traceback.print_exc()

    print("\n🧠 Calculating Ensemble Result...")
    y_true = np.array(dev_ds.targets)
    logit_files = [f for f in os.listdir(OUT_DIR) if f.endswith(".npy")]
    
    model_preds = []
    for f in logit_files:
        p = np.load(os.path.join(OUT_DIR, f))
        if len(p) == len(y_true):
            model_preds.append(p)
            print(f"Loaded: {f}")

    if model_preds:
        avg_logits = np.sum(model_preds, axis=0)
        final_preds = np.argmax(avg_logits, axis=1)
        acc = np.mean(final_preds == y_true)
        print(f"\n🏆 FINAL ENSEMBLE ACCURACY: {acc*100:.2f}%")

if __name__ == "__main__":
    main()