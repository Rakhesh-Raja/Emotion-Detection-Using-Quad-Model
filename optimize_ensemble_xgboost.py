import os
import numpy as np
import torch
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# 1. SETUP PATHS
ROOT = os.path.dirname(os.path.abspath(__file__))
# folder containing your .npy logit files
OUT_DIR = os.path.join(ROOT, "outputs", "checkpoints_ensemble")
MODEL_SAVE_PATH = os.path.join(OUT_DIR, "xgboost_meta_learner.joblib")

LABELS = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]

def train_and_save_meta_learner():
    print("📂 Loading data for Meta-Learner...")
    
    # 2. LOAD GROUND TRUTH (y_true)
    # Using the Dev set as the training data for the Meta-Learner
    dev_path = os.path.join(ROOT, "cached_hyper_dev.pt")
    dev_data = torch.load(dev_path, weights_only=False)
    
    label2id = {l.lower(): i for i, l in enumerate(LABELS)}
    y_true = []
    for x in dev_data:
        val = str(x.get("label", "4")).lower()
        if val.isdigit(): y_true.append(int(val))
        elif val in label2id: y_true.append(label2id[val])
        else: y_true.append(4) 
    y_true = np.array(y_true)

    # 3. LOAD EXPERT LOGITS
    # We need to make sure the order of models is ALWAYS the same
    expert_names = ["RoBERTa", "Qwen", "HuBERT", "WavLM", "SigLIP"]
    features_list = []
    
    for name in expert_names:
        path = os.path.join(OUT_DIR, f"logits_{name}.npy")
        if os.path.exists(path):
            logits = np.load(path)
            features_list.append(logits)
            print(f"✅ Added {name} features")
        else:
            print(f"❌ Missing {name} logits! Cannot save full meta-learner.")
            return

    # Stack logits: If each model has 7 outputs, X will have 35 features per sample
    X = np.hstack(features_list)
    y = y_true

    print(f"📊 Meta-learner feature shape: {X.shape}")

    # 4. TRAIN XGBOOST
    print("🚀 Training XGBoost Meta-Learner...")
    # Best params for logit-based meta-learning
    meta_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=7,
        random_state=42
    )
    
    meta_model.fit(X, y)

    # 5. VERIFY AND SAVE
    y_pred = meta_model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"\n🏆 Meta-Learner Accuracy on Dev: {acc*100:.2f}%")
    
    # SAVE THE MODEL
    joblib.dump(meta_model, MODEL_SAVE_PATH)
    print(f"💾 SUCCESS: Meta-Learner saved to: {MODEL_SAVE_PATH}")

    # Save the order of experts (vital for inference!)
    config_path = os.path.join(OUT_DIR, "meta_learner_config.pkl")
    joblib.dump({'expert_order': expert_names, 'labels': LABELS}, config_path)

if __name__ == "__main__":
    train_and_save_meta_learner()