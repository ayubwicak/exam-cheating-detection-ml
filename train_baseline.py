"""
Baseline Model - Machine Learning Competition (MLC)
Tema: Honest Score, Equal Chance: Exam Cheating Detection
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from lightgbm import LGBMClassifier

# 1. Pastikan SEED konsisten (Syarat Wajib Guidebook MLC Hal 6 Poin 9)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Load Data
print("--- [1] Membaca Dataset ---")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print(f"Data Latih (Train): {train.shape}")
print(f"Data Uji (Test): {test.shape}")

# 3. Feature Selection
target_col = "is_cheating"
features = [col for col in train.columns if col not in ["student_id", target_col]]
print(f"\nFitur yang digunakan ({len(features)}): {features}")

X = train[features]
y = train[target_col]
X_test = test[features]

# 4. Cross Validation (Stratified K-Fold 5-Split)
print("\n--- [2] Memulai Pelatihan Model dengan 5-Fold Stratified CV ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
    
    # Model: LightGBM
    model = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        random_state=RANDOM_SEED,
        class_weight="balanced", # Menangani imbalance data kecurangan
        verbose=-1
    )
    
    model.fit(X_train_fold, y_train_fold)
    
    val_probs = model.predict_proba(X_val_fold)[:, 1]
    oof_preds[val_idx] = val_probs
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits
    models.append(model)
    
    fold_auc = roc_auc_score(y_val_fold, val_probs)
    print(f"Fold {fold + 1} - ROC-AUC: {fold_auc:.4f}")

# 5. Evaluasi Out-of-Fold (OOF)
oof_auc = roc_auc_score(y, oof_preds)
oof_pred_binary = (oof_preds > 0.5).astype(int)
oof_f1 = f1_score(y, oof_pred_binary)

print("\n--- [3] Evaluasi Validasi Lokal ---")
print(f"Overall OOF ROC-AUC: {oof_auc:.4f}")
print(f"Overall OOF F1-Score: {oof_f1:.4f}")
print("\nLaporan Klasifikasi:")
print(classification_report(y, oof_pred_binary, target_names=["Jujur (0)", "Curang (1)"]))

# 6. Simpan Model Terlatih (Format .pkl - Syarat Hal 6 Poin 8)
best_model = models[0]
model_path = os.path.join(OUTPUT_DIR, "cheating_detection_model.pkl")
joblib.dump(best_model, model_path)
print(f"--- [4] Model berhasil disimpan ke: {model_path} ---")

# 7. Buat Submission File untuk Kaggle
test_pred_binary = (test_preds > 0.5).astype(int)
sub = pd.DataFrame({
    "student_id": test["student_id"],
    "is_cheating": test_pred_binary
})
sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
sub.to_csv(sub_path, index=False)
print(f"--- [5] File submisi siap upload ke Kaggle: {sub_path} ---")
print(f"Prediksi Uji: {dict(sub['is_cheating'].value_counts())}")
