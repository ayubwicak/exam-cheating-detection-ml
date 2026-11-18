"""Modul Explainable AI (XAI) berbasis SHAP (SHapley Additive exPlanations).

Menghasilkan:
1. SHAP Summary Plot (Global Feature Impact: arah dan magnitudo pengaruh fitur).
2. SHAP Force / Waterfall Plot (Local Explanation: mengapa satu siswa tertentu dicurigai curang).
3. Ringkasan teks interpretasi untuk presentasi babak final / dewan juri.
"""

from __future__ import annotations

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .config import OUTPUT_DIR, SEED, TRAIN_PATH
from .features import build_feature_frame
from .models import make_model


def explain_model_shap(
    model_name: str = "lightgbm",
    feature_set: str = "C",
    sample_size: int = 500,
    output_dir: Path | str = OUTPUT_DIR,
) -> dict:
    """Hitung SHAP values dan simpan visualisasi interpretable AI."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load data & siapkan fitur
    train = pd.read_csv(TRAIN_PATH)
    X = build_feature_frame(train, feature_set)
    if isinstance(X, tuple):
        X = X[0]
    y = train["is_cheating"]

    # 2. Latih model
    print(f"[*] Melatih model '{model_name}' pada Feature Set '{feature_set}' untuk analisis SHAP...")
    model = make_model(model_name, seed=SEED)
    model.fit(X, y)

    # 3. Hitung SHAP menggunakan TreeExplainer
    print("[*] Menghitung SHAP Values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    X_sample = X.sample(n=min(sample_size, len(X)), random_state=SEED)
    shap_values = explainer(X_sample)

    # Untuk binary classification di LightGBM / XGBoost, ambil slice class 1 jika 3D
    if len(shap_values.shape) == 3:
        shap_vals_class1 = shap_values[:, :, 1]
    else:
        shap_vals_class1 = shap_values

    # 4. Generate & Simpan Summary Plot (Global Interpretation)
    summary_plot_path = output_path / "shap_summary_plot.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_class1, X_sample, show=False)
    plt.title("SHAP Global Summary: Kontribusi & Arah Pengaruh Fitur terhadap Vonis Curang", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(summary_plot_path, dpi=300)
    plt.close()
    print(f"[✓] SHAP Summary Plot tersimpan ke: {summary_plot_path}")

    # 5. Generate & Simpan Waterfall Plot untuk Siswa Kasus Tertentu (Local Interpretation)
    # Cari 1 mahasiswa dengan probabilitas curang tinggi untuk dijadikan studi kasus
    probs = model.predict_proba(X_sample)[:, 1]
    suspect_idx = int(np.argmax(probs))
    
    waterfall_plot_path = output_path / "shap_individual_case.png"
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_vals_class1[suspect_idx], show=False)
    plt.title(f"SHAP Local Waterfall: Audit Bukti Kecurangan (Probabilitas: {probs[suspect_idx]*100:.1f}%)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(waterfall_plot_path, dpi=300)
    plt.close()
    print(f"[✓] SHAP Individual Case Plot tersimpan ke: {waterfall_plot_path}")

    # 6. Hitung Top Fitur Paling Berpengaruh berdasarkan Mean Absolute SHAP
    mean_abs_shap = np.abs(shap_vals_class1.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    summary_csv_path = output_path / "shap_importance.csv"
    importance_df.to_csv(summary_csv_path, index=False)
    print(f"[✓] Peringkat SHAP tersimpan ke: {summary_csv_path}")

    return {
        "summary_plot": str(summary_plot_path),
        "case_plot": str(waterfall_plot_path),
        "top_features": importance_df.head(5).to_dict(orient="records"),
    }


if __name__ == "__main__":
    explain_model_shap()
