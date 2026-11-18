"""Definisi fitur dan feature engineering untuk dataset ujian.

Feature set dibuat eksplisit supaya eksperimen tidak diam-diam
memasukkan kolom target atau kolom ID ke dalam model.
Dilengkapi SimpleImputer otomatis untuk menjamin robustness terhadap missing values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Fitur mentah yang tersedia di dataset.
BASE_FEATURES = [
    "duration_minutes",
    "score",
    "tab_switch_count",
    "copy_paste_count",
    "avg_time_per_question",
    "mouse_movement_entropy",
    "network_ip_changes",
    "similarity_with_peers",
]

# Ide 1: speedrun cheating.
SPEED_FEATURES = [
    "score_speed_ratio",
    "is_suspicious_speed",
]

# Ide 2: joki / remote access.
JOKI_FEATURES = [
    "joki_remote_anomaly",
    "is_suspicious_joki",
]

# Interaksi aktivitas dasar.
ACTIVITY_FEATURES = [
    "activity_anomaly",
]

# Ide 3: Sindikat jawaban / Peer Collusion & Behavioral Discrepancy.
COLLUSION_FEATURES = [
    "sim_tab_interaction",
    "sim_copy_interaction",
    "collusion_anomaly",
]

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "A": tuple(BASE_FEATURES),
    "B": tuple(BASE_FEATURES + SPEED_FEATURES),
    "C": tuple(BASE_FEATURES + SPEED_FEATURES + JOKI_FEATURES + ACTIVITY_FEATURES),
    "D": tuple(BASE_FEATURES + SPEED_FEATURES + JOKI_FEATURES + ACTIVITY_FEATURES + COLLUSION_FEATURES),
}


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur turunan domain tanpa mengubah dataframe asli."""
    out = df.copy()

    # Pastikan duration tidak nol atau negatif untuk menghindari zero-division
    duration = out["duration_minutes"].fillna(out["duration_minutes"].median()).clip(lower=1e-5)
    score = out["score"].fillna(out["score"].median())

    # Ide 1: nilai tinggi dengan waktu pengerjaan singkat.
    out["score_speed_ratio"] = score / duration
    out["is_suspicious_speed"] = (
        (score >= 90) & (duration <= 25)
    ).astype(int)

    # Ide 2: pergantian IP bersamaan dengan perpindahan tab.
    ip_changes = out["network_ip_changes"].fillna(0)
    tab_switches = out["tab_switch_count"].fillna(0)
    out["joki_remote_anomaly"] = ip_changes * tab_switches
    out["is_suspicious_joki"] = (
        (ip_changes > 0) & (tab_switches >= 3)
    ).astype(int)

    # Interaksi aktivitas menyalin dan berpindah tab.
    copy_paste = out["copy_paste_count"].fillna(0)
    out["activity_anomaly"] = tab_switches * copy_paste

    # Ide 3: Kolusi Jawaban & Anomali Pasif Kursor
    sim = out["similarity_with_peers"].fillna(out["similarity_with_peers"].median())
    mouse = out["mouse_movement_entropy"].fillna(out["mouse_movement_entropy"].median())
    out["sim_tab_interaction"] = sim * tab_switches
    out["sim_copy_interaction"] = sim * copy_paste
    out["collusion_anomaly"] = sim * (1.0 - mouse)

    return out


def get_feature_columns(feature_set: str) -> list[str]:
    """Kembalikan daftar kolom untuk nama feature set (A/B/C/D)."""
    key = feature_set.upper()
    if key not in FEATURE_SETS:
        valid = ", ".join(sorted(FEATURE_SETS))
        raise ValueError(f"Feature set '{feature_set}' tidak dikenal. Pilihan: {valid}")
    return list(FEATURE_SETS[key])


def build_feature_frame(
    df: pd.DataFrame, 
    feature_set: str, 
    imputer: SimpleImputer | None = None,
    fit_imputer: bool = False
) -> tuple[pd.DataFrame, SimpleImputer | None] | pd.DataFrame:
    """Terapkan feature engineering lalu pilih kolom sesuai feature set.
    
    Dilengkapi robust imputation: mengisi nilai missing/NaN dengan median
    secara aman tanpa data leakage.
    """
    featured = add_domain_features(df)
    columns = get_feature_columns(feature_set)

    missing = [column for column in columns if column not in featured.columns]
    if missing:
        raise ValueError(f"Kolom fitur tidak ditemukan: {missing}")

    X = featured[columns].copy()

    # Cek apakah ada missing values
    if X.isnull().values.any():
        if fit_imputer or imputer is None:
            active_imputer = SimpleImputer(strategy="median")
            X_imputed = active_imputer.fit_transform(X)
            X = pd.DataFrame(X_imputed, columns=columns, index=df.index)
            if fit_imputer:
                return X, active_imputer
        else:
            X_imputed = imputer.transform(X)
            X = pd.DataFrame(X_imputed, columns=columns, index=df.index)

    if fit_imputer:
        dummy_imputer = SimpleImputer(strategy="median")
        dummy_imputer.fit(X)
        return X, dummy_imputer

    return X
