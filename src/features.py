"""Definisi fitur dan feature engineering untuk dataset ujian.

Feature set dibuat eksplisit supaya eksperimen tidak diam-diam
memasukkan kolom target atau kolom ID ke dalam model.
"""

from __future__ import annotations

import pandas as pd

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

# Interaksi tambahan.
ACTIVITY_FEATURES = [
    "activity_anomaly",
]

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "A": tuple(BASE_FEATURES),
    "B": tuple(BASE_FEATURES + SPEED_FEATURES),
    "C": tuple(BASE_FEATURES + SPEED_FEATURES + JOKI_FEATURES + ACTIVITY_FEATURES),
}


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur turunan domain tanpa mengubah dataframe asli."""
    out = df.copy()

    duration = out["duration_minutes"].clip(lower=1e-5)

    # Ide 1: nilai tinggi dengan waktu pengerjaan singkat.
    out["score_speed_ratio"] = out["score"] / duration
    out["is_suspicious_speed"] = (
        (out["score"] >= 90) & (out["duration_minutes"] <= 25)
    ).astype(int)

    # Ide 2: pergantian IP bersamaan dengan perpindahan tab.
    out["joki_remote_anomaly"] = (
        out["network_ip_changes"] * out["tab_switch_count"]
    )
    out["is_suspicious_joki"] = (
        (out["network_ip_changes"] > 0) & (out["tab_switch_count"] >= 3)
    ).astype(int)

    # Interaksi aktivitas menyalin dan berpindah tab.
    out["activity_anomaly"] = (
        out["tab_switch_count"] * out["copy_paste_count"]
    )

    return out


def get_feature_columns(feature_set: str) -> list[str]:
    """Kembalikan daftar kolom untuk nama feature set (A/B/C)."""
    key = feature_set.upper()
    if key not in FEATURE_SETS:
        valid = ", ".join(sorted(FEATURE_SETS))
        raise ValueError(f"Feature set '{feature_set}' tidak dikenal. Pilihan: {valid}")
    return list(FEATURE_SETS[key])


def build_feature_frame(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Terapkan feature engineering lalu pilih kolom sesuai feature set."""
    featured = add_domain_features(df)
    columns = get_feature_columns(feature_set)

    missing = [column for column in columns if column not in featured.columns]
    if missing:
        raise ValueError(f"Kolom fitur tidak ditemukan: {missing}")

    return featured[columns].copy()
