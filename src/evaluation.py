"""Evaluasi model: cross-validation, threshold tuning, dan metrik."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from .config import N_SPLITS, SEED
from .models import make_model


@dataclass
class CVResult:
    """Ringkasan hasil satu kombinasi model + feature set."""

    model: str
    feature_set: str
    n_features: int
    roc_auc: float
    pr_auc: float
    f1_at_05: float
    best_f1: float
    best_threshold: float
    precision: float
    recall: float
    brier: float
    log_loss_value: float
    fold_auc_mean: float
    fold_auc_std: float
    confusion: list[list[int]] = field(default_factory=list)
    oof_probabilities: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    n_repeats: int = 1
    roc_auc_std: float = 0.0
    pr_auc_std: float = 0.0
    best_f1_std: float = 0.0

    def to_row(self) -> dict[str, Any]:
        """Versi ringkas untuk disimpan sebagai CSV."""
        return {
            "model": self.model,
            "feature_set": self.feature_set,
            "n_features": self.n_features,
            "roc_auc": round(self.roc_auc, 6),
            "pr_auc": round(self.pr_auc, 6),
            "f1_at_0.5": round(self.f1_at_05, 6),
            "best_f1": round(self.best_f1, 6),
            "best_threshold": round(self.best_threshold, 4),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "brier": round(self.brier, 6),
            "log_loss": round(self.log_loss_value, 6),
            "fold_auc_mean": round(self.fold_auc_mean, 6),
            "fold_auc_std": round(self.fold_auc_std, 6),
            "confusion": self.confusion,
            "n_repeats": self.n_repeats,
            "roc_auc_std": round(self.roc_auc_std, 6),
            "pr_auc_std": round(self.pr_auc_std, 6),
            "best_f1_std": round(self.best_f1_std, 6),
        }


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: Callable[..., float] = f1_score,
) -> tuple[float, float]:
    """Cari threshold yang memaksimalkan metrik pada prediksi OOF.

    Threshold dipilih dari probabilitas out-of-fold, sehingga tidak
    menggunakan data validasi secara langsung saat model dilatih.
    """
    candidates = np.unique(
        np.concatenate([np.linspace(0.0, 1.0, 201), np.asarray(y_prob, dtype=float)])
    )
    scores = np.array(
        [
            metric(y_true, (y_prob >= threshold).astype(int), zero_division=0)
            for threshold in candidates
        ]
    )
    best_idx = int(np.argmax(scores))
    return float(candidates[best_idx]), float(scores[best_idx])


def _metrics_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def run_cv(
    model_name: str,
    feature_set: str,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    n_repeats: int = 1,
) -> CVResult:
    """Latih dan evaluasi satu model dengan Stratified K-Fold.

    Mengembalikan probabilitas out-of-fold (OOF), metrik ROC-AUC,
    PR-AUC, F1 pada threshold 0.5 dan threshold yang dioptimalkan.
    Jika ``n_repeats`` > 1, evaluasi diulang dengan seed berbeda dan
    metrik dilaporkan sebagai rata-rata beserta standar deviasinya.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats minimal 1.")

    if n_repeats > 1:
        repeats = [
            run_cv(
                model_name,
                feature_set,
                X,
                y,
                n_splits=n_splits,
                seed=seed + repeat * 1000,
                n_repeats=1,
            )
            for repeat in range(n_repeats)
        ]
        return CVResult(
            model=model_name,
            feature_set=feature_set,
            n_features=repeats[0].n_features,
            roc_auc=float(np.mean([result.roc_auc for result in repeats])),
            pr_auc=float(np.mean([result.pr_auc for result in repeats])),
            f1_at_05=float(np.mean([result.f1_at_05 for result in repeats])),
            best_f1=float(np.mean([result.best_f1 for result in repeats])),
            best_threshold=float(np.median([result.best_threshold for result in repeats])),
            precision=float(np.mean([result.precision for result in repeats])),
            recall=float(np.mean([result.recall for result in repeats])),
            brier=float(np.mean([result.brier for result in repeats])),
            log_loss_value=float(np.mean([result.log_loss_value for result in repeats])),
            fold_auc_mean=float(np.mean([result.fold_auc_mean for result in repeats])),
            fold_auc_std=float(np.mean([result.fold_auc_std for result in repeats])),
            confusion=repeats[0].confusion,
            oof_probabilities=repeats[0].oof_probabilities,
            n_repeats=n_repeats,
            roc_auc_std=float(np.std([result.roc_auc for result in repeats])),
            pr_auc_std=float(np.std([result.pr_auc for result in repeats])),
            best_f1_std=float(np.std([result.best_f1 for result in repeats])),
        )

    X = X.reset_index(drop=True)
    y = np.asarray(y, dtype=int)

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    oof = np.zeros(len(y), dtype=float)
    fold_aucs: list[float] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y)):
        y_train = y[train_idx]
        positive = int(y_train.sum())
        negative = len(y_train) - positive
        pos_weight = negative / max(positive, 1)

        model = make_model(
            model_name,
            seed=seed + fold,
            pos_weight=float(pos_weight),
        )
        model.fit(X.iloc[train_idx], y_train)

        fold_prob = model.predict_proba(X.iloc[valid_idx])[:, 1]
        oof[valid_idx] = fold_prob
        fold_aucs.append(float(roc_auc_score(y[valid_idx], fold_prob)))

    roc_auc = float(roc_auc_score(y, oof))
    pr_auc = float(average_precision_score(y, oof))
    f1_at_05 = float(f1_score(y, (oof >= 0.5).astype(int), zero_division=0))
    best_threshold, best_f1 = find_best_threshold(y, oof)
    best_metrics = _metrics_at_threshold(y, oof, best_threshold)

    return CVResult(
        model=model_name,
        feature_set=feature_set,
        n_features=X.shape[1],
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        f1_at_05=f1_at_05,
        best_f1=best_f1,
        best_threshold=best_threshold,
        precision=float(best_metrics["precision"]),
        recall=float(best_metrics["recall"]),
        brier=float(brier_score_loss(y, oof)),
        log_loss_value=float(log_loss(y, oof, labels=[0, 1])),
        fold_auc_mean=float(np.mean(fold_aucs)),
        fold_auc_std=float(np.std(fold_aucs)),
        confusion=best_metrics["confusion"],
        oof_probabilities=oof,
    )


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: Callable[..., float],
    *,
    threshold: float | None = None,
    n_boot: int = 200,
    seed: int = SEED,
) -> tuple[float, float]:
    """Interval kepercayaan 95% dengan bootstrap untuk satu metrik.

    Jika ``threshold`` diisi, metrik dihitung dari label biner hasil
    threshold tersebut (misalnya F1). Jika tidak, metrik memakai
    probabilitas langsung (misalnya ROC-AUC).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    values: list[float] = []

    for _ in range(n_boot):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        y_boot = y_true[indices]
        if len(np.unique(y_boot)) < 2:
            continue

        if threshold is None:
            value = metric(y_boot, y_prob[indices])
        else:
            value = metric(
                y_boot,
                (y_prob[indices] >= threshold).astype(int),
                zero_division=0,
            )
        values.append(float(value))

    if not values:
        return (float("nan"), float("nan"))

    low, high = np.percentile(values, [2.5, 97.5])
    return (float(low), float(high))


def fit_fold_models(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> list[Any]:
    """Latih satu model per fold untuk prediksi ensemble data uji."""
    X = X.reset_index(drop=True)
    y = np.asarray(y, dtype=int)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    models: list[Any] = []
    for fold, (train_idx, _) in enumerate(splitter.split(X, y)):
        y_train = y[train_idx]
        positive = int(y_train.sum())
        negative = len(y_train) - positive
        pos_weight = negative / max(positive, 1)

        model = make_model(
            model_name,
            seed=seed + fold,
            pos_weight=float(pos_weight),
        )
        model.fit(X.iloc[train_idx], y_train)
        models.append(model)

    return models


def predict_with_ensemble(models: list[Any], X: pd.DataFrame) -> np.ndarray:
    """Rata-rata probabilitas dari seluruh model fold."""
    if not models:
        raise ValueError("Daftar model kosong.")

    probabilities = np.zeros(len(X), dtype=float)
    for model in models:
        probabilities += model.predict_proba(X)[:, 1]
    return probabilities / len(models)
