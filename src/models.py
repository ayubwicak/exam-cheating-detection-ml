"""Registry model.

Menambah model baru cukup dengan membuat fungsi factory dan
menambahkan decorator ``@register_model("nama_model")``. Pipeline
pelatihan akan otomatis menemukan model tersebut.
"""

from __future__ import annotations

from typing import Any, Callable

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier

from .config import SEED

try:
    from xgboost import XGBClassifier
except ImportError:  # XGBoost bersifat opsional.
    XGBClassifier = None  # type: ignore[assignment]


ModelFactory = Callable[..., Any]
MODEL_REGISTRY: dict[str, ModelFactory] = {}

# Model yang dijalankan saat pengguna memilih --models all.
DEFAULT_MODELS = [
    "logreg",
    "random_forest",
    "extra_trees",
    "hist_gb",
    "lightgbm",
    "xgboost",
]


def register_model(name: str) -> Callable[[ModelFactory], ModelFactory]:
    """Decorator untuk mendaftarkan factory model baru."""

    def decorator(factory: ModelFactory) -> ModelFactory:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' sudah terdaftar.")
        MODEL_REGISTRY[name] = factory
        return factory

    return decorator


def available_models() -> list[str]:
    """Daftar nama model yang sudah terdaftar."""
    return sorted(MODEL_REGISTRY)


def make_model(
    name: str,
    *,
    seed: int = SEED,
    pos_weight: float = 1.0,
    **overrides: Any,
) -> Any:
    """Buat instance model dari registry.

    ``pos_weight`` adalah rasio jumlah kelas negatif / positif. Nilai ini
    dipakai oleh model yang mendukung bobot kelas eksplisit (XGBoost).
    """
    if name not in MODEL_REGISTRY:
        valid = ", ".join(available_models())
        raise ValueError(f"Model '{name}' tidak dikenal. Pilihan: {valid}")

    return MODEL_REGISTRY[name](
        seed=seed,
        pos_weight=pos_weight,
        **overrides,
    )


@register_model("dummy")
def build_dummy(*, seed: int = SEED, pos_weight: float = 1.0, **kwargs: Any) -> DummyClassifier:
    """Baseline sanity-check: selalu memprediksi proporsi kelas."""
    return DummyClassifier(strategy="prior")


@register_model("logreg")
def build_logreg(
    *, seed: int = SEED, pos_weight: float = 1.0, **kwargs: Any
) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
            **kwargs,
        ),
    )


@register_model("random_forest")
def build_random_forest(
    *, seed: int = SEED, pos_weight: float = 1.0, **kwargs: Any
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
        **kwargs,
    )


@register_model("extra_trees")
def build_extra_trees(
    *, seed: int = SEED, pos_weight: float = 1.0, **kwargs: Any
) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
        **kwargs,
    )


@register_model("hist_gb")
def build_hist_gb(
    *, seed: int = SEED, pos_weight: float = 1.0, **kwargs: Any
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=seed,
        **kwargs,
    )


@register_model("lightgbm")
def build_lightgbm(
    *, seed: int = SEED, pos_weight: float = 1.0, **kwargs: Any
) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
        verbose=-1,
        **kwargs,
    )


@register_model("xgboost")
def build_xgboost(
    *, seed: int = SEED, pos_weight: float = 1.0, **kwargs: Any
) -> Any:
    if XGBClassifier is None:
        raise ImportError(
            "XGBoost belum terpasang. Jalankan: pip install xgboost"
        )

    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=max(pos_weight, 1.0),
        eval_metric="logloss",
        n_jobs=-1,
        random_state=seed,
        **kwargs,
    )
