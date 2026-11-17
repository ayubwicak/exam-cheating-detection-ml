"""Turnamen model: bandingkan model dan feature set dengan CV.

Contoh pemakaian:
    python -m src.train --models all --feature-sets A,B,C
    python -m src.train --models lightgbm,xgboost --feature-sets C
    python -m src.train --models catboost --feature-sets C

Hasil yang dibuat di folder output/:
    - tournament_results.csv
    - best_model_bundle.pkl
    - best_model_metrics.json
    - submission_tournament.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from .config import (
    ID_COLUMN,
    N_SPLITS,
    OUTPUT_DIR,
    SEED,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
    ensure_output_dir,
)
from .evaluation import (
    CVResult,
    bootstrap_ci,
    fit_fold_models,
    predict_with_ensemble,
    run_cv,
)
from .features import FEATURE_SETS, build_feature_frame
from .models import DEFAULT_MODELS, available_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turnamen model deteksi kecurangan ujian.",
    )
    parser.add_argument(
        "--models",
        default="all",
        help=(
            "Daftar model dipisahkan koma, atau 'all'. "
            f"Tersedia: {', '.join(available_models())}"
        ),
    )
    parser.add_argument(
        "--feature-sets",
        default="A,B,C",
        help="Daftar feature set dipisahkan koma, atau 'all'.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=N_SPLITS,
        help="Jumlah fold Stratified K-Fold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed untuk split dan model.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Jumlah pengulangan CV dengan seed berbeda (misalnya 3 atau 5).",
    )
    parser.add_argument(
        "--select-metric",
        choices=["pr_auc", "roc_auc", "best_f1"],
        default="pr_auc",
        help="Metrik untuk memilih model terbaik.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Jumlah resample bootstrap untuk confidence interval (0 = nonaktif).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Folder untuk menyimpan hasil.",
    )
    parser.add_argument(
        "--no-submission",
        action="store_true",
        help="Jangan buat file submission.",
    )
    return parser.parse_args()


def resolve_models(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        names = list(DEFAULT_MODELS)
    else:
        names = [name.strip() for name in spec.split(",") if name.strip()]

    unknown = [name for name in names if name not in available_models()]
    if unknown:
        raise SystemExit(
            f"Model tidak dikenal: {unknown}. Pilihan: {available_models()}"
        )
    if not names:
        raise SystemExit("Daftar model kosong.")
    return names


def resolve_feature_sets(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return sorted(FEATURE_SETS)

    names = [name.strip().upper() for name in spec.split(",") if name.strip()]
    unknown = [name for name in names if name not in FEATURE_SETS]
    if unknown:
        raise SystemExit(
            f"Feature set tidak dikenal: {unknown}. Pilihan: {sorted(FEATURE_SETS)}"
        )
    if not names:
        raise SystemExit("Daftar feature set kosong.")
    return names


def print_results(results: pd.DataFrame) -> None:
    columns = [
        "model",
        "feature_set",
        "n_features",
        "roc_auc",
        "roc_auc_std",
        "pr_auc",
        "pr_auc_std",
        "f1_at_0.5",
        "best_f1",
        "best_f1_std",
        "best_threshold",
        "precision",
        "recall",
    ]
    print("\n" + "=" * 100)
    print("HASIL TURNAMEN (diurutkan berdasarkan metrik pilihan)")
    print("=" * 100)
    print(results[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    model_names = resolve_models(args.models)
    feature_sets = resolve_feature_sets(args.feature_sets)

    print("=" * 100)
    print("MEMUAT DATA")
    print("=" * 100)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"Train: {train.shape} | Test: {test.shape}")

    if TARGET not in train.columns:
        raise SystemExit(f"Kolom target '{TARGET}' tidak ditemukan di data train.")
    if ID_COLUMN not in test.columns:
        raise SystemExit(f"Kolom ID '{ID_COLUMN}' tidak ditemukan di data test.")

    y = train[TARGET].to_numpy(dtype=int)
    print(f"Distribusi target: {dict(pd.Series(y).value_counts())}")
    print(f"Model: {model_names}")
    print(f"Feature set: {feature_sets}")
    print(f"Fold: {args.folds} | Repeat: {args.repeats} | Seed: {args.seed}")

    results: list[CVResult] = []
    for feature_set in feature_sets:
        X = build_feature_frame(train, feature_set)
        print(f"\n[{feature_set}] {X.shape[1]} fitur")

        for model_name in model_names:
            print(f"  → {model_name} ...", end=" ", flush=True)
            try:
                result = run_cv(
                    model_name,
                    feature_set,
                    X,
                    y,
                    n_splits=args.folds,
                    seed=args.seed,
                    n_repeats=args.repeats,
                )
            except ImportError as exc:
                print(f"dilewati ({exc})")
                continue

            results.append(result)
            print(
                f"ROC-AUC={result.roc_auc:.4f}±{result.roc_auc_std:.4f} "
                f"PR-AUC={result.pr_auc:.4f}±{result.pr_auc_std:.4f} "
                f"best-F1={result.best_f1:.4f}±{result.best_f1_std:.4f}"
            )

    if not results:
        raise SystemExit("Tidak ada model yang berhasil dijalankan.")

    results_df = pd.DataFrame([result.to_row() for result in results])
    results_df = results_df.sort_values(
        by=[args.select_metric, "best_f1"],
        ascending=False,
    ).reset_index(drop=True)

    results_path = output_dir / "tournament_results.csv"
    results_df.to_csv(results_path, index=False)
    print_results(results_df)
    print(f"\nHasil turnamen disimpan di: {results_path}")

    best_result = max(results, key=lambda result: getattr(result, args.select_metric))
    print("\n" + "=" * 100)
    print(
        f"MODEL TERBAIK: {best_result.model} | feature set {best_result.feature_set} "
        f"| {args.select_metric}={getattr(best_result, args.select_metric):.4f}"
    )
    print("=" * 100)

    best_features = build_feature_frame(train, best_result.feature_set)
    best_models = fit_fold_models(
        best_result.model,
        best_features,
        y,
        n_splits=args.folds,
        seed=args.seed,
    )

    bundle: dict[str, Any] = {
        "models": best_models,
        "features": list(best_features.columns),
        "threshold": best_result.best_threshold,
        "model_name": best_result.model,
        "feature_set": best_result.feature_set,
        "metrics": best_result.to_row(),
        "seed": args.seed,
        "target": TARGET,
        "id_column": ID_COLUMN,
        "feature_engineering": "src.features.add_domain_features",
    }

    if args.bootstrap > 0:
        auc_low, auc_high = bootstrap_ci(
            y,
            best_result.oof_probabilities,
            roc_auc_score,
            n_boot=args.bootstrap,
            seed=args.seed,
        )
        f1_low, f1_high = bootstrap_ci(
            y,
            best_result.oof_probabilities,
            f1_score,
            threshold=best_result.best_threshold,
            n_boot=args.bootstrap,
            seed=args.seed,
        )
        bundle["bootstrap_ci"] = {
            "roc_auc_95": [auc_low, auc_high],
            "best_f1_95": [f1_low, f1_high],
            "n_boot": args.bootstrap,
        }
        print(
            f"Bootstrap 95% CI ROC-AUC: [{auc_low:.4f}, {auc_high:.4f}] | "
            f"F1: [{f1_low:.4f}, {f1_high:.4f}]"
        )

    bundle_path = output_dir / "best_model_bundle.pkl"
    joblib.dump(bundle, bundle_path)
    print(f"Bundle model disimpan di: {bundle_path}")

    metrics_path = output_dir / "best_model_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(bundle["metrics"], file, indent=2, ensure_ascii=False)
    print(f"Metrik model disimpan di: {metrics_path}")

    if not args.no_submission:
        X_test = build_feature_frame(test, best_result.feature_set)
        test_probabilities = predict_with_ensemble(best_models, X_test)
        submission = pd.DataFrame(
            {
                ID_COLUMN: test[ID_COLUMN],
                TARGET: (test_probabilities >= best_result.best_threshold).astype(int),
            }
        )
        submission_path = output_dir / "submission_tournament.csv"
        submission.to_csv(submission_path, index=False)
        print(f"Submission disimpan di: {submission_path}")
        print(f"Distribusi prediksi: {dict(submission[TARGET].value_counts())}")

    print("\nSelesai.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
