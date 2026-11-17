"""Inference dari bundle model yang disimpan oleh src.train."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import ID_COLUMN, OUTPUT_DIR, TARGET, TEST_PATH
from .evaluation import predict_with_ensemble
from .features import add_domain_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Buat submission dari bundle model.")
    parser.add_argument(
        "--input",
        type=Path,
        default=TEST_PATH,
        help="File CSV yang akan diprediksi.",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=OUTPUT_DIR / "best_model_bundle.pkl",
        help="Path bundle hasil src.train.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "submission.csv",
        help="Path file submission hasil prediksi.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override threshold dari bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.bundle.exists():
        raise SystemExit(
            f"Bundle tidak ditemukan: {args.bundle}. "
            "Jalankan `python -m src.train` terlebih dahulu."
        )

    bundle = joblib.load(args.bundle)
    data = pd.read_csv(args.input)
    featured = add_domain_features(data)

    features = bundle["features"]
    missing = [column for column in features if column not in featured.columns]
    if missing:
        raise SystemExit(f"Kolom fitur tidak ditemukan di input: {missing}")

    probabilities = predict_with_ensemble(bundle["models"], featured[features])
    threshold = args.threshold if args.threshold is not None else bundle["threshold"]
    predictions = (probabilities >= threshold).astype(int)

    id_column = bundle.get("id_column", ID_COLUMN)
    if id_column not in data.columns:
        raise SystemExit(f"Kolom ID '{id_column}' tidak ditemukan di input.")

    submission = pd.DataFrame(
        {
            id_column: data[id_column],
            bundle.get("target", TARGET): predictions,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)

    print(f"Model      : {bundle['model_name']} ({bundle['feature_set']})")
    print(f"Threshold  : {threshold:.4f}")
    print(f"Prediksi   : {dict(submission.iloc[:, 1].value_counts())}")
    print(f"Submission : {args.output}")


if __name__ == "__main__":
    main()
