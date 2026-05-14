"""Generate mortality predictions from a saved model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.data_loading import load_solution_template, load_unlabeled_data
from src.features import ID_COLUMN, TARGET_COLUMN, get_raw_feature_columns


def predict_unlabeled(model_path: Path, output_path: Path) -> None:
    model = joblib.load(model_path)
    unlabeled_df = load_unlabeled_data()
    feature_columns = get_raw_feature_columns(unlabeled_df.columns)
    probabilities = model.predict_proba(unlabeled_df[feature_columns])[:, 1]
    predictions = pd.DataFrame(
        {
            ID_COLUMN: unlabeled_df[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: probabilities,
        }
    )
    template = load_solution_template()[[ID_COLUMN]]
    predictions = template.merge(predictions, on=ID_COLUMN, how="left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_unlabeled(args.model_path, args.output_path)
