"""Data loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRAINING_PATH = DATA_DIR / "training_v2.csv" / "training_v2.csv"
UNLABELED_PATH = DATA_DIR / "unlabeled.csv" / "unlabeled.csv"
SOLUTION_TEMPLATE_PATH = DATA_DIR / "solution_template.csv"
DICTIONARY_PATH = DATA_DIR / "WiDS Datathon 2020 Dictionary.csv"


def load_training_data(path: Path = TRAINING_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_unlabeled_data(path: Path = UNLABELED_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_solution_template(path: Path = SOLUTION_TEMPLATE_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_dictionary(path: Path = DICTIONARY_PATH) -> pd.DataFrame:
    return pd.read_csv(path)

