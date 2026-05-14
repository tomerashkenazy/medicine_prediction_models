"""Shared preprocessing for every model."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import CATEGORICAL_CODE_FEATURES


def split_feature_types(X) -> tuple[list[str], list[str], list[str]]:
    object_categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_categorical = [
        column
        for column in CATEGORICAL_CODE_FEATURES
        if column in X.columns and column not in object_categorical
    ]
    categorical = set(object_categorical + numeric_categorical)
    numeric = [column for column in X.columns if column not in categorical]
    return numeric, object_categorical, numeric_categorical


def build_preprocessor(X) -> ColumnTransformer:
    numeric_features, object_categorical_features, numeric_categorical_features = (
        split_feature_types(X)
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    object_categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    numeric_categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    transformers = [("numeric", numeric_pipeline, numeric_features)]
    if object_categorical_features:
        transformers.append(
            ("object_categorical", object_categorical_pipeline, object_categorical_features)
        )
    if numeric_categorical_features:
        transformers.append(
            ("numeric_categorical", numeric_categorical_pipeline, numeric_categorical_features)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )
