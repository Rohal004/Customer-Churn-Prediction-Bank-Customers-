"""Train a churn prediction model for bank customers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "Exited"
DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
DEFAULT_CATEGORICAL = ["Geography", "Gender"]


def test_size_arg(value: str) -> float:
    size = float(value)
    if not 0 < size < 1:
        raise argparse.ArgumentTypeError("--test-size must be between 0 and 1 (exclusive)")
    return size


def load_and_prepare_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset and split into feature and target sets."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)

    missing_target = TARGET_COLUMN not in df.columns
    if missing_target:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in dataset")

    cleaned_df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns]).copy()

    y = cleaned_df[TARGET_COLUMN]
    X = cleaned_df.drop(columns=[TARGET_COLUMN])
    return X, y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Create preprocessing + model pipeline."""
    available_cats = [c for c in DEFAULT_CATEGORICAL if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in available_cats]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                available_cats,
            ),
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
    )

    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def get_feature_importance(model_pipeline: Pipeline) -> pd.Series:
    """Extract and rank feature importances from the trained pipeline."""
    preprocess = model_pipeline.named_steps["preprocess"]
    model = model_pipeline.named_steps["model"]

    feature_names = preprocess.get_feature_names_out()
    importances = model.feature_importances_

    feature_importance = pd.Series(importances, index=feature_names)
    return feature_importance.sort_values(ascending=False)


def train_and_evaluate(data_path: Path, test_size: float) -> None:
    """Run model training, evaluation, and feature importance analysis."""
    X, y = load_and_prepare_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline(X)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("Classification report:\n")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}\n")

    print("Top 10 feature importances:\n")
    print(get_feature_importance(pipeline).head(10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a bank churn classifier")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("./Churn_Modelling.csv"),
        help="Path to the churn CSV file (default: ./Churn_Modelling.csv)",
    )
    parser.add_argument(
        "--test-size",
        type=test_size_arg,
        default=0.2,
        help="Fraction of data to reserve for test split (default: 0.2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_evaluate(data_path=args.data, test_size=args.test_size)


if __name__ == "__main__":
    main()