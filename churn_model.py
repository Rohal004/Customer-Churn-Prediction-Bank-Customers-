from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DEFAULT_DROP_COLUMNS = ("RowNumber", "CustomerId", "Surname")


@dataclass
class ChurnModelResult:
    accuracy: float
    classification_report_text: str
    feature_importance: pd.DataFrame
    model: Pipeline
    cleaned_rows: int
    dropped_duplicate_rows: int
    target_distribution: pd.Series


def clean_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    for column in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[column] = cleaned[column].astype(str).str.strip()

    return cleaned


def build_pipeline(features: pd.DataFrame, n_estimators: int = 100) -> Pipeline:
    numeric_columns = features.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_columns = features.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train_churn_model(
    dataframe: pd.DataFrame,
    target_column: str = "Exited",
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    drop_columns: tuple[str, ...] = DEFAULT_DROP_COLUMNS,
) -> ChurnModelResult:
    original_row_count = len(dataframe)
    cleaned = clean_dataset(dataframe)
    if target_column not in cleaned.columns:
        raise ValueError(f"Target column '{target_column}' is missing from the dataset.")

    prepared = cleaned.drop(columns=list(drop_columns), errors="ignore")

    X = prepared.drop(columns=[target_column])
    y = prepared[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_pipeline(X, n_estimators=n_estimators)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    feature_importance = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return ChurnModelResult(
        accuracy=accuracy,
        classification_report_text=report,
        feature_importance=feature_importance,
        model=model,
        cleaned_rows=len(cleaned),
        dropped_duplicate_rows=original_row_count - len(cleaned),
        target_distribution=y.value_counts().sort_index(),
    )


def format_target_distribution(target_distribution: pd.Series) -> str:
    lines = ["Target distribution:"]
    total = int(target_distribution.sum())
    for label, count in target_distribution.items():
        percentage = (count / total) * 100 if total else 0.0
        lines.append(f"  {label}: {count} ({percentage:.1f}%)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a bank customer churn classifier.")
    parser.add_argument("data_path", help="Path to the churn modelling CSV file.")
    parser.add_argument(
        "--target",
        default="Exited",
        help="Target column name in dataset (default: Exited).",
    )
    parser.add_argument(
        "--top-n",
        default=10,
        type=int,
        help="Number of top feature importances to display.",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        help="Optional path to save the trained pipeline as a pickle file.",
    )
    args = parser.parse_args()

    dataframe = pd.read_csv(args.data_path)
    result = train_churn_model(dataframe, target_column=args.target)

    print(f"Rows loaded: {len(dataframe)}")
    print(f"Rows after cleaning: {result.cleaned_rows}")
    print(f"Duplicate rows removed: {result.dropped_duplicate_rows}")
    print()
    print(f"Accuracy: {result.accuracy:.4f}")
    print("\nClassification report:\n")
    print(result.classification_report_text)
    print()
    print(format_target_distribution(result.target_distribution))
    print(f"\nTop {args.top_n} feature importances:\n")
    print(result.feature_importance.head(args.top_n).to_string(index=False))

    if args.save_model:
        import pickle

        with args.save_model.open("wb") as model_file:
            pickle.dump(result.model, model_file)
        print(f"\nSaved trained model to: {args.save_model}")


if __name__ == "__main__":
    main()
