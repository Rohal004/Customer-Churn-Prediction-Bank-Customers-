from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class ChurnModelResult:
    accuracy: float
    classification_report_text: str
    feature_importance: pd.DataFrame
    model: Pipeline


def clean_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    for column in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[column] = cleaned[column].astype(str).str.strip()

    return cleaned


def build_pipeline(features: pd.DataFrame) -> Pipeline:
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
                    n_estimators=300,
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
) -> ChurnModelResult:
    cleaned = clean_dataset(dataframe)
    if target_column not in cleaned.columns:
        raise ValueError(f"Target column '{target_column}' is missing from the dataset.")

    prepared = cleaned.drop(columns=["RowNumber", "CustomerId", "Surname"], errors="ignore")

    X = prepared.drop(columns=[target_column])
    y = prepared[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_pipeline(X)
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
    )


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
    args = parser.parse_args()

    dataframe = pd.read_csv(args.data_path)
    result = train_churn_model(dataframe, target_column=args.target)

    print(f"Accuracy: {result.accuracy:.4f}")
    print("\nClassification report:\n")
    print(result.classification_report_text)
    print(f"\nTop {args.top_n} feature importances:\n")
    print(result.feature_importance.head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
