# Customer-Churn-Prediction-Bank-Customers-

Identify customers who are likely to leave the bank.

## What this project now includes

- Dataset cleaning and preparation (duplicate removal, whitespace cleanup, robust missing-value handling).
- Categorical feature encoding for fields such as `Geography` and `Gender` via one-hot encoding.
- Supervised churn classification using `RandomForestClassifier`.
- Feature-importance analysis to explain churn drivers.

## Install

```bash
pip install -r requirements.txt
```

## Train and inspect churn model

```bash
python churn_model.py /path/to/Churn_Modelling.csv
```

Optional arguments:

- `--target` to use a different target column (default `Exited`)
- `--top-n` to control how many top feature importances are shown
- `--save-model /path/to/model.pkl` to save the trained pipeline as a pickle file

When the script runs, it prints:

- how many rows were loaded
- how many rows remain after cleaning
- how many duplicate rows were removed
- the target-class distribution
- accuracy, the classification report, and the top feature importances

Example:

```bash
python churn_model.py /path/to/Churn_Modelling.csv --top-n 15 --save-model churn_model.pkl
```

## Run focused tests

```bash
python -m unittest discover -s tests -q
```
