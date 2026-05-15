# Customer-Churn-Prediction-Bank-Customers-

Identify customers who are likely to leave the bank.

## What this project now does

- Cleans and prepares a bank churn dataset.
- Encodes categorical features such as `Geography` and `Gender`.
- Trains a churn classification model.
- Prints feature importance scores to show what influences churn.

## Expected dataset

The training script expects a CSV file with an `Exited` target column.
Typical columns include: `RowNumber`, `CustomerId`, `Surname`, `Geography`, `Gender`, and numeric account features.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python3 churn_model.py --data ./Churn_Modelling.csv
```

Optional argument:

```bash
python3 churn_model.py --data ./Churn_Modelling.csv --test-size 0.25
```
