import unittest

import pandas as pd

from churn_model import train_churn_model


class TestChurnModel(unittest.TestCase):
    def test_training_and_feature_importance(self):
        rows = []
        for i in range(30):
            rows.append(
                {
                    "RowNumber": i + 1,
                    "CustomerId": 100000 + i,
                    "Surname": f"Customer{i}",
                    "CreditScore": 650 + (i % 40),
                    "Geography": ["France", "Spain", "Germany"][i % 3],
                    "Gender": "Female" if i % 2 == 0 else "Male",
                    "Age": 30 + (i % 20),
                    "Tenure": i % 10,
                    "Balance": 0 if i % 4 == 0 else 10000 + i * 50,
                    "NumOfProducts": 1 + (i % 3),
                    "HasCrCard": i % 2,
                    "IsActiveMember": 1 if i % 5 else 0,
                    "EstimatedSalary": 50000 + i * 1000,
                    "Exited": 1 if i % 6 in (0, 1) else 0,
                }
            )
        dataframe = pd.DataFrame(rows)

        result = train_churn_model(dataframe)

        self.assertGreater(result.accuracy, 0.3)
        self.assertLessEqual(result.accuracy, 1.0)
        self.assertFalse(result.feature_importance.empty)
        self.assertIn("feature", result.feature_importance.columns)
        self.assertIn("importance", result.feature_importance.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(result.feature_importance["importance"]))
        self.assertTrue((result.feature_importance["importance"] >= 0).all())
        joined_features = " ".join(result.feature_importance["feature"].tolist())
        self.assertIn("cat__Geography_", joined_features)
        self.assertIn("cat__Gender_", joined_features)


if __name__ == "__main__":
    unittest.main()
