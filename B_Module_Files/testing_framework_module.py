"""
testing_framework_module.py

Absolute File Path (example):
  /Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/testing_framework_module.py

PURPOSE:
  - Provides an automated testing framework (using unittest).
  - Demonstrates how to test various functionalities such as:
      1) Model loading & prediction (classification_model).
      2) Data preprocessing steps (missing value fill, numeric checks).
      3) Additional modules if needed.

USAGE:
  python "/Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/testing_framework_module.py"

NOTES:
  - Adjust the file paths in setUp() or other tests to match your system.
  - This script focuses on unit tests; integration tests might occur elsewhere.
"""

import os
import unittest
import logging
import numpy as np
import pandas as pd
from joblib import load


class TestClassificationModel(unittest.TestCase):
    """
    Tests for the Titanic classification model (XGBoost) stored in classification_model.joblib.
    """

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Path to the saved classification model (XGBoost)
        self.classification_model_path = (
            "/Users/robertsalinas/Desktop/energy_optimization_project/C_Model_Files/classification_model.joblib"
        )

        # Assert that model file actually exists
        self.assertTrue(
            os.path.exists(self.classification_model_path),
            f"Classification model file not found at {self.classification_model_path}"
        )

        # Load the XGBoost classifier
        self.classifier = load(self.classification_model_path)

    def _encode_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Quick fix to encode 'Sex' and 'Embarked' into numeric columns 
        matching the way the training pipeline encoded them.
        
        If your training pipeline used a different scheme, match it here.
        """
        # Example: map sex => {"male":1, "female":0}
        df["Sex"] = df["Sex"].map({"female": 0, "male": 1}).astype("int")

        # Example: map embarked => {"S":0, "C":1, "Q":2}, fill missing with 0 (if any)
        mapping_embarked = {"S": 0, "C": 1, "Q": 2}
        df["Embarked"] = df["Embarked"].map(mapping_embarked).fillna(0).astype("int")

        # Ensure numeric dtypes
        df["Pclass"] = df["Pclass"].astype(int)
        df["Age"] = df["Age"].astype(float)
        df["SibSp"] = df["SibSp"].astype(int)
        df["Parch"] = df["Parch"].astype(int)
        df["Fare"] = df["Fare"].astype(float)

        return df

    def test_prediction_shape(self):
        """
        Test that the model returns predictions for sample input.
        """
        sample_input = pd.DataFrame({
            "Pclass": [3],
            "Sex": ["female"],
            "Age": [22.0],
            "SibSp": [1],
            "Parch": [0],
            "Fare": [7.25],
            "Embarked": ["S"]
        })

        # Manually encode
        sample_input = self._encode_sample(sample_input)

        y_pred = self.classifier.predict(sample_input)
        self.assertEqual(len(y_pred), 1, "Expected exactly one prediction for single sample input.")

    def test_prediction_values(self):
        """
        Test that the model's predicted value is in [0, 1] for classification (Titanic).
        """
        sample_input = pd.DataFrame({
            "Pclass": [1, 3],
            "Sex": ["male", "female"],
            "Age": [40.0, 20.0],
            "SibSp": [0, 1],
            "Parch": [0, 0],
            "Fare": [70.0, 8.05],
            "Embarked": ["C", "S"]
        })

        # Manually encode
        sample_input = self._encode_sample(sample_input)

        y_pred = self.classifier.predict(sample_input)
        self.logger.info(f"Predicted classes: {y_pred}")

        # Titanic model's classes: 0 or 1
        for pred in y_pred:
            self.assertIn(pred, [0, 1], f"Prediction must be 0 or 1, got {pred}")


class TestDataPreprocessing(unittest.TestCase):
    """
    Tests for data preprocessing logic (example).
    """

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def test_numeric_imputation(self):
        """
        Test numeric imputation logic or median fill (like in Titanic script).
        """
        df_test = pd.DataFrame({
            "Age": [22, 30, np.nan, 45],
            "Fare": [7.25, np.nan, 13.00, 25.8]
        })
        age_median = df_test["Age"].median()
        fare_median = df_test["Fare"].median()

        # Avoid chained assignment by assigning the result of fillna back
        df_test["Age"] = df_test["Age"].fillna(age_median)
        df_test["Fare"] = df_test["Fare"].fillna(fare_median)

        self.assertFalse(df_test["Age"].isnull().any(), "Age still has NaN after median fill.")
        self.assertFalse(df_test["Fare"].isnull().any(), "Fare still has NaN after median fill.")
        self.assertAlmostEqual(df_test["Age"][2], age_median, msg="Median fill not applied correctly for Age.")
        self.assertAlmostEqual(df_test["Fare"][1], fare_median, msg="Median fill not applied correctly for Fare.")

if __name__ == "__main__":
    unittest.main()