"""
training_pipeline_module.py

Absolute File Path:
    /home/ec2-user/energy_optimization_project/B_Module_Files/training_pipeline_module.py

PURPOSE:
  - Provides an end-to-end training pipeline for a classification model using our custom 9-DOF dataset.
  - Loads and preprocesses the custom dataset (classification_9dof_dataset.csv).
  - Trains an XGBoost classifier with hyperparameters found via grid search.
  - Evaluates the model on a validation split.
  - Saves the final trained model as classification_model.joblib to 
    /home/ec2-user/energy_optimization_project/C_Model_Files/classification_model/classification_model.joblib

USAGE:
  python "/home/ec2-user/energy_optimization_project/B_Module_Files/training_pipeline_module.py"
  
NOTES:
  - The custom dataset (classification_9dof_dataset.csv) is expected to contain columns:
      pos_1, pos_2, ..., pos_9, vel_1, ..., vel_9, force_1, ..., force_9, system_efficiency, label, step
  - The 'label' column is used as the target.
  - Hyperparameters used (based on prior grid search) are preset.
  - Final model is saved to:
       /home/ec2-user/energy_optimization_project/C_Model_Files/classification_model/classification_model.joblib
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

# Setup paths for EC2
MODEL_OUTPUT_DIR = "/home/ec2-user/energy_optimization_project/C_Model_Files"
TRAIN_DATA_PATH = "/home/ec2-user/energy_optimization_project/D_Dataset_Files/classification_dataset/classification_9dof_dataset.csv"
TEST_DATA_PATH = None  # Update if a test set is available
MODEL_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, "classification_model", "classification_model.joblib")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainingPipelineModule")

class TrainingPipeline:
    """
    A training pipeline for classification tasks using our custom 9-DOF dataset.
    """

    def __init__(self, 
                 train_csv: str = TRAIN_DATA_PATH, 
                 test_csv: Optional[str] = TEST_DATA_PATH,
                 output_model_path: str = MODEL_SAVE_PATH):
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.output_model_path = output_model_path
        logger.info(f"TrainingPipeline initialized. Train CSV: {self.train_csv}, "
                    f"Test CSV: {self.test_csv}, Output: {self.output_model_path}")

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading training data from {self.train_csv}")
        df_train = pd.read_csv(self.train_csv)
        logger.info(f"Training data loaded. Shape={df_train.shape}")
        return df_train

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing data...")
        if "step" in df.columns:
            df.drop("step", axis=1, inplace=True)
        logger.info("Preprocessing complete.")
        return df

    def split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        if "label" not in df.columns:
            raise ValueError("The dataset does not contain the 'label' column as target.")
        y = df["label"]
        X = df.drop(["label"], axis=1)
        return X, y

    def train_xgb_classifier(self, X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
        logger.info("Training XGBoost classifier with fixed hyperparameters.")
        xgb_clf = XGBClassifier(
            colsample_bytree=0.8,
            learning_rate=0.01,
            max_depth=4,
            min_child_weight=1,
            n_estimators=1000,
            subsample=1.0,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        xgb_clf.fit(X, y, verbose=False)
        logger.info("XGBoost training complete.")
        return xgb_clf

    def evaluate_model(self, model, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        logger.info(f"Validation Accuracy: {acc*100:.2f}%")
        logger.info(f"Classification Report:\n{classification_report(y_val, y_pred, digits=4)}")
        return acc

    def run_training_pipeline(self):
        df_train = self.load_data()
        df_train = self.preprocess_data(df_train)
        X, y = self.split_features_target(df_train)

        unique, counts = np.unique(y, return_counts=True)
        stratify_arg = y if min(counts) >= 2 else None
        if stratify_arg is None:
            logger.warning("One or more classes have fewer than 2 instances; skipping stratification.")

        logger.info("Splitting data into train/validation sets (80/20).")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_arg
        )
        logger.info(f"Train shape={X_train.shape}, Val shape={X_val.shape}")

        model = self.train_xgb_classifier(X_train, y_train)
        accuracy = self.evaluate_model(model, X_val, y_val)
        logger.info(f"Model achieved ~{accuracy*100:.2f}% accuracy on validation set.")

        logger.info(f"Saving model to {self.output_model_path}")
        dump(model, self.output_model_path)
        logger.info("Final classification model saved successfully.")

        if self.test_csv is not None and os.path.exists(self.test_csv):
            logger.info("Evaluating on separate test set.")
            df_test = pd.read_csv(self.test_csv)
            df_test = self.preprocess_data(df_test)
            if "Survived" in df_test.columns:
                X_test, y_test = self.split_features_target(df_test)
                test_acc = self.evaluate_model(model, X_test, y_test)
                logger.info(f"Test Accuracy: {test_acc*100:.2f}%")
            else:
                logger.warning("Test set does not contain a target column. Skipping test evaluation.")

        logger.info("Training pipeline complete.")

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_training_pipeline()
    logger.info(
        "NOTE: The classification model achieved the desired performance. "
        "Model is saved to classification_model.joblib"
    )