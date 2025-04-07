"""
energy_efficiency_optimization_model.py

DESCRIPTION:
    This script defines a regression-based model for optimizing energy efficiency.
    It uses data from multiple sensors (e.g., temperature, power consumption, load) 
    to predict and recommend optimized robotic or system settings that reduce 
    overall energy usage while maintaining operational performance.

KEY FEATURES:
    1. Flexible Input Dimensions:
       - Can handle any number of sensors/features. Includes dimension checks.
    2. Advanced Ensemble Regressor:
       - Uses a GradientBoostingRegressor by default for robust generalization.
       - Easily extendable for hyperparameter tuning or alternative regressors.
    3. Scaler Support:
       - Optionally uses a StandardScaler to normalize numeric inputs.
    4. Evaluation and Utility Methods:
       - Evaluate the model using regression metrics (RMSE, R^2).
       - Save and load the entire regression pipeline (model + scaler) using joblib.
    5. Future-Proofing:
       - Clear class-based architecture for integration with other modules.
       - Easily extendable for new sensors or changes in data distribution.

USAGE EXAMPLE (In Another Script):
    from energy_efficiency_optimization_model import EnergyEfficiencyModel

    model = EnergyEfficiencyModel(
        model_path="/path/to/energy_efficiency_pipeline.joblib"
    )

    # If a pre-trained pipeline exists, it loads automatically.
    # Otherwise, build and train a new model:
    # model.build_model(use_scaler=True, n_estimators=200, max_depth=6, learning_rate=0.05)
    # model.fit(X_train, y_train)
    
    # Single-sample inference:
    sample_features = [22.5, 0.87, 145.0, 1012.3]  # example sensor readings
    pred = model.predict_settings(sample_features)
    print("Recommended energy setting:", pred)

    # Batch inference:
    batch_features = [
        [22.5, 0.87, 145.0, 1012.3],
        [24.0, 0.90, 160.0, 1015.2],
    ]
    batch_preds = model.batch_predict(batch_features)
    print("Batch predictions:", batch_preds)
"""

import os
import joblib
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

###############################################################################
# DEFAULT PATHS (can override via environment variables or pass explicitly)
###############################################################################
DEFAULT_PIPELINE_PATH = os.getenv("EE_PIPELINE_PATH", "energy_efficiency_pipeline.joblib")

class EnergyEfficiencyModel:
    """
    A class encapsulating the regression pipeline for optimizing energy efficiency.
    This model predicts an "optimal energy setting" (continuous value) based on 
    multi-sensor inputs using a GradientBoostingRegressor.

    The entire pipeline (model + scaler) is saved/loaded as a single file for ease of deployment.
    """

    def __init__(self, pipeline_path: str = DEFAULT_PIPELINE_PATH):
        """
        Constructor tries to load an existing pipeline (model and scaler) from disk.
        If not found, self.model and self.scaler remain None until build_model() is called.
        """
        self.pipeline_path = pipeline_path
        self.model: Optional[RegressorMixin] = None
        self.scaler: Optional[TransformerMixin] = None
        self.expected_dim: Optional[int] = None

        if os.path.exists(self.pipeline_path):
            self.load_pipeline(self.pipeline_path)
        else:
            print(f"[WARNING] No pipeline found at {self.pipeline_path}; pipeline not loaded.")

    def build_model(self, use_scaler: bool = True, **kwargs):
        """
        Build a new GradientBoostingRegressor for energy efficiency prediction.
        Optionally initializes a StandardScaler if use_scaler=True.

        :param use_scaler: Whether to integrate a StandardScaler for features.
        :param kwargs: Additional hyperparameters for GradientBoostingRegressor 
                       (e.g., n_estimators=200, learning_rate=0.05, max_depth=6, etc.)
        """
        self.model = GradientBoostingRegressor(
            n_estimators=kwargs.get("n_estimators", 150),
            learning_rate=kwargs.get("learning_rate", 0.1),
            max_depth=kwargs.get("max_depth", 5),
            random_state=42
        )

        self.scaler = StandardScaler() if use_scaler else None

        print("[INFO] Energy Efficiency Model built with GradientBoostingRegressor.")
        print(f"[INFO] Hyperparameters: {self.model.get_params()}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the pipeline (scaler and model) on the training data.
        :param X: 2D array of shape (n_samples, n_features)
        :param y: 1D array of shape (n_samples,)
        """
        self.expected_dim = X.shape[1]
        if self.scaler:
            X = self.scaler.fit_transform(X)
        if self.model is None:
            print("[ERROR] Model not built or loaded. Call build_model() first.")
            return
        self.model.fit(X, y)
        print("[INFO] Energy Efficiency Model training complete.")

    def predict_settings(self, data: Union[List[float], np.ndarray]) -> float:
        """
        Predict the optimal energy setting for a single sample.
        :param data: 1D list/array of sensor readings.
        :return: A float representing the predicted energy setting.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded or built.")
        arr = np.array(data).reshape(1, -1)
        if self.expected_dim is not None and arr.shape[1] != self.expected_dim:
            print(f"[WARNING] Feature mismatch: expected {self.expected_dim}, got {arr.shape[1]}")
        if self.scaler:
            arr = self.scaler.transform(arr)
        prediction = self.model.predict(arr)[0]
        return float(prediction)

    def batch_predict(self, data_matrix: Union[List[List[float]], np.ndarray]) -> List[float]:
        """
        Predict optimal energy settings for a batch of samples.
        :param data_matrix: 2D array/list of shape (n_samples, n_features)
        :return: List of float predictions.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded or built.")
        mat = np.array(data_matrix)
        if self.expected_dim is not None and mat.shape[1] != self.expected_dim:
            print(f"[WARNING] Feature mismatch: expected {self.expected_dim}, got {mat.shape[1]}")
        if self.scaler:
            mat = self.scaler.transform(mat)
        preds = self.model.predict(mat)
        return preds.tolist()

    def evaluate_model(self, X: np.ndarray, y_true: np.ndarray):
        """
        Evaluate the model on a labeled dataset, printing out RMSE and R^2 metrics.
        :param X: Feature matrix.
        :param y_true: True regression targets.
        """
        if self.model is None:
            print("[ERROR] Model not loaded or built.")
            return
        if self.scaler:
            X = self.scaler.transform(X)
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        print("[INFO] Model Evaluation:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R^2:  {r2:.4f}")

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        """
        Attempt partial fit if supported. For GradientBoostingRegressor, partial_fit is not available.
        """
        if self.model is None:
            print("[ERROR] No model available for partial fit.")
            return
        if not hasattr(self.model, "partial_fit"):
            print("[WARNING] The current regressor does not support partial_fit.")
            return
        # If supported, one might do:
        # X_scaled = self.scaler.transform(X) if self.scaler else X
        # self.model.partial_fit(X_scaled, y)
        pass

    def save_pipeline(self, pipeline_path: Optional[str] = None):
        """
        Save the entire pipeline (model and scaler) to disk as a single file.
        :param pipeline_path: Destination path. Defaults to self.pipeline_path.
        """
        if pipeline_path is None:
            pipeline_path = self.pipeline_path
        if self.model is None:
            print("[WARNING] No model to save.")
            return
        try:
            pipeline = {"model": self.model, "scaler": self.scaler}
            joblib.dump(pipeline, pipeline_path)
            print(f"[INFO] Pipeline saved to {pipeline_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save pipeline: {e}")

    def load_pipeline(self, pipeline_path: str):
        """
        Load the entire pipeline (model and scaler) from disk.
        :param pipeline_path: Path to the saved pipeline file.
        """
        try:
            pipeline = joblib.load(pipeline_path)
            self.model = pipeline.get("model", None)
            self.scaler = pipeline.get("scaler", None)
            print(f"[INFO] Pipeline loaded from: {pipeline_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load pipeline: {e}")
            self.model = None
            self.scaler = None

# Example usage (commented out):
# if __name__ == "__main__":
#     # Generate some dummy training data for demonstration:
#     X_train = np.random.rand(100, 4)
#     y_train = np.random.rand(100) * 50
#
#     eef_model = EnergyEfficiencyModel(
#         pipeline_path="energy_efficiency_pipeline.joblib"
#     )
#
#     eef_model.build_model(use_scaler=True, n_estimators=200, max_depth=6, learning_rate=0.05)
#     eef_model.fit(X_train, y_train)
#     eef_model.evaluate_model(X_train, y_train)
#
#     # Save the pipeline:
#     eef_model.save_pipeline()
#
#     # Predict on a new sample:
#     sample = [22.5, 0.87, 145.0, 1012.3]
#     print("Prediction:", eef_model.predict_settings(sample))