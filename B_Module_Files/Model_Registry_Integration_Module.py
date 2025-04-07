"""
Model_Registry_Integration_Module.py
Absolute File Path:
  /Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/Model_Registry_Integration_Module.py

PURPOSE:
  - Integrate with a model registry for versioning models, storing metadata
    (parameters, metrics, tags, etc.), and retrieving them.
  - For demonstration, stores registry data in a local JSON file.

NOTES:
  - If you use a real MLOps platform (MLflow, Sagemaker, etc.), you'd adapt the
    logic to call those services' APIs instead of using a local JSON file.

"""

import logging
import json
import os
from typing import Dict, Any, Optional, List

class ModelRegistryIntegration:
    """
    A class for registering models with a JSON-based local "registry".
    Each model entry may include:
      - name
      - version
      - path
      - metrics (accuracy, f1, etc.)
      - parameters (optional hyperparams)
      - tags (arbitrary dictionary)
    """

    def __init__(self, registry_file: str = "model_registry.json"):
        """
        :param registry_file: Path to the JSON file holding registry data.
        """
        self.logger = logging.getLogger("ModelRegistryIntegration")
        self.registry_file = registry_file

        # If the file doesn't exist, create an empty structure
        if not os.path.exists(registry_file):
            with open(registry_file, "w") as f:
                json.dump({"models": []}, f)
            self.logger.info(f"Created new registry file at {registry_file}")
        else:
            self.logger.info(f"Using existing registry file at {registry_file}")

    def load_registry(self) -> Dict[str, Any]:
        """Loads the entire registry JSON into a dictionary."""
        with open(self.registry_file, "r") as f:
            data = json.load(f)
        return data

    def save_registry(self, data: Dict[str, Any]) -> None:
        """Saves the dictionary back into the registry JSON file."""
        with open(self.registry_file, "w") as f:
            json.dump(data, f, indent=4)

    def register_model(
        self,
        model_name: str,
        version: str,
        path: str,
        metrics: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a new model entry to the registry (or update existing version).
        
        :param model_name: Name of the model, e.g. "classification_model"
        :param version: Model version, e.g. "v1.0.0"
        :param path: Filesystem or remote path to the model file
        :param metrics: Dictionary of performance metrics, e.g. {"accuracy": 0.92}
        :param parameters: Dictionary of parameters/hyperparams, e.g. {"max_depth": 10}
        :param tags: Arbitrary key-value pairs (e.g., "env": "dev", "team": "NLP")
        """
        reg_data = self.load_registry()

        # Check if model + version already exists
        existing_index = None
        for i, entry in enumerate(reg_data["models"]):
            if entry["name"] == model_name and entry["version"] == version:
                existing_index = i
                break

        model_info = {
            "name": model_name,
            "version": version,
            "path": path,
            "metrics": metrics or {},
            "parameters": parameters or {},
            "tags": tags or {},
        }

        if existing_index is not None:
            # Update existing entry
            reg_data["models"][existing_index] = model_info
            self.logger.info(
                f"Updated existing model: {model_name} v{version} at {path}"
            )
        else:
            # Append new entry
            reg_data["models"].append(model_info)
            self.logger.info(
                f"Registered new model: {model_name} v{version} at {path}"
            )

        self.save_registry(reg_data)

    def get_model_info(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific model entry by (name, version), or None if not found.
        """
        reg_data = self.load_registry()
        for entry in reg_data["models"]:
            if entry["name"] == model_name and entry["version"] == version:
                return entry
        self.logger.warning(
            f"Model {model_name} v{version} not found in registry."
        )
        return None

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Return a list of all models in the registry.
        """
        reg_data = self.load_registry()
        return reg_data.get("models", [])

    def remove_model(self, model_name: str, version: str) -> bool:
        """
        Remove a model entry from the registry. Returns True if removed, else False.
        """
        reg_data = self.load_registry()
        original_len = len(reg_data["models"])
        reg_data["models"] = [
            m
            for m in reg_data["models"]
            if not (m["name"] == model_name and m["version"] == version)
        ]
        new_len = len(reg_data["models"])
        self.save_registry(reg_data)

        removed = (new_len < original_len)
        if removed:
            self.logger.info(
                f"Removed model {model_name} v{version} from registry."
            )
        else:
            self.logger.warning(
                f"Attempted to remove non-existent model {model_name} v{version}."
            )
        return removed

    def find_best_model(self, model_name: str, metric_key: str = "accuracy") -> Optional[Dict[str, Any]]:
        """
        Example method to find the 'best model' by a certain metric (like max accuracy).
        Returns the entry with highest metric_key value, or None if not found.
        """
        models = [
            m for m in self.list_models() if m["name"] == model_name
        ]
        if not models:
            self.logger.warning(f"No entries found for model {model_name}")
            return None

        # Filter out models that don't have the metric_key in their 'metrics'
        valid_models = [m for m in models if metric_key in m["metrics"]]
        if not valid_models:
            self.logger.warning(
                f"No valid metric '{metric_key}' found for any version of {model_name}"
            )
            return None

        best = max(valid_models, key=lambda m: m["metrics"][metric_key])
        self.logger.info(
            f"Best {model_name} by '{metric_key}' is version {best['version']} with value {best['metrics'][metric_key]}"
        )
        return best


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    registry = ModelRegistryIntegration(registry_file="model_registry.json")

    # Example: Register a classification model
    registry.register_model(
        model_name="classification_model",
        version="v1.0.0",
        path="/Users/username/path/to/classification_model.joblib",
        metrics={"accuracy": 0.806, "f1": 0.791},
        parameters={"max_depth": 5, "learning_rate": 0.01},
        tags={"framework": "xgboost", "env": "production"}
    )

    # Retrieve info
    info = registry.get_model_info("classification_model", "v1.0.0")
    print("\nRetrieved model info:\n", info)

    # List all models
    print("\nAll models in registry:\n", registry.list_models())

    # Optionally find best model by accuracy
    best_model = registry.find_best_model("classification_model", metric_key="accuracy")
    if best_model:
        print("\nBest classification model by accuracy:\n", best_model)

    # Example: remove a model
    # removed = registry.remove_model("classification_model", "v0.5.0")
    # print("\nWas v0.5.0 removed?", removed)