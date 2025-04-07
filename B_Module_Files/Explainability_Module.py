"""
Explainability_Module.py

Absolute File Path (example):
    /Users/username/Desktop/energy_optimization_project/B_Module_Files/Explainability_Module.py

PURPOSE:
  - Provide model prediction explanations via SHAP or LIME.
  - Gracefully handle missing libraries or unrecognized methods.
  - Can be extended for advanced methods (Integrated Gradients, etc.).

NOTES:
  - If method is 'lime' or 'shap' but the library isn't installed, a warning is logged
    and the module returns empty or no-op explanations.
  - To use LIME or SHAP properly, the environment must have them installed.
  - The user should provide a suitable `training_data` and a `model_predict` function
    for instance-level explanations.
"""

import logging
from typing import Any, Dict, List, Optional, Callable


class ExplainabilityModule:
    """
    A class that provides methods to explain model predictions using
    a chosen approach (e.g. 'lime' or 'shap'). If these libraries are
    missing or the method is unrecognized, it falls back to a no-op.
    """

    def __init__(
        self,
        method: str = "lime",
        training_data: Optional[List[List[float]]] = None,
        feature_names: Optional[List[str]] = None,
        mode: str = "regression"
    ):
        """
        :param method: Explanation method, e.g., 'lime' or 'shap' (case-insensitive).
        :param training_data: Data on which to fit an explainer, typically the
                              training set (list of lists or numpy array).
        :param feature_names: List of feature names for readability in the explanations.
        :param mode: 'regression' or 'classification', for LIME usage.
        """
        self.logger = logging.getLogger("ExplainabilityModule")
        self.method = method.lower().strip()
        self.training_data = training_data or []
        self.feature_names = feature_names or []
        self.mode = mode
        self._explainer = None

        self.logger.info(
            f"Initializing ExplainabilityModule with method='{self.method}', mode='{self.mode}'."
        )

        # Attempt to set up the chosen explainer
        self._setup_explainer()

    def _setup_explainer(self):
        """
        Initialize the chosen explainability approach.
        If the method is not recognized or libraries are missing, logs a warning
        and leaves _explainer as None.
        """
        if self.method == "lime":
            self._init_lime_explainer()
        elif self.method == "shap":
            self._init_shap_explainer()
        else:
            self.logger.warning(
                f"Explainability method '{self.method}' not recognized. No explainer will be used."
            )

    def _init_lime_explainer(self):
        """
        Attempt to initialize a LIME explainer if the library is installed.
        """
        try:
            import lime
            from lime.lime_tabular import LimeTabularExplainer

            # For LIME, training_data is required for building the explainer
            if not self.training_data:
                self.logger.warning(
                    "No training data provided for LIME; using fallback no-op."
                )
                return

            self.logger.info("Initializing LIME Tabular Explainer...")
            self._explainer = LimeTabularExplainer(
                training_data=self.training_data,
                feature_names=self.feature_names if self.feature_names else None,
                mode=self.mode,
                discretize_continuous=True
            )
            self.logger.info("LIME explainer successfully initialized.")
        except ImportError:
            self.logger.warning("LIME library not installed. No LIME explainer will be used.")
            self._explainer = None
        except Exception as e:
            self.logger.error(
                f"Unexpected error initializing LIME: {e}",
                exc_info=True
            )
            self._explainer = None

    def _init_shap_explainer(self):
        """
        Attempt to initialize a SHAP explainer if the library is installed.
        (We set up a placeholder reference to shap.KernelExplainer here.)
        """
        try:
            import shap  # just to check existence
            self.logger.info("SHAP library found. Will initialize SHAP explainer on demand.")
            # For a real usage, you'd pick e.g. shap.Explainer(model_predict, data)
            # We'll store just a reference to shap here or a minimal approach
            self._explainer = "shap_kernel_stub"
        except ImportError:
            self.logger.warning("SHAP library not installed. No SHAP explainer will be used.")
            self._explainer = None
        except Exception as e:
            self.logger.error(
                f"Unexpected error initializing SHAP: {e}",
                exc_info=True
            )
            self._explainer = None

    def explain_instance(self, model_predict: Callable[[Any], Any], instance: Any, **kwargs) -> Dict[str, float]:
        """
        Generate an explanation for a single instance using the chosen method.

        :param model_predict: A callable that takes input (array-like) and returns predictions.
        :param instance: The data row to explain (list, numpy array, or 1D vector).
        :param kwargs: Additional arguments to pass to the explainer's explain method.
        :return: A dictionary {feature_name: contribution} if available, else empty.
        """
        # If we have no explainer, return empty
        if not self._explainer:
            self.logger.warning("No explainer available. Returning empty explanation.")
            return {}

        # If LIME method
        if self.method == "lime" and "lime_tabular" in str(type(self._explainer)):
            return self._explain_lime_instance(model_predict, instance, **kwargs)

        # If SHAP method
        elif self.method == "shap" and self._explainer == "shap_kernel_stub":
            return self._explain_shap_instance(model_predict, instance, **kwargs)

        # Fallback if something is off
        self.logger.warning(f"No recognized explainer flow for method='{self.method}'. Returning empty.")
        return {}

    def _explain_lime_instance(self, model_predict: Callable[[Any], Any], instance: List[float], **kwargs) -> Dict[str, float]:
        """
        Actually produce a LIME explanation for one instance.
        """
        try:
            explanation = self._explainer.explain_instance(
                data_row=instance,
                predict_fn=model_predict,
                num_features=len(self.feature_names) or 5,
                **kwargs
            )
            # Convert to dict: explanation.as_list() => list of tuples (feature, contribution)
            return dict(explanation.as_list())
        except Exception as e:
            self.logger.error(
                f"Error generating LIME explanation: {e}",
                exc_info=True
            )
            return {}

    def _explain_shap_instance(self, model_predict: Callable[[Any], Any], instance: List[float], **kwargs) -> Dict[str, float]:
        """
        Actually produce a SHAP explanation for one instance.
        We'll do a minimal approach for demonstration. In real usage, you'd instantiate
        a shap explainer object with model + data. We'll just log a placeholder.
        """
        import numpy as np
        try:
            # In real usage, you'd do something like:
            # shap_explainer = shap.KernelExplainer(model_predict, self.training_data)
            # shap_values = shap_explainer.shap_values(np.array([instance]))
            self.logger.info("Simulated SHAP kernel explainer, returning random placeholders.")
            # Return random placeholders as a demonstration
            shap_values = np.random.randn(len(self.feature_names) or 5)
            explanation = {
                f_name: float(val)
                for f_name, val in zip(self.feature_names or [f"feature_{i}" for i in range(shap_values.size)], shap_values)
            }
            return explanation
        except Exception as e:
            self.logger.error(
                f"Error generating SHAP explanation: {e}",
                exc_info=True
            )
            return {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage: create a dummy dataset and a dummy model_predict function
    dummy_data = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]
    dummy_feature_names = ["feat1", "feat2"]

    def dummy_predict(X):
        # simple sum of columns as a mock "prediction"
        import numpy as np
        arr = np.array(X)
        return arr.sum(axis=1)

    # Initialize with LIME approach
    lime_explainer = ExplainabilityModule(
        method="lime",
        training_data=dummy_data,
        feature_names=dummy_feature_names,
        mode="regression"
    )
    explanation_lime = lime_explainer.explain_instance(dummy_predict, [9.0, 10.0])
    print("LIME Explanation:", explanation_lime)

    # Initialize with SHAP approach
    shap_explainer = ExplainabilityModule(
        method="shap",
        training_data=dummy_data,
        feature_names=dummy_feature_names
    )
    explanation_shap = shap_explainer.explain_instance(dummy_predict, [9.0, 10.0])
    print("SHAP Explanation:", explanation_shap)

    # Initialize with an unknown method, fallback scenario
    unknown_explainer = ExplainabilityModule(
        method="unknown_method",
        training_data=dummy_data,
        feature_names=dummy_feature_names
    )
    explanation_unknown = unknown_explainer.explain_instance(dummy_predict, [9.0, 10.0])
    print("Unknown Method Explanation:", explanation_unknown)