"""
Edge_Inference_Module_Prototype.py

Absolute File Path (example):
    /Users/username/Desktop/energy_optimization_project/B_Module_Files/Edge_Inference_Module_Prototype.py

PURPOSE:
  - Simulate lightweight edge inference by loading a small PyTorch model.
  - Provide a fallback if loading fails.
  - Preprocess input data for model inference.
  - Return predictions, optionally applying postprocessing.

NOTES:
  - This script is a prototype for edge computing scenarios.
  - You can extend it with real quantization, hardware checks (like GPU vs. CPU vs. TPU),
    or model optimizations for edge devices.
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class EdgeInferenceModule:
    """
    A prototype for edge-based inference using a small torch model.
    - Loads (or creates) a model with minimal parameters.
    - Processes input data (numpy array), returning predictions as numpy array.
    - Supports potential fallback and optional device specification.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: int = 4,
        hidden_size: int = 8,
        output_size: int = 2,
        device: str = "cpu",
        quantized: bool = False,
    ):
        """
        :param model_path: Path to a saved torch model. If None or load fails, use default model.
        :param input_size: Dimension of input vector (e.g., 4 for a small sensor reading).
        :param hidden_size: Hidden layer dimension for the fallback default model.
        :param output_size: Number of output classes or regression dimension.
        :param device: "cpu" or "cuda" (for edge devices with GPU, or fallback CPU).
        :param quantized: If True, attempt to load or run model in quantized mode (placeholder).
        """
        self.logger = logging.getLogger("EdgeInferenceModule")
        self.logger.info(
            f"Initializing EdgeInferenceModule with device={device}, "
            f"quantized={quantized}, model_path={model_path}"
        )

        self.device = device
        self.quantized = quantized
        self.model = None

        # Attempt to load a model from file if provided
        if model_path:
            self.model = self._load_model_safely(model_path, input_size, hidden_size, output_size)
        else:
            self.logger.info("No model_path provided. Using default model architecture.")
            self.model = self._default_model(input_size, hidden_size, output_size)

        # Move the model to the chosen device
        self.model.to(self.device)
        self.model.eval()

    def _load_model_safely(self, model_path: str, in_size: int, hid: int, out_size: int) -> nn.Module:
        """
        Attempts to load a Torch model from disk. Falls back to default if anything fails.
        """
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            # If the file was saved as a whole model, `torch.load` might return a complete model object
            # or just a state_dict. We'll assume it's a state_dict here:
            fallback_model = self._default_model(in_size, hid, out_size)
            if isinstance(state_dict, dict):
                # Attempt to load it as a state_dict
                fallback_model.load_state_dict(state_dict)
                self.logger.info(f"Successfully loaded model state from {model_path}")
                return fallback_model
            else:
                # It's a full model object
                self.logger.info(f"Loaded a full model object from {model_path}")
                return state_dict
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {e}", exc_info=True)
            self.logger.info("Falling back to default model.")
            return self._default_model(in_size, hid, out_size)

    def _default_model(self, in_size: int, hid: int, out_size: int) -> nn.Module:
        """
        Creates a minimal feed-forward network for fallback usage.
        """
        self.logger.info(
            f"Building default fallback model with "
            f"input_size={in_size}, hidden_size={hid}, output_size={out_size}."
        )
        return nn.Sequential(
            nn.Linear(in_size, hid),
            nn.ReLU(),
            nn.Linear(hid, out_size)
        )

    def run_inference(self, data: np.ndarray) -> np.ndarray:
        """
        Run forward pass on input data. Input shape [batch_size, input_size].
        Returns a numpy array [batch_size, output_size].
        """
        if self.model is None:
            self.logger.error("No model loaded. Cannot run inference.")
            return np.array([])

        # Convert data to float32 tensor on the correct device
        tensor_data = torch.from_numpy(data.astype(np.float32)).to(self.device)

        self.logger.debug(f"Input data shape: {tensor_data.shape}")

        with torch.no_grad():
            outputs = self.model(tensor_data)
        # Move outputs back to CPU for numpy conversion
        results = outputs.cpu().numpy()
        return results

    def quantize_model(self):
        """
        Placeholder for quantization logic (int8, etc.).
        In real usage, you'd do PyTorch static/dynamic quantization.
        """
        if not self.quantized:
            self.logger.warning("quantized=True was not set; skipping quantization.")
            return
        # Example placeholder:
        self.logger.info("Attempting to quantize the model (placeholder).")
        # PyTorch quantization logic could be inserted here
        # For demonstration, we do nothing:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    edge_module = EdgeInferenceModule(
        model_path=None,       # Or path to a saved model
        input_size=4,
        hidden_size=8,
        output_size=2,
        device="cpu",
        quantized=False
    )

    # Generate some sample data (2 samples, each of size 4)
    sample_data = np.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.float32)

    # Run inference
    preds = edge_module.run_inference(sample_data)
    print("Edge Inference predictions:", preds)