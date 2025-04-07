"""
vision_transformer_module.py

Purpose:
  This module provides a class `VisionTransformerModule` which loads a
  Vision Transformer (ViT) model from Hugging Face Transformers, and
  exposes methods for inference.
"""

import os
import logging
import torch
import numpy as np
from PIL import Image
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification
)


class VisionTransformerModule:
    """
    A class for performing image classification using a Vision Transformer.
    """

    def __init__(self, model_name: str = "google/vit-base-patch16-224", device: str = "cpu"):
        """
        :param model_name: Name of the Hugging Face ViT model to load.
        :param device: 'cpu' or 'cuda' for GPU usage.
        """
        self.logger = logging.getLogger("VisionTransformerModule")
        self.logger.info(f"Initializing VisionTransformerModule with model_name='{model_name}' and device='{device}'.")

        self.model_name = model_name
        self.device = device

        # Load the feature extractor / image processor
        self.logger.info(f"Loading ViTImageProcessor for '{model_name}'...")
        self.image_processor = ViTImageProcessor.from_pretrained(model_name)

        # Load the ViT model
        self.logger.info(f"Loading ViTForImageClassification for '{model_name}'...")
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.model.to(device)  # Move model to CPU or GPU

        self.logger.info("VisionTransformerModule initialized successfully.")

    def predict_image(self, image: Image.Image) -> dict:
        """
        Classify a single PIL image using the loaded ViT model.
        
        :param image: A PIL Image object.
        :return: A dictionary with top predicted label and confidence.
        """
        # Convert PIL image to model's input format
        self.logger.debug("Processing the image with ViTImageProcessor...")
        inputs = self.image_processor(images=image, return_tensors="pt")

        # Move inputs to the correct device
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        with torch.no_grad():
            self.logger.debug("Running inference on the image...")
            outputs = self.model(**inputs)

        # Get predicted logits
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        class_idx = int(np.argmax(probs))

        # Hugging Face's ViT models typically have model.config.id2label
        # which is a dict: {class_idx: "LABEL_NAME"}
        label = self.model.config.id2label[class_idx]
        confidence = float(probs[class_idx])

        self.logger.info(f"Predicted: {label} ({confidence:.4f})")
        return {"label": label, "confidence": confidence}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Simple demonstration usage:
    from PIL import Image

    # Instantiate the module
    vit_module = VisionTransformerModule(model_name="google/vit-base-patch16-224", device="cpu")

    # Load a test image from disk (replace with a real local path)
    # e.g. image_path = "/Users/youruser/Desktop/some_image.jpg"
    image_path = "/Users/robertsalinas/Desktop/energy_optimization_project/sample_cat_image.jpg"
    
    try:
        img = Image.open(image_path).convert("RGB")
        prediction = vit_module.predict_image(img)
        print("Prediction:", prediction)
    except FileNotFoundError:
        print(f"[WARNING] Sample image not found at {image_path}.\nPlease provide a valid image file path.")