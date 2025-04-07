"""
imagenet_module.py

PURPOSE:
    Provides a simple interface to use a pre-trained ResNet50 (ImageNet) model for image classification.
    Includes:
      - Image preprocessing (resize, convert to array, expand dims, preprocess_input).
      - Model inference to obtain top-3 class predictions.
      - Optionally, a main() function to run a test inference on 'cat_image.jpg' (located in D_Dataset_Files).

USAGE EXAMPLE:
    # Command-line usage to test the module:
    # python "/Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/imagenet_module.py 
"

    # Programmatic usage:
    from imagenet_module import ImageNetModule

    # Instantiate
    classifier = ImageNetModule()
    # Inference
    results = classifier.classify_image("/path/to/some_image.jpg")
    print(results)
"""

import os
import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image


class ImageNetModule:
    """
    A simple interface to load a pre-trained ResNet50 model (on ImageNet)
    and classify images (returns top-k predictions).
    """

    def __init__(self):
        """
        Initialize the ResNet50 model (weights=ImageNet).
        """
        print("[INFO] Loading ResNet50 model (ImageNet pre-trained)...")
        self.model = ResNet50(weights="imagenet")
        print("[INFO] Model loaded.")

    def classify_image(self, img_path, top_k=3):
        """
        Classify an image and return the top-k predictions.

        :param img_path: Path to an image file (JPEG, PNG, etc.).
        :param top_k: Number of top predictions to return.
        :return: List of (class_name, probability) tuples.
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"[ERROR] Image not found at {img_path}")

        # Load and preprocess the image
        print(f"[INFO] Loading and preprocessing image from: {img_path}")
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)  # ResNet50-specific preprocessing

        # Predict
        print("[INFO] Running inference...")
        preds = self.model.predict(x, verbose=0)

        # Decode predictions => list of tuples (class, description, prob)
        decoded = decode_predictions(preds, top=top_k)[0]

        # Format the result: (class_name, probability)
        results = [(d[1], float(d[2])) for d in decoded]  # (label, prob)
        return results


def main():
    """
    If run as a script, test the module on 'cat_image.jpg' located in D_Dataset_Files.
    """
    # Adjust the path as needed if your folder structure changes
    TEST_IMAGE = "/Users/robertsalinas/Desktop/energy_optimization_project/D_Dataset_Files/cat_image.jpg"

    print("[INFO] Instantiating ImageNetModule...")
    classifier = ImageNetModule()
    print("[INFO] Classifying test image...")

    try:
        results = classifier.classify_image(TEST_IMAGE, top_k=3)
        print("[INFO] Top-3 Predictions:")
        for label, prob in results:
            print(f"  {label}: {prob:.4f}")
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()