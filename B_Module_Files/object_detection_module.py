"""
File Path (example):
/Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/object_detection_module.py

PURPOSE:
    - Perform object detection using a pre-trained Faster R-CNN model (similar to coco_detection_module 
      but with extended functionality).
    - Supports batch processing of images from a folder.
    - Saves detection results (bounding boxes, labels, confidence) in a JSON file if requested.
    - Potential integration with environmental or analytics modules.

USAGE EXAMPLE (command line):
    (sensor_fusion_env) $ python "/Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/object_detection_module.py" \
        --input_folder "/path/to/images" \
        --output_json "detections_output.json"

    # Or specify a single image:
    (sensor_fusion_env) $ python "/Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/object_detection_module.py" \
        --input_image "/path/to/single_image.jpg"
"""

import os
import sys
import json
import argparse
import torch
import torchvision
from torchvision import transforms
from PIL import Image

# For fasterrcnn_resnet50_fpn model weights
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)

class ObjectDetectionModule:
    def __init__(self, device=None):
        """
        Initialize the extended object detection module using Faster R-CNN.
        :param device: (str) 'cpu', 'cuda', or 'mps'. If None, auto-detect available hardware.
        """
        # Check if user explicitly provided a device
        if device is None:
            # Auto-detect Apple Silicon MPS, then CUDA, else CPU
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"[INFO] Using device: {self.device}")

        print("[INFO] Initializing Faster R-CNN (COCO) model from TorchVision with extended features...")
        # Load a pre-trained Faster R-CNN model
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("[INFO] Extended object detection model loaded and ready.")

        # We can fetch the category names from the weights meta if desired
        self.category_names = weights.meta["categories"]

        # Transform to convert PIL images to model input
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def detect_objects_in_image(self, image_path, confidence_threshold=0.5):
        """
        Perform object detection on a single image.

        :param image_path: Path to the image file
        :param confidence_threshold: Minimum confidence score for detection
        :return: A list of detections, each detection is a dict:
                 {
                    "label": str,
                    "confidence": float,
                    "bbox": [x1, y1, x2, y2]
                 }
        """
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            return []

        # Load and transform the image
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).to(self.device).unsqueeze(0)

        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)

        # Process detections
        detections = []
        for idx in range(len(outputs[0]["scores"])):
            score = float(outputs[0]["scores"][idx].cpu().item())
            if score >= confidence_threshold:
                label_idx = int(outputs[0]["labels"][idx].cpu().item())
                label_name = self.category_names[label_idx] if label_idx < len(self.category_names) else f"Class_{label_idx}"

                bbox_raw = outputs[0]["boxes"][idx].cpu().numpy().tolist()
                # Convert [xmin, ymin, xmax, ymax] -> more readable or keep as is
                x1, y1, x2, y2 = bbox_raw

                detections.append({
                    "label": label_name,
                    "confidence": round(score, 4),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                })
        return detections

    def detect_objects_in_folder(self, folder_path, confidence_threshold=0.5):
        """
        Perform object detection on all images in a given folder.

        :param folder_path: Path to the folder containing images
        :param confidence_threshold: Minimum confidence score for detection
        :return: A dict, where each key is an image file name, and each value is a list of detections
        """
        if not os.path.isdir(folder_path):
            print(f"[ERROR] Folder not found: {folder_path}")
            return {}

        detections_dict = {}
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(valid_extensions):
                image_path = os.path.join(folder_path, file_name)
                detections = self.detect_objects_in_image(image_path, confidence_threshold)
                detections_dict[file_name] = detections
        return detections_dict

def main():
    parser = argparse.ArgumentParser(description="Extended Object Detection Module - Faster R-CNN (COCO)")
    parser.add_argument("--input_image", type=str, default=None, help="Path to a single image file.")
    parser.add_argument("--input_folder", type=str, default=None, help="Path to a folder of images for batch processing.")
    parser.add_argument("--output_json", type=str, default=None, help="Path to output JSON file of detections.")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Confidence threshold for detections.")
    parser.add_argument("--device", type=str, default=None, help="Device to use: 'cpu' or 'cuda'. Defaults to auto.")
    args = parser.parse_args()

    # Initialize the object detection module
    detector = ObjectDetectionModule(device=args.device)

    # Determine single image or folder
    if args.input_image is None and args.input_folder is None:
        print("[ERROR] You must specify either --input_image or --input_folder.")
        sys.exit(1)

    results = None

    # Single image detection
    if args.input_image:
        print(f"[INFO] Processing single image: {args.input_image}")
        results = detector.detect_objects_in_image(args.input_image, args.confidence_threshold)
        # Print to console
        print("[INFO] Detections:")
        for det in results:
            print(f"  Label: {det['label']}, Confidence: {det['confidence']}, BBox: {det['bbox']}")
    # Folder detection
    elif args.input_folder:
        print(f"[INFO] Processing folder: {args.input_folder}")
        results = detector.detect_objects_in_folder(args.input_folder, args.confidence_threshold)
        # Print to console
        for file_name, detections in results.items():
            print(f"[INFO] File: {file_name}")
            for det in detections:
                print(f"  Label: {det['label']}, Confidence: {det['confidence']}, BBox: {det['bbox']}")

    # Optionally save to JSON
    if args.output_json and results is not None:
        print(f"[INFO] Saving detections to JSON: {args.output_json}")
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print("[INFO] JSON saved successfully.")

    print("[INFO] Done with object_detection_module.")

if __name__ == "__main__":
    main()