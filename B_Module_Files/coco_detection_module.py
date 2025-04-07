"""
File: coco_detection_module.py
Absolute File Path:
  /Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/coco_detection_module.py

PURPOSE:
    Utilizes a Faster R-CNN model (pre-trained on COCO) for object detection. 
    Demonstrates:
      • Image loading and preprocessing
      • Model inference for bounding boxes, labels, confidence scores
      • Basic post-processing and confidence thresholding

USAGE EXAMPLE (terminal):
    (sensor_fusion_env) $ python "/Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/coco_detection_module.py"

REQUIREMENTS:
    • PyTorch, TorchVision (with pre-trained models)
    • PIL (for image loading)
    • Matplotlib or OpenCV if you want to visualize bounding boxes (optional)
"""

import os
import sys
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image

# COCO class names (a subset for brevity or the entire list)
# You can find the full 91 categories from the official COCO object detection dataset.
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet",
    "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

class CocoDetectionModule:
    """
    A class that loads a pre-trained Faster R-CNN model (COCO) from TorchVision,
    performs object detection on an input image, and prints out bounding boxes,
    labels, and confidence scores above a given threshold.
    """

    def __init__(self, confidence_threshold=0.5):
        """
        :param confidence_threshold: Only detections with confidence > threshold are printed.
        """
        self.confidence_threshold = confidence_threshold
        # CPU or GPU?
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("[INFO] Initializing Faster R-CNN (COCO) model from TorchVision...")
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.model.to(self.device)
        print("[INFO] Model loaded and ready.")

        # Transforms for the input image
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # convert PIL Image or numpy array to FloatTensor
        ])

    def detect(self, image_path):
        """
        Run object detection on a single image.

        :param image_path: Path to the .jpg/.png image
        :return: list of dictionaries with bounding boxes, labels, and scores
        """
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found at {image_path}")
            sys.exit(1)

        print(f"[INFO] Loading and preprocessing image from: {image_path}")
        image = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(image).to(self.device)

        print("[INFO] Running inference...")
        with torch.no_grad():
            outputs = self.model([img_tensor])  # List[Dict] with "boxes", "labels", "scores"
        # We only have one image, so outputs[0]
        detections = outputs[0]

        # Convert to CPU if on GPU
        boxes = detections["boxes"].cpu().numpy()
        labels = detections["labels"].cpu().numpy()
        scores = detections["scores"].cpu().numpy()

        results = []
        for box, label_idx, score in zip(boxes, labels, scores):
            if score >= self.confidence_threshold:
                category = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else str(label_idx)
                box = box.tolist()
                results.append({
                    "box": box,
                    "label": category,
                    "score": float(score)
                })
        return results

    def print_results(self, results):
        """
        Print the detection results in a readable format.
        """
        if not results:
            print(f"[INFO] No detections found above confidence threshold.")
            return

        print("[INFO] Detections:")
        for i, det in enumerate(results, start=1):
            print(f"  Detection #{i}:")
            print(f"    Label: {det['label']}")
            print(f"    Confidence: {det['score']:.4f}")
            print(f"    BBox: {det['box']}")

def main():
    # Example usage with a cat image
    # You can adjust the confidence threshold if you like
    module = CocoDetectionModule(confidence_threshold=0.5)

    # For demonstration, we'll use a cat_image in D_Dataset_Files
    cat_image_path = "/Users/robertsalinas/Desktop/energy_optimization_project/D_Dataset_Files/cat_image.jpg"
    detections = module.detect(cat_image_path)
    module.print_results(detections)
    print("[INFO] Done with coco_detection_module demo.")

if __name__ == "__main__":
    main()