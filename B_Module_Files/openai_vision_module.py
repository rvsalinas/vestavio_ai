#!/usr/bin/env python
"""
File: openai_vision_module.py

Purpose:
  - Demonstrates how to call OpenAI's vision-capable model to analyze an image.
  - Supports either a direct image URL or a Base64-encoded image string.
  - Also supports analyzing multiple images in a single request.

Dependencies:
  - openai>=1.0.0 (the new function-based or .chat.completions interface).
  - Python 3.7+ (for typing and f-strings).
  - (Optional) 'base64' if you want to convert local images to base64.

Usage Example:
  from openai_vision_module import OpenAIVisionModule

  vision = OpenAIVisionModule(model_name="gpt-4o-mini")
  # For URL-based single image
  result_text = vision.analyze_image_url(
      image_url="https://upload.wikimedia.org/...",
      user_prompt="What is shown here?"
  )
  print(result_text)

  # For Base64-based single image
  base64_str = vision.load_image_as_base64("/path/to/local_image.jpg")
  result_text = vision.analyze_image_base64(
      base64_str,
      user_prompt="Describe this image in detail."
  )
  print(result_text)

  # For multiple images in a single request
  multi_images = [
      ("https://upload.wikimedia.org/...", "auto"),
      ("https://example.com/another_image.jpg", "high")
  ]
  result_text = vision.analyze_multiple_images(
      multi_images,
      user_prompt="Compare these images. Are they similar?"
  )
  print(result_text)
"""

import os
import logging
import base64
import openai
from typing import Optional, List, Tuple


class OpenAIVisionModule:
    """
    A simple class for using OpenAI's image+vision feature via chat completions.
    Allows analyzing an image (URL or Base64) by passing it to the model as
    part of a user message (with "type": "image_url").
    Also includes a method to handle multiple images in one request.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        :param model_name: Name of the OpenAI vision-capable model (e.g. "gpt-4o-mini").
        :param api_key: OpenAI API key. If None, will use OPENAI_API_KEY from env.
        :param logger: Optional logger instance.
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("No API key provided or found in environment (OPENAI_API_KEY).")

        openai.api_key = self.api_key

        if logger is None:
            self.logger = logging.getLogger("OpenAIVisionModule")
            if not self.logger.handlers:
                console = logging.StreamHandler()
                console.setLevel(logging.INFO)
                self.logger.addHandler(console)
                self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger

        self.logger.info(f"OpenAIVisionModule initialized with model='{self.model_name}'.")

    def analyze_image_url(
        self,
        image_url: str,
        user_prompt: str = "What’s in this image?",
        detail: str = "auto"
    ) -> str:
        """
        Analyzes an image from a direct URL.
        :param image_url: The full URL of the image, or a data URL with Base64 content.
        :param user_prompt: The text portion of the user’s question or instruction.
        :param detail: 'low', 'high', or 'auto' controlling how detailed the analysis is.
        :return: Model’s textual answer describing/analyzing the image.
        """
        self.logger.info(f"Analyzing image from URL with detail='{detail}'.")
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                    "detail": detail
                                }
                            }
                        ],
                    }
                ],
                max_tokens=300,
            )
            # Extract the assistant's textual reply
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error calling OpenAI vision model: {e}", exc_info=True)
            return "I'm sorry, I encountered an error while analyzing the image."

    def analyze_image_base64(
        self,
        base64_str: str,
        user_prompt: str = "What’s in this image?",
        detail: str = "auto"
    ) -> str:
        """
        Analyzes an image from a Base64-encoded string.
        :param base64_str: Base64-encoded data for the image (JPEG/PNG).
        :param user_prompt: The text portion of the user’s question or instruction.
        :param detail: 'low', 'high', or 'auto' controlling how detailed the analysis is.
        :return: Model’s textual answer describing/analyzing the image.
        """
        self.logger.info(f"Analyzing image from Base64 with detail='{detail}'.")
        # Build a data URL with the base64 string
        data_url = f"data:image/jpeg;base64,{base64_str}"

        # Same call, but using the data URL
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url,
                                    "detail": detail
                                }
                            }
                        ],
                    }
                ],
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error calling OpenAI vision model: {e}", exc_info=True)
            return "I'm sorry, I encountered an error while analyzing the image."

    def analyze_multiple_images(
        self,
        image_data: List[Tuple[str, str]],
        user_prompt: str = "What’s in these images?"
    ) -> str:
        """
        Analyzes multiple images in a single request.
        :param image_data: A list of (image_url, detail) pairs. detail can be 'low', 'high', or 'auto'.
        :param user_prompt: The text portion of the user’s question or instruction.
        :return: Model’s textual answer describing/analyzing the images collectively.
        """
        self.logger.info(f"Analyzing {len(image_data)} images in one request.")
        # Build the content array
        content_array = [{"type": "text", "text": user_prompt}]
        for (img_url, detail) in image_data:
            content_array.append({
                "type": "image_url",
                "image_url": {
                    "url": img_url,
                    "detail": detail
                }
            })

        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": content_array,
                    }
                ],
                max_tokens=500,  # slightly larger for multi-image
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error calling OpenAI vision model for multiple images: {e}", exc_info=True)
            return "I'm sorry, I encountered an error while analyzing these images."

    @staticmethod
    def load_image_as_base64(image_path: str) -> str:
        """
        Utility function to load a local image file and convert to Base64 string.
        :param image_path: Local path to the image file.
        :return: Base64-encoded string.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


if __name__ == "__main__":
    # Quick demonstration
    logging.basicConfig(level=logging.INFO)
    vision = OpenAIVisionModule(model_name="gpt-4o-mini")

    # Example: analyzing an image from a URL
    sample_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Golden_retriever.jpg/2560px-Golden_retriever.jpg"
    reply = vision.analyze_image_url(sample_url, user_prompt="Describe this dog.")
    print("Analysis from URL:\n", reply)

    # Example: analyzing multiple images
    multi_images = [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Golden_retriever.jpg/2560px-Golden_retriever.jpg", "auto"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Collie_dog.jpg/2560px-Collie_dog.jpg", "high")
    ]
    reply_multi = vision.analyze_multiple_images(multi_images, user_prompt="Compare these two dogs.")
    print("\nAnalysis from multiple images:\n", reply_multi)

    # Example: analyzing an image from a local file
    # 1) Convert local file to base64
    # base64_str = vision.load_image_as_base64("/path/to/local_dog.jpg")
    # 2) Pass to the method
    # reply2 = vision.analyze_image_base64(base64_str, user_prompt="What breed is this dog?")
    # print("\nAnalysis from Base64:\n", reply2)