"""
Image Preprocessing Module for Dot Detection
Enhances contrast and removes background gradients to make dots more visible
Works on both color and grayscale images
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from image_process import save_step
import os

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    picture_folder = os.path.join(base_path, "../../Pictures")
    image_path = os.path.join(base_path, picture_folder, "04_normalized.jpg")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")  # 'L' = grayscale
    img = np.array(img)                        # convert to NumPy array

    # Invert the image
    img = 255 - img  # now 'img' is inverted and ready to use
    save_step(img, "Inverted", 41234)

    brightness_threshold = 20
    for i in range(30):
        strict_bw = np.where(img >= brightness_threshold, 255, 0).astype(np.uint8)
        
        save_step(strict_bw, "BW", brightness_threshold)
        brightness_threshold += 5


    threshold = 70  # change this value as needed

    """ # Convert to black and white
    for i in range(50):
        threshold = 50 + 2*i
        _, bw_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        save_step(bw_img, f"Inverted_{i}", threshold)
    # Example: get width and height
    h, w = img.shape"""