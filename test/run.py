import os
import torch
import sys
from PIL import Image
import numpy as np
import pandas as pd
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from utils import crop_image, detect_word, show_box

# A - Load models
# Load model detecting corners of the card
model_detect_corners = YOLO('../checkpoints/detect_corners.pt')

# Load model detecting information on the card
model_detect_info = YOLO('../checkpoints/detect_info.pt')

# Load model OCR reading information
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = '../checkpoints/transformerocr.pth'
config['cnn']['pretrained'] = False
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False

model_read_info = Predictor(config)

# B - Read image
org_image = cv2.imread("./test_images/can_cuoc.jpg")
image = org_image.copy()  # Copy image to not change origin image when drawing bounding box at the corners of card
plt.figure()
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()


# C - Extract information
# 1. Detect corners
results_detect_corner = model_detect_corners(image)  # use YOLOv8 to detect corners of card
image = show_box.draw_bbox(image, results_detect_corner[0].boxes.xyxy)  # Draw bbox at 4 corners of card
plt.figure()
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# 2. Crop image
# Crop the image at the center of 4 corner-boxes of card
img_crop = crop_image.CropImg(results_detect_corner[0].boxes, org_image)
plt.figure()
plt.imshow(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))

# 3. Detect information
results_detect_info = model_detect_info(img_crop)   # use YOLOv8 to detect information of card
img_crop = show_box.draw_bbox(img_crop, results_detect_info[0].boxes.xyxy)  # Draw bbox at information of card
plt.figure()
plt.imshow(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))

# 4. Read information by VietOCR
dict_info = detect_word.OCR(results_detect_info[0].boxes,
                            results_detect_info[0].names,
                            img_crop, model_read_info)

dict_themes = {
    'hometown': 'Que quan',
    'dob': 'Ngay sinh',
    'name': 'Ho va ten',
    'id': 'So can cuoc'
}

for key, value in dict_info.items():
    if key != 'face':
        print(f"{dict_themes[key]}: {value}")

plt.show()
