import os
import cv2
import matplotlib.pyplot as plt
import easyocr
import torch
import numpy as np
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

def predict(image_path):
    # reading the image
    image = cv2.imread(image_path)

    # predicting the results
    results = model.predict(image, save=True, conf=0.25, iou=0.2)
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Class probabilities for classification outputs

    # slicing the results.boxes
    tensor = torch.tensor(boxes.data)
    tensor_cpu = tensor.cpu()
    numpy_array = np.array(tensor_cpu)
    detection = numpy_array[0]
    bbox = []
    for cord in detection[:-2]:
        cord = int(cord)
        bbox.append(cord)

    # Function to crop the image based on the bounding box
    def crop_image(image, bounding_box):
        x, y, h, w = bounding_box
        roi = image[y:w, x:h]
        return roi

    roi = crop_image(image, bbox)

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=False)

    # Perform OCR on the cropped image
    ocr_result = reader.readtext(roi)

    # Function to filter the extracted text based on a threshold
    def filter_text(roi, ocr_result, roi_threshold):
        rectangle_size = roi.shape[0] * roi.shape[1]

        plate = []

        for result in ocr_result:
            length = np.sum(np.subtract(result[0][1], result[0][0]))
            height = np.sum(np.subtract(result[0][2], result[0][1]))

            if length * height / rectangle_size > roi_threshold:
                plate.append(result[1])
        return plate

    plate_number = filter_text(roi, ocr_result, roi_threshold=0.3)

    # Display the cropped image with bounding box and print the extracted plate number
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    print(plate_number)

predict("your image path")