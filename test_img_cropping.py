import csv
import json
import os

import cv2
import numpy as np

DATA_DIR_PATH = "/home/chihung/mango-defect-detection-yolov5/mango-defects-data"
IMAGE_DIR = "Test"
LABEL_NAME = "Test_mangoXYWH.csv"
SAVE_CROPPED_TEST_IMG_DIR = "/home/chihung/mango-defect-detection-yolov5/cropped_test"

def crop_test_img():
    with open(os.path.join(DATA_DIR_PATH, LABEL_NAME), 'r', newline='', encoding='UTF-8-sig') as train_csv:
        rows = list(csv.reader(train_csv))
    del rows[0]
    cropping_list = [[y for y in x if len(y) > 0] for x in rows]
    for img in cropping_list:
        file_name = img[0]
        mango_coordinate = img[1:]
        crop_img(
            input_path= os.path.join(DATA_DIR_PATH, IMAGE_DIR, file_name),
            coordinate=mango_coordinate,
            output_path= os.path.join(SAVE_CROPPED_TEST_IMG_DIR, file_name),
            )

    print("Test image cropping finished")



def crop_img(input_path,coordinate,output_path):
    if not os.path.exists(input_path):
        raise Exception(f"Cropping Path Not Found - {input_path}")

    img = cv2.imread(input_path)
    [x, y, w, h] = coordinate
    x, y, w, h = int(float(x)), int(float(y)), int(float(w)), int(float(h))
    crop_img = img[y:y + h, x:x + w]
    cv2.imwrite(output_path, crop_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(f"write cropped img to {output_path}")



def main():
    crop_test_img()

if __name__ == "__main__":
    main()
