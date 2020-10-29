import csv
import json
import os

import cv2
import numpy as np
import argparse
import yaml

parser = argparse.ArgumentParser()

parser.add_argument('--dir')
parser.add_argument('--process')
args = parser.parse_args()
if not args.dir:
    raise Exception("Plz set C2_TrainDev_DIR_PATH wit --dir")

BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
C2_TrainDev_DIR_PATH = args.dir
if args.process=="train":
    SOURCE_IMAGE_DIR = "Train"
    SOURCE_LABEL_NAME = "train.csv"
    DATA_DIR = 'train'
    yolo_label_dir = os.path.join(BASE_DIR, 'mango-defects-data/labels/train')
    yolo_image_dir = os.path.join(BASE_DIR, 'mango-defects-data/images/train')
elif args.process=="dev":
    SOURCE_IMAGE_DIR = "Dev"
    SOURCE_LABEL_NAME = "dev.csv"
    DATA_DIR = 'dev'
    yolo_label_dir = os.path.join(BASE_DIR, 'mango-defects-data/labels/dev')
    yolo_image_dir = os.path.join(BASE_DIR, 'mango-defects-data/images/dev')
else:
    raise Exception("Plz set process data dev/train")



with open(os.path.join(BASE_DIR, 'data/mango-defect.yaml'), 'r') as yaml_file:
    config = yaml.load(yaml_file)


def create_labels():
    with open(os.path.join(C2_TrainDev_DIR_PATH, SOURCE_LABEL_NAME), 'r', newline='',
              encoding='UTF-8-sig') as train_csv:
        rows = list(csv.reader(train_csv))
    data = [[y for y in x if len(y) > 0] for x in rows]

    if not os.path.exists(yolo_image_dir):
        os.mkdir(yolo_image_dir)
    if not os.path.exists(yolo_label_dir):
        os.mkdir(yolo_label_dir)

    for img in data:
        _defect_label_creation(img, yolo_label_dir)


def _defect_label_creation(l: list, label_file_dir: str):
    if not os.path.exists(label_file_dir):
        raise Exception(f"{label_file_dir} not exists")

    filename = l[0]
    img = cv2.imread(os.path.join(C2_TrainDev_DIR_PATH, SOURCE_IMAGE_DIR, filename))
    (img_h, img_w, _) = img.shape
    del l[0]
    defect_num = int(len(l) / 5)
    converted_list = np.array(l).reshape(defect_num, 5).tolist()
    labels = []
    for defect in converted_list:
        [x, y, w, h, label] = [float(n) for n in defect[:4]] + [defect[-1]]
        cen_x = (x + w / 2) / img_w
        cen_y = (y + h / 2) / img_h
        box_w = w / img_w
        box_h = h / img_h
        class_idx = config['names'].index(config['names_mapping'][label])
        labels.append([class_idx, cen_x, cen_y, box_w, box_h])
    label_file_name = os.path.join(label_file_dir, filename.replace('.jpg', '.txt'))
    with open(label_file_name, 'w') as txt:
        txt.writelines([" ".join([str(ele) for ele in line]) + "\n" for line in labels])
    print(f"writing {label_file_name} ...")
    return label_file_name


create_labels()
