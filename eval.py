import csv
from sklearn.metrics import f1_score, classification_report
import os
import yaml

true_label_dir = '/home/chihung/mango-defect-detection-yolov5/mango-defects-data/labels/dev'
pred_label_dir = '/home/chihung/mango-defect-detection-yolov5/inference/output'
yaml_path = '/home/chihung/mango-defect-detection-yolov5/data/mango-defect.yaml'


def main():
    gt_vector_all = []
    pred_vector_all = []
    for label_file_name in os.listdir(true_label_dir):
        path_1 = os.path.join(true_label_dir, label_file_name)
        with open(path_1, 'r') as txt_file:
            gt_label_rows = txt_file.readlines()
        gt_vector = [0] * 5
        for class_id in list(set([int(row[0]) for row in gt_label_rows])):
            gt_vector[class_id] = 1

        path_2 = os.path.join(pred_label_dir, label_file_name)
        if not os.path.exists(path_2):
            pred_vector = [0] * 5
        else:
            with open(path_2, 'r') as txt_file:
                pred_label_rows = txt_file.readlines()
            pred_vector = [0] * 5
            for pred_class_id in list(set([int(row[0]) for row in pred_label_rows])):
                pred_vector[pred_class_id] = 1

        gt_vector_all.append(gt_vector)
        pred_vector_all.append(pred_vector)

        print(f"reading {path_2}")

    with open(yaml_path, 'r') as yaml_file:
        config = yaml.load(yaml_file,Loader=yaml.FullLoader)
    r=classification_report(gt_vector_all, pred_vector_all, target_names=config['names'],zero_division=0)
    print(r)


if __name__ == '__main__':
    main()
