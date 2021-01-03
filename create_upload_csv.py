import csv
import os
from config import *


def main():
    with open(UPLOAD_CSV_TEMPLATE_PATH, 'r', newline='', encoding='UTF-8-sig') as csv_f:
        csv_content = list(csv.reader(csv_f))

    for idx, mango_result in enumerate(csv_content):
        if idx == 0:
            continue
        img_label_name = mango_result[0].replace(".jpg", ".txt")
        pred_label_txt = os.path.join(PRED_LABEL_DIR, img_label_name)
        if not os.path.exists(pred_label_txt):
            pred_vector = [0] * 5
        else:
            with open(pred_label_txt, 'r') as txt_file:
                pred_label_rows = txt_file.readlines()
            pred_vector = [0] * 5
            for pred_class_id in list(set([int(row[0]) for row in pred_label_rows])):
                pred_vector[pred_class_id] = 1

        mango_result[1:] = pred_vector

    with open(UPLOAD_CSV_PATH, 'w', newline='', encoding='UTF-8-sig') as res_csv:
        csv_ = csv.writer(res_csv)
        csv_.writerows(csv_content)
    print(f"Write upload format CSV to {UPLOAD_CSV_PATH}")


if __name__ == '__main__':
    main()
