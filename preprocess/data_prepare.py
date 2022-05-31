"""
Prepare data to below for training

- id_label_transcript
    id label transcript
    - train
    - test

- id_path
    id path

- id_label_trans_prob
    id label transcript prob1 prob2 ...

"""

import os
import easyocr
import os
import glob
from pathlib import Path
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

def prepar_data(fine_grain_dir, label2num):
    """
    Prepare id_lab_trans, id_path csv for training
    :param fine_grain_dir: should follow below structure
    ./
        category1/
            1.jpg
            2.jpg
            ...
        category2/
            ...
        label2num: dict
    :return:
    """
    # got reader
    reader = easyocr.Reader(['en'])    # need to run only once to load model into memory
    df_no_prob = pd.DataFrame()
    df_id_path = pd.DataFrame()
    ls_no_prob = []
    ls_id_path = []

    for r, s, f in os.walk(fine_grain_dir):
        if len(s) >= 1 and len(f) == 0:  # first root
            pass
        if len(s) == 0 and len(f) > 0:   # last layer
            category_label = os.path.basename(r)
            category_num = label2num[category_label]
            for img in f:
                #file, ext = os.path.splitext(img)
                img_p = os.path.join(r, img)
                print("start {}".format(img_p))
                try:
                    result = reader.readtext(img_p)
                except:
                    break
                trans = [a[1].lower() for a in result]
                prob = [a[2] for a in result]
                ls_id_path.append([category_label + "_" + img[:-4], img_p])
                ls_no_prob.append([category_label + "_" + img[:-4],
                             category_num,
                             " ".join(trans)])
    df_no_prob = df_no_prob.append(ls_no_prob)
    df_id_path = df_id_path.append(ls_id_path)

    out_1 = os.path.join(os.path.dirname(fine_grain_dir), "id_lab_trans.csv")
    df_no_prob.to_csv(out_1, index=False, sep=",", header=["id", "lab", "trans"])

    out_2 = os.path.join(os.path.dirname(fine_grain_dir), "id_path.csv")
    df_id_path.to_csv(out_2, index=False, sep=",", header=["id", "path"])


def separte_train_test(id_lab_trans_f, id_path_f, img_dir_traintest):
    """
    Separate img, id_path, id_lab_trans to train testr
    :param id_lab_trans_f:
    :param id_path_f:
    :param img_dir_traintest:
    :return:
    """
    # separate id_lab_trans to train/test
    train_f = os.path.join(os.path.dirname(id_lab_trans_f), "id_lab_trans_train.csv")
    test_f = os.path.join(os.path.dirname(id_lab_trans_f), "id_lab_trans_test.csv")
    train_path_f = os.path.join(os.path.dirname(id_path_f), "id_path_train.csv")
    test_path_f = os.path.join(os.path.dirname(id_path_f), "id_path_test.csv")

    df = pd.read_csv(id_lab_trans_f, sep=",")
    df_id_path = pd.read_csv(id_path_f, sep=",")

    train, test, train_path, test_path = train_test_split(df, df_id_path, test_size=0.2, random_state=44)
    train.to_csv(train_f, index=False, header=["id", "lab", "trans"], sep=",")
    test.to_csv(test_f, index=False, header=["id", "lab", "trans"], sep=",")
    train_path.to_csv(train_path_f, index=False, header=["id", "path"], sep=",")
    test_path.to_csv(test_path_f, index=False, header=["id", "path"], sep=",")


    # separate image dir to train/tst
    train_sub_dir = os.path.join(img_dir_traintest, "train")
    test_sub_dir = os.path.join(img_dir_traintest, "test")

    Path(train_sub_dir).mkdir(parents=True, exist_ok=True)
    Path(test_sub_dir).mkdir(parents=True, exist_ok=True)
    id_path_dict = df_id_path.set_index("id").T.to_dict()

    for id in train["id"]:
        p = id_path_dict[id]
        os.system("cp {} {}".format(id_path_dict[id]["path"], train_sub_dir))
    for id in test["id"]:
        os.system("cp {} {}".format(id_path_dict[id]["path"], test_sub_dir))


if __name__ == '__main__':
    fine_grain_dir = "/home/Data/image_data/fine_grain_img_clean"
    label2num = {'strike': 0,
                 'feminist': 1,
                 'racism': 2,
                 'anti_politician': 3,
                 'other': 4}
    #prepar_data(fine_grain_dir, label2num)

    id_lab_trans = "/home/Data/image_data/id_lab_trans.csv"
    id_path = "/home/Data/image_data/id_path.csv"
    img_dir_traintest = "/home/Data/image_data/fine_grain_img_traintest"
    separte_train_test(id_lab_trans, id_path, img_dir_traintest)