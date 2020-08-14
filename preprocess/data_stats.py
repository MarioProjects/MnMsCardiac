#!/usr/bin/env python
# coding: utf-8

import os
import warnings

import numpy as np
import pandas as pd

from common import load_nii

warnings.filterwarnings('ignore')

if __name__ == "__main__":

    if not os.path.exists("utils/data/train.csv"):
        assert False, "Please generate train information (csv) first. Read the README!"

    if os.environ.get('MMsCardiac_DATA_PATH') is not None:
        MMs_DATA_PATH = os.environ.get('MMsCardiac_DATA_PATH')
    else:
        assert False, "Please set the environment variable MMs_DATA_PATH. Read the README!"

    data_info_df = pd.read_csv("utils/data/train.csv")

    print("\nCalculating global Mean and STD...")
    global_mean, global_std = [], []
    min_value, max_value = 10e5, 0
    for index, row in data_info_df.iterrows():

        if row["Labeled"]:
            img_path = os.path.join(
                MMs_DATA_PATH, "Training-corrected", "Labeled",
                row['External code'], "{}_sa.nii.gz".format(row['External code'])
            )
            current_img, _, _ = load_nii(img_path)

            global_mean.append(current_img.mean())
            global_std.append(np.std(current_img))

            if current_img.max() > max_value:
                max_value = current_img.max()
            if current_img.min() > min_value:
                min_value = current_img.min()

    print("Max: {}".format(max_value))
    print("Min: {}".format(min_value))

    print("Mean: {}".format(np.array(global_mean).mean()))
    print("STD: {}".format(np.array(global_std).mean()))

    print("\nFinish!\n")
