import os
import numpy as np
import pandas as pd
from utils.dataload import load_nii

df = pd.read_csv("utils/data/train.csv")

if os.environ.get('MMsCardiac_DATA_PATH') is not None:
    MMs_DATA_PATH = os.environ.get('MMsCardiac_DATA_PATH')
else:
    assert False, "Please set the environment variable MMsCardiac_DATA_PATH. Read the README!"
base_dir = MMs_DATA_PATH

for index, row in df.iterrows():

    external_code = row['External code']
    current_slice = row["Slice"]
    current_phase = row["Phase"]

    if row["Labeled"]:

        mask_path = os.path.join(
            base_dir, "Training-corrected-contours", "Labeled",
            external_code, "{}_sa_gt.nii.gz".format(external_code)
        )
        mask = load_nii(mask_path)[0][..., current_slice, current_phase]

        img_path = os.path.join(
            base_dir, "Training-corrected", "Labeled",
            external_code, "{}_sa.nii.gz".format(external_code)
        )
        image = load_nii(img_path)[0][..., current_slice, current_phase]

        new_mask_dir_path = os.path.join(
            base_dir, "Training-corrected-contours", "Labeled_npy",
            external_code,
        )

        os.makedirs(new_mask_dir_path, exist_ok=True)

        np.save(os.path.join(
            new_mask_dir_path, "{}_slice{}_phase{}.npy".format(external_code, current_slice, current_phase)
        ), mask)

        new_img_dir_path = os.path.join(
            base_dir, "Training-corrected", "Labeled_npy",
            external_code,
        )

        os.makedirs(new_img_dir_path, exist_ok=True)

        np.save(os.path.join(
            new_img_dir_path, "{}_slice{}_phase{}.npy".format(external_code, current_slice, current_phase)
        ), image)

    else:

        img_path = os.path.join(
            base_dir, "Training-corrected", "Unlabeled",
            external_code, "{}_sa.nii.gz".format(external_code)
        )
        image = load_nii(img_path)[0][..., current_slice, current_phase]

        new_img_dir_path = os.path.join(
            base_dir, "Training-corrected", "Unlabeled_npy",
            external_code,
        )

        os.makedirs(new_img_dir_path, exist_ok=True)

        np.save(os.path.join(
            new_img_dir_path, "{}_slice{}_phase{}.npy".format(external_code, current_slice, current_phase)
        ), image)
