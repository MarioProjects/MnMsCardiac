#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_nii

warnings.filterwarnings('ignore')


def get_args():
    parser = argparse.ArgumentParser(description='M&Ms 2020 Challenge - Training info generation')
    parser.add_argument("--save_dir", type=str, default="overlays", help="Where to save overlays")
    parser.add_argument("--overlay_alpha", type=float, default=0.25, help="Mask overlay alpha")
    arguments = parser.parse_args()
    return arguments


def save_overlay(img, mask, save_path, overlay_alpha):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax1.imshow(img, cmap="gray")
    ax2.imshow(mask, cmap="gray")

    masked = np.ma.masked_where(mask == 0, mask)
    ax3.imshow(img, cmap="gray")
    ax3.imshow(masked, 'jet', interpolation='bilinear', alpha=overlay_alpha)
    plt.savefig(save_path, dpi=150, pad_inches=0.2, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    args = get_args()

    if not os.path.exists("utils/data/train.csv"):
        assert False, "Please generate train information (csv) first. Read the README!"

    if os.environ.get('MMsCardiac_DATA_PATH') is not None:
        MMs_DATA_PATH = os.environ.get('MMsCardiac_DATA_PATH')
    else:
        assert False, "Please set the environment variable MMs_DATA_PATH. Read the README!"

    data_info_df = pd.read_csv("utils/data/train.csv")

    os.makedirs(args.save_dir, exist_ok=True)

    print("\nGenerating overlays...")
    for index, row in data_info_df.iterrows():
        os.makedirs(os.path.join(args.save_dir, row['External code']), exist_ok=True)

        if row["Labeled"]:
            mask_path = os.path.join(
                MMs_DATA_PATH, "Training-corrected-contours", "Labeled",
                row['External code'], "{}_sa_gt.nii.gz".format(row['External code'])
            )
            current_mask, _, _ = load_nii(mask_path)

            img_path = os.path.join(
                MMs_DATA_PATH, "Training-corrected", "Labeled",
                row['External code'], "{}_sa.nii.gz".format(row['External code'])
            )
            current_img, _, _ = load_nii(img_path)

            c_slice = row['Slice']
            c_phase = row['Phase']
            save_overlay(
                current_img[..., c_slice, c_phase], current_mask[..., c_slice, c_phase],
                os.path.join(args.save_dir, row['External code'], "slice{}_{}.png".format(c_slice, row['Type'])),
                args.overlay_alpha
            )

    print("\nTrain overlays generated!\n")
