#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import pandas as pd
import seaborn as sns

from common import load_nii


def get_args():
    parser = argparse.ArgumentParser(description='M&Ms 2020 Challenge - Training info generation')
    parser.add_argument("--meta_graphs", action='store_true', help="Generate train meta information graphs")
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = get_args()

    if os.environ.get('MMsCardiac_DATA_PATH') is not None:
        MMs_DATA_PATH = os.environ.get('MMsCardiac_DATA_PATH')
    else:
        assert False, "Please set the environment variable MMs_DATA_PATH. Read the README!"

    data_info_path = os.path.join(MMs_DATA_PATH, "Training-corrected", "M&Ms Dataset Information.xlsx")
    data_info_df = pd.read_excel(data_info_path)

    data_info_df["Labeled"] = False
    for index, row in data_info_df.iterrows():
        if os.path.exists(os.path.join(MMs_DATA_PATH, "Training-corrected", "Labeled", row["External code"])):
            data_info_df.at[index, 'Labeled'] = True
        elif not os.path.exists(os.path.join(MMs_DATA_PATH, "Training-corrected", "Unlabeled", row["External code"])):
            assert False, "Not labeled or unlabeled case '{}'?".format(row["External code"])

    if args.meta_graphs:

        print("\nGenerating meta graphs...")

        # Analyze image shapes
        shapes = {}
        for idx in range(len(data_info_df)):
            if data_info_df.loc[idx]["Labeled"]:
                img_example = data_info_df.loc[idx]["External code"]
                img_example = os.path.join(
                    MMs_DATA_PATH, "Training-corrected", "Labeled",
                    img_example, "{}_sa.nii.gz".format(img_example)
                )
                img, _, _ = load_nii(img_example)
                if img.shape[0:2] not in shapes:
                    shapes[img.shape[0:2]] = 1
                else:
                    shapes[img.shape[0:2]] += 1

        # Convert key shapes (tuple) to string for visualization
        shapes = {str(k): int(v) for k, v in shapes.items()}

        # Plot meta information and save
        fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(18, 12))
        ax1[0].set_title("Centre Count")
        sns.countplot(data_info_df["Centre"], ax=ax1[0])
        ax1[1].set_title("Vendor Count")
        sns.countplot(data_info_df["Vendor"], ax=ax1[1])
        ax2[0].set_title("Labeled Count")
        sns.countplot(data_info_df["Labeled"], ax=ax2[0])
        ax2[0].set_title("Labeled Count")
        sns.countplot(data_info_df["Labeled"], ax=ax2[0])

        ax2[1].set_title("Shapes Count")
        plt.ylabel("Count")
        ax2[1] = plt.bar(shapes.keys(), shapes.values())
        plt.xticks(rotation=65)

        # Move a bit xticks/labels
        ax = plt.gca()
        trans = mtrans.Affine2D().translate(-10, 0)
        for t in ax.get_xticklabels():
            t.set_transform(t.get_transform() + trans)

        plt.savefig("train_meta_cases.png", dpi=150)
        print("Done!")

    print("\nGenerating Slice and Phase information...")

    data_info_df["Type"] = "Delete"
    data_info_df["Phase"] = -1

    for index, row in data_info_df.iterrows():

        if row["ED"] == -1:
            break  # We are at 'new' items

        data_info_df.loc[len(data_info_df)] = {
            "External code": row["External code"], "Vendor": row["Vendor"],
            "Centre": row["Centre"], "ED": -1, "ES": -1, "Labeled": row["Labeled"],
            "Type": "ED", "Phase": row["ED"]
        }

        data_info_df.loc[len(data_info_df)] = {
            "External code": row["External code"], "Vendor": row["Vendor"],
            "Centre": row["Centre"], "ED": -1, "ES": -1, "Labeled": row["Labeled"],
            "Type": "ES", "Phase": row["ES"]
        }

    data_info_df = data_info_df[data_info_df["Type"] != "Delete"]
    data_info_df = data_info_df.drop(columns=['ED', 'ES'])
    data_info_df = data_info_df.reset_index(drop=True)

    data_info_df["Slice"] = -1

    for index, row in data_info_df.iterrows():

        if row["Slice"] != -1:
            print("Done")
            break  # We are at 'new' items

        img_example = data_info_df.loc[index]["External code"]
        img_example = os.path.join(
            MMs_DATA_PATH, "Training-corrected", "Labeled" if row["Labeled"] else "Unlabeled",
            img_example, "{}_sa.nii.gz".format(img_example)
        )

        num_slices = load_nii(img_example)[0].shape[2]

        for c_slice in range(num_slices):
            data_info_df.loc[len(data_info_df)] = {
                "External code": row["External code"], "Vendor": row["Vendor"],
                "Centre": row["Centre"], "Labeled": row["Labeled"],
                "Type": row["Type"], "Phase": row["Phase"], "Slice": c_slice
            }

    data_info_df = data_info_df[data_info_df["Slice"] != -1]
    data_info_df = data_info_df.reset_index(drop=True)

    os.makedirs("utils/data", exist_ok=True)
    data_info_df.to_csv("utils/data/train.csv", index=False)

    print("{}Train information (csv) generated!\n".format("\n" if not args.meta_graphs else ""))
