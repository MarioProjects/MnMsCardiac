import nibabel as nib
import albumentations
import numpy as np
import os
import pandas as pd
import torch
import copy
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, DataLoader
from utils.data_augmentation import common_test_augmentation

VENDOR_MAP = {'A': 0, 'B': 1, 'C': 2}


def add_depth_channels(image_tensor):
    _, h, w = image_tensor.size()
    image = torch.zeros([3, h, w])
    image[0] = image_tensor
    for row, const in enumerate(np.linspace(0, 1, h)):
        image[1, row, :] = const
    image[2] = image[0] * image[1]
    return image


def apply_augmentations(image, transform, img_transform, mask=None):
    if transform:
        if mask is not None:
            augmented = transform(image=image, mask=mask)
            mask = augmented['mask']
        else:
            augmented = transform(image=image)

        image = augmented['image']

    if img_transform:
        augmented = img_transform(image=image)
        image = augmented['image']

    return image, mask


def apply_normalization(image, normalization_type):
    """
    https://www.statisticshowto.com/normalized/
    :param image:
    :param normalization_type:
    :return:
    """
    if normalization_type == "none":
        return image
    elif normalization_type == "reescale":
        image_min = image.min()
        image_max = image.max()
        image = (image - image_min) / (image_max - image_min)
        return image
    elif normalization_type == "standardize":
        mean = np.mean(image)
        std = np.std(image)
        image = image - mean
        image = image / std
        return image
    assert False, "Unknown normalization: '{}'".format(normalization_type)


def apply_torch_normalization(image, normalization_type):
    """
    https://www.statisticshowto.com/normalized/
    :param image:
    :param normalization_type:
    :return:
    """
    if normalization_type == "none":
        return image
    elif normalization_type == "reescale":
        image_min = image.min()
        image_max = image.max()
        image = (image - image_min) / (image_max - image_min)
        return image
    elif normalization_type == "standardize":
        mean = image.mean().detach()
        std = image.std().detach()
        image = image - mean
        image = image / (std + 1e-10)
        return image
    assert False, "Unknown normalization: '{}'".format(normalization_type)


def apply_batch_local_torch_normalization(batch, normalization_type):
    with torch.no_grad():
        for indx, image in enumerate(batch):
            batch[indx, ...] = apply_torch_normalization(image, normalization_type)
    return batch


def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everyting needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header
    :param img_path: (string) Path of the 'nii' or 'nii.gz' image file name
    :return: Three element, the first is a numpy array of the image values (height, width, slices, phases),
             ## (No) the second is the affine transformation of the image, and the
             ## (No) last one is the header of the image.
    """
    nimg = nib.load(img_path)
    return np.asanyarray(nimg.dataobj), nimg.affine, nimg.header


def save_nii(img_path, data, affine, header):
    """
    Save a nifty file
    :param img_path: Path to save image file name
    :param data: numpy array of the image values
    :param affine: nii affine transformation of the image
    :param header: nii header of the image
    :return:(void)
    """
    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)


class MMsDataset(Dataset):
    """
    Dataset for Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation Challenge (M&Ms).
    """

    def __init__(self, mode, transform, img_transform, train_fold=0, val_fold=0, folding_system="vendor",
                 label_type="mask", add_depth=False, normalization="normalize", get_id=False):
        """
        :param mode: (string) Dataset mode in ["train", "validation"]
        :param transform: (list) List of albumentations applied to image and mask
        :param img_transform: (list) List of albumentations applied to image only
        :param train_fold: (int) Fold number for k fold validation
        :param folding_system: (str) How to create data folds
        :param label_type: (str) One of 'mask' - 'vendor_label'
        :param add_depth: (bool) If apply image transformation 1 to 3 channels or not
        :param normalization: (str) Normalization mode. One of 'reescale', 'standardize', 'global_standardize'
        """

        if mode not in ["train", "validation"]:
            assert False, "Unknown mode '{}'".format(mode)

        if os.path.exists("utils/data/train.csv"):
            data_info_path = "utils/data/train.csv"
        elif os.path.exists("../utils/data/train.csv"):  # Notebooks execution
            data_info_path = "../utils/data/train.csv"
        else:
            assert False, "Please generate train information (csv) first. Read the README!"

        if os.environ.get('MMsCardiac_DATA_PATH') is not None:
            MMs_DATA_PATH = os.environ.get('MMsCardiac_DATA_PATH')
        else:
            assert False, "Please set the environment variable MMsCardiac_DATA_PATH. Read the README!"

        self.base_dir = MMs_DATA_PATH
        self.mode = mode
        self.add_depth = add_depth
        self.normalization = normalization
        self.label_type = label_type
        self.get_id = get_id

        df = pd.read_csv(data_info_path)

        if label_type == "mask":
            # Select only labeled data
            df = df[df['Labeled']]
            df = df.reset_index(drop=True)

        if folding_system == "vendor":

            if mode == "train":
                print("Possible vendor folds: {} - Using Vendor '{}'".format(df["Vendor"].unique(), train_fold))
                df = df.loc[df["Vendor"] == train_fold].reset_index(drop=True)

                # Remove some patients that will be used for validation partition
                g = df.groupby(['External code'])
                np.random.seed(42)
                a = np.arange(g.ngroups)
                np.random.shuffle(a)
                subpart = df[g.ngroup().isin(a[:11])]  # we remove 11 random patients

                df.drop(subpart.index, inplace=True)

            elif mode == "validation":
                print("Possible vendor folds: {} - Using Vendor '{}'".format(df["Vendor"].unique(), val_fold))
                val_df = df.loc[df["Vendor"] == val_fold].reset_index(drop=True)
                train_df = df.loc[df["Vendor"] == train_fold].reset_index(drop=True)

                # Add some patients from train partition
                g = train_df.groupby(['External code'])
                np.random.seed(42)
                a = np.arange(g.ngroups)
                np.random.shuffle(a)
                subpart = train_df[g.ngroup().isin(a[:11])]  # we take 11 random train patients

                df = pd.concat([val_df, subpart])

            else:
                assert False, "Not implemented folding system vendor with mode '{}'".format(mode)

        elif folding_system == "patient":

            if label_type == "vendor_label_binary":  # We want do binary classification
                print("\n---------------------------------------------------")
                print("Note: Using only Vendor 'A' and 'B' - Binary classification")
                print("---------------------------------------------------\n")
                df = df.loc[(df["Vendor"] == "A") | (df["Vendor"] == "B")].reset_index(drop=True)

            train_fold, fold_splits = int(train_fold), 5

            if train_fold >= fold_splits:
                assert False, "Wrong Fold number (can't bre greater than total folds)"

            skf = GroupKFold(n_splits=fold_splits)
            target = df["Vendor"]

            # Get current fold data
            for fold_indx, (train_index, val_index) in enumerate(skf.split(np.zeros(len(target)),
                                                                           target, groups=df["External code"])):
                if fold_indx == train_fold:  # If current iteration is the desired fold, take it!
                    if mode == "train":
                        df = df.loc[train_index]
                    elif mode == "validation":
                        df = df.loc[val_index]

        elif folding_system == "all":
            print("\n-------------------------")
            print("USING ALL DATA FOR TRAINING")
            print("-------------------------\n")
            if mode == "validation":
                df = df.sample(frac=0.15, replace=True, random_state=2020)

        else:
            assert False, "Unknown folding system '{}'".format(folding_system)

        self.df = df.reset_index(drop=True)

        self.transform = albumentations.Compose(transform)
        self.img_transform = albumentations.Compose(img_transform)

    def __len__(self):
        return len(self.df)

    def mask_problem(self, external_code, current_slice, current_phase, data_parent):

        mask_parent = os.path.join("Training-corrected-contours", "Labeled_npy", external_code)
        mask_path = os.path.join(
            self.base_dir, mask_parent, "{}_slice{}_phase{}.npy".format(external_code, current_slice, current_phase)
        )
        mask = np.load(mask_path)

        img_path = os.path.join(
            self.base_dir, data_parent,
            external_code, "{}_slice{}_phase{}.npy".format(external_code, current_slice, current_phase)
        )
        image = np.load(img_path)

        original_image = copy.deepcopy(image)
        original_mask = copy.deepcopy(mask)

        if self.mode == "validation":  # We don't want modify mask in validation mode
            image, mask = apply_augmentations(image, self.transform, self.img_transform, mask)
        elif self.mode == "train":
            image, mask = apply_augmentations(image, self.transform, self.img_transform, mask)
        else:
            assert False, "Not implemented mode: {}".format(self.mode)

        image = apply_normalization(image, self.normalization)

        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        if self.add_depth:
            image = add_depth_channels(image)

        mask = torch.from_numpy(mask).long()

        if self.mode == "validation" or self.get_id:  # We don't want modify mask in validation mode
            img_id = "{}_slice{}_phase{}".format(external_code, current_slice, current_phase)
            return [image, original_image, original_mask, mask, img_id]

        return [image, mask]

    def vendor_label_problem(self, external_code, current_slice, current_phase, vendor_label, data_parent):
        img_path = os.path.join(
            self.base_dir, data_parent,
            external_code, "{}_slice{}_phase{}.npy".format(external_code, current_slice, current_phase)
        )
        image = np.load(img_path)
        original_image = copy.deepcopy(image)

        image, _ = apply_augmentations(image, self.transform, self.img_transform, None)

        image = apply_normalization(image, self.normalization)

        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        if self.add_depth:
            image = add_depth_channels(image)

        if self.mode == "validation" or self.get_id:  # We don't want modify mask in validation mode
            img_id = "{}_slice{}_phase{}".format(external_code, current_slice, current_phase)
            return [image, VENDOR_MAP[vendor_label], original_image, img_id]

        return [image, VENDOR_MAP[vendor_label]]

    def __getitem__(self, idx):

        external_code = self.df.loc[idx]['External code']
        current_slice = self.df.loc[idx]['Slice']
        current_phase = self.df.loc[idx]['Phase']
        vendor_label = self.df.loc[idx]['Vendor']

        if self.df.loc[idx]['Labeled']:
            data_parent = os.path.join("Training-corrected", "Labeled_npy")
        else:
            data_parent = os.path.join("Training-corrected", "Unlabeled_npy")

        if self.label_type == "mask":
            return self.mask_problem(external_code, current_slice, current_phase, data_parent)
        elif self.label_type == "vendor_label_binary" or self.label_type == "vendor_label_full":
            return self.vendor_label_problem(external_code, current_slice, current_phase, vendor_label, data_parent)
        else:
            assert False, "Unknown label type '{}'".format(self.label_type)


class MMsWeaklyDataset(Dataset):
    """
    Weakly Labeled Dataset for Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation Challenge (M&Ms).
    """

    def __init__(self, mode, transform, img_transform, train_fold=0, val_fold=0, folding_system="vendor",
                 label_type="mask", add_depth=False, normalization="normalize", get_id=False, exclusion_patients=[]):
        """
        :param mode: (string) Dataset mode in ["train", "validation"]
        :param transform: (list) List of albumentations applied to image and mask
        :param img_transform: (list) List of albumentations applied to image only
        :param train_fold: (int) Fold number for k fold validation
        :param folding_system: (str) How to create data folds
        :param label_type: (str) One of 'mask' - 'vendor_label'
        :param add_depth: (bool) If apply image transformation 1 to 3 channels or not
        :param normalization: (str) Normalization mode. One of 'reescale', 'standardize', 'global_standardize'
        """

        if mode not in ["train", "validation"]:
            assert False, "Unknown mode '{}'".format(mode)

        if os.path.exists("utils/data/weakly.csv"):
            data_info_path = "utils/data/weakly.csv"
        elif os.path.exists("../utils/data/weakly.csv"):  # Notebooks execution
            data_info_path = "../utils/data/weakly.csv"
        else:
            assert False, "Please generate train information (csv) first. Read the README!"

        if os.environ.get('MMsCardiac_DATA_PATH') is not None:
            MMs_DATA_PATH = os.environ.get('MMsCardiac_DATA_PATH')
        else:
            assert False, "Please set the environment variable MMsCardiac_DATA_PATH. Read the README!"

        self.base_dir = MMs_DATA_PATH
        self.mode = mode
        self.add_depth = add_depth
        self.normalization = normalization
        self.label_type = label_type
        self.get_id = get_id

        df = pd.read_csv(data_info_path)

        if folding_system == "vendor":

            if mode == "train":
                print("Possible vendor folds: {} - Using Vendor '{}'".format(df["Vendor"].unique(), train_fold))
                df = df.loc[df["Vendor"] == train_fold].reset_index(drop=True)

                # Remove some patients that will be used for validation partition
                g = df.groupby(['External code'])
                np.random.seed(42)
                a = np.arange(g.ngroups)
                np.random.shuffle(a)
                subpart = df[g.ngroup().isin(a[:11])]  # we remove 11 random patients

                df.drop(subpart.index, inplace=True)

            elif mode == "validation":
                print("Possible vendor folds: {} - Using Vendor '{}'".format(df["Vendor"].unique(), val_fold))
                val_df = df.loc[df["Vendor"] == val_fold].reset_index(drop=True)
                train_df = df.loc[df["Vendor"] == train_fold].reset_index(drop=True)

                # Add some patients from train partition
                g = train_df.groupby(['External code'])
                np.random.seed(42)
                a = np.arange(g.ngroups)
                np.random.shuffle(a)
                subpart = train_df[g.ngroup().isin(a[:11])]  # we take 11 random train patients

                df = pd.concat([val_df, subpart])

            else:
                assert False, "Not implemented folding system vendor with mode '{}'".format(mode)

        elif folding_system == "patient":

            if label_type == "vendor_label_binary":  # We want do binary classification
                print("\n---------------------------------------------------")
                print("Note: Using only Vendor 'A' and 'B' - Binary classification")
                print("---------------------------------------------------\n")
                df = df.loc[(df["Vendor"] == "A") | (df["Vendor"] == "B")].reset_index(drop=True)

            train_fold, fold_splits = int(train_fold), 5

            if train_fold >= fold_splits:
                assert False, "Wrong Fold number (can't bre greater than total folds)"

            skf = GroupKFold(n_splits=fold_splits)
            target = df["Vendor"]

            # Get current fold data
            for fold_indx, (train_index, val_index) in enumerate(skf.split(np.zeros(len(target)),
                                                                           target, groups=df["External code"])):
                if fold_indx == train_fold:  # If current iteration is the desired fold, take it!
                    if mode == "train":
                        df = df.loc[train_index]
                    elif mode == "validation":
                        df = df.loc[val_index]

        elif folding_system == "all":
            print("\n-------------------------")
            print("USING ALL DATA FOR TRAINING")
            print("-------------------------\n")
            if mode == "validation":
                df = df.sample(frac=0.1, replace=True, random_state=2020)

        elif folding_system == "exclusion":
            df = df.loc[~df["External code"].isin(exclusion_patients)]
            if mode == "validation":
                df = df.sample(frac=0.1, replace=True, random_state=2020)

        else:
            assert False, "Unknown folding system '{}'".format(folding_system)

        self.df = df.reset_index(drop=True)

        self.transform = albumentations.Compose(transform)
        self.img_transform = albumentations.Compose(img_transform)

    def __len__(self):
        return len(self.df)

    def mask_problem(self, external_code, current_slice, current_phase, data_parent):

        mask_path = os.path.join(
            self.base_dir, "Weakly", "Masks", data_parent,
            "{}_slice{}_phase{}.npy".format(external_code, current_slice, current_phase)
        )
        mask = np.load(mask_path)

        img_path = os.path.join(
            self.base_dir, "Weakly", "Images", data_parent,
            "{}_slice{}_phase{}.npy".format(external_code, current_slice, current_phase)
        )
        image = np.load(img_path)

        original_image = copy.deepcopy(image)
        original_mask = copy.deepcopy(mask)

        if self.mode == "validation":  # We don't want modify mask in validation mode
            image, mask = apply_augmentations(image, self.transform, self.img_transform, mask)
        elif self.mode == "train":
            image, mask = apply_augmentations(image, self.transform, self.img_transform, mask)
        else:
            assert False, "Not implemented mode: {}".format(self.mode)

        image = apply_normalization(image, self.normalization)

        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        if self.add_depth:
            image = add_depth_channels(image)

        mask = torch.from_numpy(mask).long()

        if self.mode == "validation" or self.get_id:  # We don't want modify mask in validation mode
            img_id = "{}_slice{}_phase{}".format(external_code, current_slice, current_phase)
            return [image, original_image, original_mask, mask, img_id]

        return [image, mask]

    def vendor_label_problem(self, external_code, current_slice, current_phase, vendor_label, data_parent):
        img_path = os.path.join(
            self.base_dir, "Weakly", "Images", data_parent,
            "{}_slice{}_phase{}.npy".format(external_code, current_slice, current_phase)
        )
        image = np.load(img_path)
        original_image = copy.deepcopy(image)

        image, _ = apply_augmentations(image, self.transform, self.img_transform, None)

        image = apply_normalization(image, self.normalization)

        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        if self.add_depth:
            image = add_depth_channels(image)

        if self.mode == "validation" or self.get_id:  # We don't want modify mask in validation mode
            img_id = "{}_slice{}_phase{}".format(external_code, current_slice, current_phase)
            return [image, VENDOR_MAP[vendor_label], original_image, img_id]

        return [image, VENDOR_MAP[vendor_label]]

    def __getitem__(self, idx):

        external_code = self.df.loc[idx]['External code']
        current_slice = self.df.loc[idx]['Slice']
        current_phase = self.df.loc[idx]['Phase']
        vendor_label = self.df.loc[idx]['Vendor']

        if self.df.loc[idx]['FromLabeled']:
            data_parent = "Labeled"
        else:
            data_parent = "Unlabeled"

        if self.label_type == "mask":
            return self.mask_problem(external_code, current_slice, current_phase, data_parent)
        elif self.label_type == "vendor_label_binary" or self.label_type == "vendor_label_full":
            return self.vendor_label_problem(external_code, current_slice, current_phase, vendor_label, data_parent)
        else:
            assert False, "Unknown label type '{}'".format(self.label_type)


class MMsEntropyDataset(Dataset):
    """
    Weakly Labeled Dataset for Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation Challenge (M&Ms).
    """

    def __init__(self, mode, transform, img_transform, train_fold=0, val_fold=0, folding_system="vendor",
                 label_type="mask", add_depth=False, normalization="normalize", get_id=False, exclusion_patients=[]):
        """
        :param mode: (string) Dataset mode in ["train", "validation"]
        :param transform: (list) List of albumentations applied to image and mask
        :param img_transform: (list) List of albumentations applied to image only
        :param train_fold: (int) Fold number for k fold validation
        :param folding_system: (str) How to create data folds
        :param label_type: (str) One of 'mask' - 'vendor_label'
        :param add_depth: (bool) If apply image transformation 1 to 3 channels or not
        :param normalization: (str) Normalization mode. One of 'reescale', 'standardize', 'global_standardize'
        """

        if mode not in ["train", "validation"]:
            assert False, "Unknown mode '{}'".format(mode)

        if os.path.exists("utils/data/entropy_dataset.csv"):
            data_info_path = "utils/data/entropy_dataset.csv"
        elif os.path.exists("../utils/data/entropy_dataset.csv"):  # Notebooks execution
            data_info_path = "../utils/data/entropy_dataset.csv"
        else:
            assert False, "Please generate train information (csv) first. Read the README!"

        if os.environ.get('MMsCardiac_DATA_PATH') is not None:
            MMs_DATA_PATH = os.environ.get('MMsCardiac_DATA_PATH')
        else:
            assert False, "Please set the environment variable MMsCardiac_DATA_PATH. Read the README!"

        self.base_dir = MMs_DATA_PATH
        self.mode = mode
        self.add_depth = add_depth
        self.normalization = normalization
        self.label_type = label_type
        self.get_id = get_id

        df = pd.read_csv(data_info_path)

        if folding_system == "vendor":

            if mode == "train":
                print("Possible vendor folds: {} - Using Vendor '{}'".format(df["VendorInit"].unique(), train_fold))
                df = df.loc[df["VendorInit"] == train_fold].reset_index(drop=True)

                # Remove some patients that will be used for validation partition
                g = df.groupby(['External code'])
                np.random.seed(42)
                a = np.arange(g.ngroups)
                np.random.shuffle(a)
                subpart = df[g.ngroup().isin(a[:11])]  # we remove 11 random patients

                df.drop(subpart.index, inplace=True)

            elif mode == "validation":
                print("Possible vendor folds: {} - Using Vendor '{}'".format(df["VendorInit"].unique(), val_fold))
                val_df = df.loc[df["VendorInit"] == val_fold].reset_index(drop=True)
                train_df = df.loc[df["VendorInit"] == train_fold].reset_index(drop=True)

                # Add some patients from train partition
                g = train_df.groupby(['External code'])
                np.random.seed(42)
                a = np.arange(g.ngroups)
                np.random.shuffle(a)
                subpart = train_df[g.ngroup().isin(a[:11])]  # we take 11 random train patients

                df = pd.concat([val_df, subpart])

            else:
                assert False, "Not implemented folding system vendor with mode '{}'".format(mode)

        elif folding_system == "patient":

            if label_type == "vendor_label_binary":  # We want do binary classification
                print("\n---------------------------------------------------")
                print("Note: Using only Vendor 'A' and 'B' - Binary classification")
                print("---------------------------------------------------\n")
                df = df.loc[(df["VendorInit"] == "A") | (df["VendorInit"] == "B")].reset_index(drop=True)

            train_fold, fold_splits = int(train_fold), 5

            if train_fold >= fold_splits:
                assert False, "Wrong Fold number (can't bre greater than total folds)"

            skf = GroupKFold(n_splits=fold_splits)
            target = df["VendorInit"]

            # Get current fold data
            for fold_indx, (train_index, val_index) in enumerate(skf.split(np.zeros(len(target)),
                                                                           target, groups=df["External code"])):
                if fold_indx == train_fold:  # If current iteration is the desired fold, take it!
                    if mode == "train":
                        df = df.loc[train_index]
                    elif mode == "validation":
                        df = df.loc[val_index]

        elif folding_system == "all":
            print("\n-------------------------")
            print("USING ALL DATA FOR TRAINING")
            print("-------------------------\n")
            if mode == "validation":
                df = df.sample(frac=0.1, replace=True, random_state=2020)

        elif folding_system == "exclusion":
            df = df.loc[~df["External code"].isin(exclusion_patients)]
            if mode == "validation":
                df = df.sample(frac=0.1, replace=True, random_state=2020)

        else:
            assert False, "Unknown folding system '{}'".format(folding_system)

        self.df = df.reset_index(drop=True)

        self.transform = albumentations.Compose(transform)
        self.img_transform = albumentations.Compose(img_transform)

    def __len__(self):
        return len(self.df)

    def mask_problem(self, external_code, current_slice, current_phase, init_vendor_label, target_vendor_label):

        mask_parent = os.path.join("Training-corrected-contours", "Labeled_npy", external_code)
        mask_path = os.path.join(
            self.base_dir, mask_parent, "{}_slice{}_phase{}.npy".format(external_code, current_slice, current_phase)
        )
        mask = np.load(mask_path)

        img_path = os.path.join(
            self.base_dir, "Entropy-samples",
            "{}_slice{}_phase{}_{}to{}_standardized.npy".format(
                external_code, current_slice, current_phase, init_vendor_label, target_vendor_label.capitalize()
            )
        )
        image = np.load(img_path)

        original_image = copy.deepcopy(image)
        original_mask = copy.deepcopy(mask)

        # Images were processed and all are 224x224... We need to perform masks adapatation first!
        common_reshape = common_test_augmentation(224)
        mask = albumentations.Compose(common_reshape)(image=mask)["image"]

        if self.mode == "validation":  # We don't want modify mask in validation mode
            image, mask = apply_augmentations(image, self.transform, self.img_transform, mask)
        elif self.mode == "train":
            image, mask = apply_augmentations(image, self.transform, self.img_transform, mask)
        else:
            assert False, "Not implemented mode: {}".format(self.mode)

        image = apply_normalization(image, self.normalization)

        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        if self.add_depth:
            image = add_depth_channels(image)

        mask = torch.from_numpy(mask).long()

        if self.mode == "validation" or self.get_id:  # We don't want modify mask in validation mode
            img_id = "{}_slice{}_phase{}".format(external_code, current_slice, current_phase)
            return [image, original_image, original_mask, mask, img_id]

        return [image, mask]

    def vendor_label_problem(self, external_code, current_slice, current_phase, init_vendor_label, target_vendor_label):
        img_path = os.path.join(
            self.base_dir, "Entropy-samples",
            "{}_slice{}_phase{}_{}to{}_standardized.npy".format(
                external_code, current_slice, current_phase, init_vendor_label, target_vendor_label.capitalize()
            )
        )
        image = np.load(img_path)
        original_image = copy.deepcopy(image)

        image, _ = apply_augmentations(image, self.transform, self.img_transform, None)

        image = apply_normalization(image, self.normalization)

        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        if self.add_depth:
            image = add_depth_channels(image)

        if self.mode == "validation" or self.get_id:  # We don't want modify mask in validation mode
            img_id = "{}_slice{}_phase{}".format(external_code, current_slice, current_phase)
            return [image, VENDOR_MAP[init_vendor_label], original_image, img_id]

        return [image, VENDOR_MAP[init_vendor_label]]

    def __getitem__(self, idx):

        external_code = self.df.loc[idx]['External code']
        current_slice = self.df.loc[idx]['Slice']
        current_phase = self.df.loc[idx]['Phase']
        init_vendor_label = self.df.loc[idx]['VendorInit']
        target_vendor_label = self.df.loc[idx]['VendorTarget']

        if self.label_type == "mask":
            return self.mask_problem(external_code, current_slice, current_phase, init_vendor_label,
                                     target_vendor_label)
        elif self.label_type == "vendor_label_binary" or self.label_type == "vendor_label_full":
            return self.vendor_label_problem(external_code, current_slice, current_phase, vendor_label)
        else:
            assert False, "Unknown label type '{}'".format(self.label_type)


class MMsSubmissionDataset(Dataset):
    """
    Submission Dataset for Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation Challenge (M&Ms).
    """

    def __init__(self, input_dir, transform, img_transform, add_depth=False, normalization="standardize"):
        """
        :param input_dir: (string) Path with volume folders (usually where info.csv is located)
        :param transform: (list) List of albumentations applied to image and mask
        :param img_transform: (list) List of albumentations applied to image only
        :param add_depth: (bool) If apply image transformation 1 to 3 channels or not
        :param normalization: (str) Normalization mode. One of 'reescale', 'standardize', 'global_standardize'
        """

        input_dir = os.path.join(input_dir, "mnms")  # Default folder structure has 'mnms' as parent directory
        info_csv = os.path.join(input_dir, "info.csv")
        if not os.path.exists(info_csv):
            assert False, "Cannot find info.csv at input path {}!".format(input_dir)
        self.df = pd.read_csv(info_csv)

        self.base_dir = input_dir
        self.add_depth = add_depth
        self.normalization = normalization
        self.transform = albumentations.Compose(transform)
        if not img_transform:
            self.img_transform = None
        else:
            self.img_transform = albumentations.Compose(img_transform)

    def __len__(self):
        return len(self.df)

    def apply_volume_augmentations(self, list_images):
        """
        Apply same augmentations to volume images
        :param list_images: (array) [num_images, height, width] Images to transform
        :return: (array) [num_images, height, width] Transformed Images
        """
        if self.img_transform:
            # Independent augmentations...
            for indx, img in enumerate(list_images):
                augmented = self.img_transform(image=img)
                list_images[indx] = augmented['image']

        if self.transform:
            # All augmentations applied in same proportion and values
            imgs_ids = ["image"] + ["image{}".format(idx + 2) for idx in range(len(list_images) - 1)]
            aug_args = dict(zip(imgs_ids, list_images))

            pair_ids_imgs = ["image{}".format(idx + 2) for idx in range(len(list_images) - 1)]
            base_id_imgs = ["image"] * len(pair_ids_imgs)
            list_additional_targets = dict(zip(pair_ids_imgs, base_id_imgs))

            volumetric_aug = albumentations.Compose(self.transform, additional_targets=list_additional_targets)
            augmented = volumetric_aug(**aug_args)

            list_images = np.stack([augmented[img] for img in imgs_ids])

        return list_images

    def apply_volume_normalization(self, list_images):
        for indx, image in enumerate(list_images):
            list_images[indx, ...] = apply_normalization(image, self.normalization)
        return list_images

    def add_volume_depth_channels(self, list_images):
        b, d, h, w = list_images.shape
        new_list_images = torch.empty((b, 3, h, w))
        for indx, image in enumerate(list_images):
            new_list_images[indx, ...] = add_depth_channels(image)
        return new_list_images

    def simple_collate(self, batch):
        return batch[0]

    def __getitem__(self, idx):

        external_code = self.df.loc[idx]['External code']
        ed_phase = self.df.loc[idx]["ED"]
        es_phase = self.df.loc[idx]["ES"]

        img_path = os.path.join(
            self.base_dir, external_code, "{}_sa.nii.gz".format(external_code)
        )
        volume, affine, header = load_nii(img_path)
        ed_volume = volume[..., :, ed_phase]
        es_volume = volume[..., :, es_phase]

        original_ed = copy.deepcopy(ed_volume)
        original_es = copy.deepcopy(es_volume)
        initial_shape = es_volume.shape

        ed_volume = ed_volume.transpose(2, 0, 1)
        es_volume = es_volume.transpose(2, 0, 1)

        ed_volume = self.apply_volume_augmentations(ed_volume)
        es_volume = self.apply_volume_augmentations(es_volume)

        ed_volume = self.apply_volume_normalization(ed_volume)
        es_volume = self.apply_volume_normalization(es_volume)

        # We have to stack volume as batch
        ed_volume = np.expand_dims(ed_volume, axis=1)
        es_volume = np.expand_dims(es_volume, axis=1)

        ed_volume = torch.from_numpy(ed_volume)
        es_volume = torch.from_numpy(es_volume)

        if self.add_depth:
            ed_volume = self.add_volume_depth_channels(ed_volume)
            es_volume = self.add_volume_depth_channels(es_volume)

        img_id = "{}".format(external_code)

        return [ed_volume, es_volume, affine, header, initial_shape, img_id, original_ed, original_es]


class BalancedConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.max_len = max(len(d) for d in self.datasets)
        self.min_len = min(len(d) for d in self.datasets)

    def __getitem__(self, i):
        return tuple(d[i % len(d)] for d in self.datasets)

    def masks_collate(self, batch):
        # Only image - mask
        images, masks = [], []
        for item in range(len(batch)):
            for c_dataset in range(len(batch[item])):
                images.append(batch[item][c_dataset][0])
                masks.append(batch[item][c_dataset][1])
        images = torch.stack(images)
        masks = torch.stack(masks)
        return images, masks

    def __len__(self):
        return self.max_len


def dataset_selector(train_aug, train_aug_img, val_aug, args):
    if args.dataset == "mnms":
        train_dataset = MMsDataset(
            mode="train", transform=train_aug, img_transform=train_aug_img,
            folding_system=args.fold_system, normalization=args.normalization, label_type=args.label_type,
            train_fold=args.data_fold, val_fold=args.data_fold_validation, add_depth=args.add_depth
        )

        val_dataset = MMsDataset(
            mode="validation", transform=val_aug, img_transform=[],
            folding_system=args.fold_system, normalization=args.normalization, label_type=args.label_type,
            train_fold=args.data_fold, val_fold=args.data_fold_validation, add_depth=args.add_depth
        )

        print("[{}] Total {} Images: {}".format(args.dataset, "Train Dataset", len(train_dataset)))
        print("[{}] Total {} Images: {}".format(args.dataset, "Validation Dataset", len(val_dataset)))

    if args.dataset == "weakly":

        excluded_patients = []
        if args.fold_system == "exclusion":
            val_dataset = MMsDataset(
                mode="validation", transform=val_aug, img_transform=[],
                folding_system="patient", normalization=args.normalization, label_type=args.label_type,
                train_fold=args.data_fold, val_fold=args.data_fold_validation, add_depth=args.add_depth
            )
            excluded_patients = val_dataset.df["External code"].unique()

        train_dataset = MMsWeaklyDataset(
            mode="train", transform=train_aug, img_transform=train_aug_img,
            folding_system=args.fold_system, normalization=args.normalization, label_type=args.label_type,
            train_fold=args.data_fold, val_fold=args.data_fold_validation, add_depth=args.add_depth,
            exclusion_patients=excluded_patients
        )

        val_dataset = MMsWeaklyDataset(
            mode="validation", transform=train_aug, img_transform=train_aug_img,
            folding_system=args.fold_system, normalization=args.normalization, label_type=args.label_type,
            train_fold=args.data_fold, val_fold=args.data_fold_validation, add_depth=args.add_depth,
            exclusion_patients=excluded_patients
        )

        print("[{}] Total {} Images: {}".format(args.dataset, "Train Dataset", len(train_dataset)))
        print("[{}] Total {} Images: {}".format(args.dataset, "Validation Dataset", len(val_dataset)))

    if args.dataset == "mnms_and_weakly":
        mnms_dataset = MMsDataset(
            mode="train", transform=train_aug, img_transform=train_aug_img,
            folding_system=args.fold_system, normalization=args.normalization, label_type=args.label_type,
            train_fold=args.data_fold, val_fold=args.data_fold_validation, add_depth=args.add_depth
        )

        val_dataset = MMsDataset(
            mode="validation", transform=val_aug, img_transform=[],
            folding_system=args.fold_system, normalization=args.normalization, label_type=args.label_type,
            train_fold=args.data_fold, val_fold=args.data_fold_validation, add_depth=args.add_depth
        )

        excluded_patients = []
        if args.fold_system != "all":
            excluded_patients = val_dataset.df["External code"].unique()

        weakly_dataset = MMsWeaklyDataset(
            mode="train", transform=train_aug, img_transform=train_aug_img,
            folding_system=args.fold_system, normalization=args.normalization, label_type=args.label_type,
            train_fold=args.data_fold, val_fold=args.data_fold_validation, add_depth=args.add_depth,
            exclusion_patients=excluded_patients
        )

        train_dataset = torch.utils.data.ConcatDataset([mnms_dataset, weakly_dataset])

        print("[{}] Total {} Images: {}".format(args.dataset, "Train Dataset", len(train_dataset)))
        print("[{}] Total {} Images: {}".format(args.dataset, "Validation Dataset", len(val_dataset)))

    if args.dataset == "entropy":

        excluded_patients = []
        if args.fold_system == "exclusion":
            val_dataset = MMsDataset(
                mode="validation", transform=val_aug, img_transform=[],
                folding_system="patient", normalization=args.normalization, label_type=args.label_type,
                train_fold=args.data_fold, val_fold=args.data_fold_validation, add_depth=args.add_depth
            )
            excluded_patients = val_dataset.df["External code"].unique()

        train_dataset = MMsEntropyDataset(
            mode="train", transform=train_aug, img_transform=train_aug_img,
            folding_system=args.fold_system, normalization=args.normalization, label_type=args.label_type,
            train_fold=args.data_fold, val_fold=args.data_fold_validation, add_depth=args.add_depth,
            exclusion_patients=excluded_patients
        )

        val_dataset = MMsEntropyDataset(
            mode="validation", transform=train_aug, img_transform=train_aug_img,
            folding_system=args.fold_system, normalization=args.normalization, label_type=args.label_type,
            train_fold=args.data_fold, val_fold=args.data_fold_validation, add_depth=args.add_depth,
            exclusion_patients=excluded_patients
        )

        print("[{}] Total {} Images: {}".format(args.dataset, "Train Dataset", len(train_dataset)))
        print("[{}] Total {} Images: {}".format(args.dataset, "Validation Dataset", len(val_dataset)))

    if args.dataset == "mnms_and_entropy":
        mnms_dataset = MMsDataset(
            mode="train", transform=train_aug, img_transform=train_aug_img,
            folding_system=args.fold_system, normalization=args.normalization, label_type=args.label_type,
            train_fold=args.data_fold, val_fold=args.data_fold_validation, add_depth=args.add_depth
        )

        val_dataset = MMsDataset(
            mode="validation", transform=val_aug, img_transform=[],
            folding_system=args.fold_system, normalization=args.normalization, label_type=args.label_type,
            train_fold=args.data_fold, val_fold=args.data_fold_validation, add_depth=args.add_depth
        )

        excluded_patients = []
        if args.fold_system != "all":
            excluded_patients = val_dataset.df["External code"].unique()

        entropy_dataset = MMsEntropyDataset(
            mode="train", transform=train_aug, img_transform=train_aug_img,
            folding_system=args.fold_system, normalization=args.normalization, label_type=args.label_type,
            train_fold=args.data_fold, val_fold=args.data_fold_validation, add_depth=args.add_depth,
            exclusion_patients=excluded_patients
        )

        train_dataset = BalancedConcatDataset(mnms_dataset, entropy_dataset)
        # train_dataset = torch.utils.data.ConcatDataset([mnms_dataset, entropy_dataset])

        print("[{}] Total {} Images: {}".format(args.dataset, "Train Dataset", len(train_dataset)))
        print("[{}] Total {} Images: {}".format(args.dataset, "Validation Dataset", len(val_dataset)))

    if args.dataset == "mnms_and_entropy_and_weakly":
        mnms_dataset = MMsDataset(
            mode="train", transform=train_aug, img_transform=train_aug_img,
            folding_system=args.fold_system, normalization=args.normalization, label_type=args.label_type,
            train_fold=args.data_fold, val_fold=args.data_fold_validation, add_depth=args.add_depth
        )

        val_dataset = MMsDataset(
            mode="validation", transform=val_aug, img_transform=[],
            folding_system=args.fold_system, normalization=args.normalization, label_type=args.label_type,
            train_fold=args.data_fold, val_fold=args.data_fold_validation, add_depth=args.add_depth
        )

        excluded_patients = []
        if args.fold_system != "all":
            excluded_patients = val_dataset.df["External code"].unique()

        entropy_dataset = MMsEntropyDataset(
            mode="train", transform=train_aug, img_transform=train_aug_img,
            folding_system=args.fold_system, normalization=args.normalization, label_type=args.label_type,
            train_fold=args.data_fold, val_fold=args.data_fold_validation, add_depth=args.add_depth,
            exclusion_patients=excluded_patients
        )

        weakly_dataset = MMsWeaklyDataset(
            mode="train", transform=train_aug, img_transform=train_aug_img,
            folding_system=args.fold_system, normalization=args.normalization, label_type=args.label_type,
            train_fold=args.data_fold, val_fold=args.data_fold_validation, add_depth=args.add_depth,
            exclusion_patients=excluded_patients
        )

        train_dataset = BalancedConcatDataset(mnms_dataset, entropy_dataset, weakly_dataset)
        # train_dataset = torch.utils.data.ConcatDataset([mnms_dataset, entropy_dataset, weakly_dataset])

        print("[{}] Total {} Images: {}".format(args.dataset, "Train Dataset", len(train_dataset)))
        print("[{}] Total {} Images: {}".format(args.dataset, "Validation Dataset", len(val_dataset)))

    if args.dataset == "mnms_test":
        dataset = MMsSubmissionDataset(
            args.input_data_directory, val_aug, [],
            normalization=args.normalization, add_depth=args.add_depth
        )

        print("[{}] Total {} Images: {}".format(args.dataset, "Test", len(dataset)))
        return dataset

    return train_dataset, val_dataset
