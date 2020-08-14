import numpy as np

SMOOTH = 1e-10


def jaccard_coef(y_true, y_pred):
    """
    Size of the intersection divided by the size of the union of two label sets
    :param y_true: Ground truth (correct) labels
    :param y_pred: Predicted labels
    :return: Jaccard similarity coefficient
    """
    intersection = np.sum(y_true * y_pred, axis=None)
    union = np.sum(y_true, axis=None) + np.sum(y_pred, axis=None) - intersection
    return float(intersection + SMOOTH) / float(union + SMOOTH)


def dice_coef(y_true, y_pred):
    """
    Computes the Dice coefficient, a measure of set similarity.
    :param y_true: Ground truth (correct) labels
    :param y_pred: Predicted labels
    :return: Dice similarity coefficient
    """
    intersection = np.sum(y_true * y_pred, axis=None)
    summation = np.sum(y_true, axis=None) + np.sum(y_pred, axis=None)
    return (2.0 * intersection + SMOOTH) / (summation + SMOOTH)
