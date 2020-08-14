from utils.training import *

warnings.filterwarnings('ignore')


def binarize_volume_prediction(volume_pred, original_shape=None):
    """
    Takes a prediction mask with shape [slices, classes, height, width]
    and binarizes it to [slices, height, width]
    :param volume_pred: (array) [slices, classes, height, width] volume mask predictions
    :param original_shape: (tuple) Original volume shape to reshape to
    :return: (array) [slices, height, width] volume binarized mask
    """
    s, c, h, w = volume_pred.shape
    if original_shape is not None: h, w, s = original_shape
    output_volume = np.empty([s, h, w])

    for indx, single_pred in enumerate(volume_pred):
        pred_mask = convert_multiclass_mask(single_pred.unsqueeze(0)).data.cpu().numpy()

        if original_shape is not None:
            # Resize prediction to original image shape
            pred_mask = reshape_masks(pred_mask.squeeze(0), (h, w))
        output_volume[indx, ...] = pred_mask

    return output_volume


def plot_save_pred_volume(img_volume, pred_mask_volume, save_dir, img_id):
    """
    Save overlays of images and predictions using volumes
    :param img_volume: (array) [height, width, slices] Original image
    :param pred_mask_volume: (array) [height, width, slices] Prediction mask
    :param save_dir: (string) Folder to save overlays
    :param img_id: (string) Image identifier
    :return:
    """

    os.makedirs(save_dir, exist_ok=True)

    img_volume = img_volume.transpose(2, 0, 1)  # [height, width, slices] -> [slices, height, width]
    pred_mask_volume = pred_mask_volume.transpose(2, 0, 1)  # [height, width, slices] -> [slices, height, width]

    for indx, (img) in enumerate(img_volume):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 16))
        ax1.axis('off')
        ax2.axis('off')

        ax1.imshow(img, cmap="gray")
        ax1.set_title("Original Image")

        mask = pred_mask_volume[indx, ...]
        masked = np.ma.masked_where(mask == 0, mask)
        ax2.imshow(img, cmap="gray")
        ax2.imshow(masked, 'jet', interpolation='bilinear', alpha=0.25)
        ax2.set_title("Original Overlay")

        pred_filename = os.path.join(
            save_dir,
            "mask_pred_{}_slice{}.png".format(img_id, indx),
        )
        plt.savefig(pred_filename, dpi=200, pad_inches=0.2, bbox_inches='tight')
        plt.close()
