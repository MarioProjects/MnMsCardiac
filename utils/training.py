import copy
import matplotlib.pyplot as plt
import os
import pandas as pd
import uuid
import warnings
import medpy.metric.binary as medmetrics
from time import gmtime, strftime

from utils.dataload import apply_batch_local_torch_normalization
from utils.losses import *
from utils.metrics import *
from utils.onecyclelr import OneCycleLR
from utils.radam import *

warnings.filterwarnings('ignore')


def current_time():
    """
    Gives current time
    :return: (String) Current time formated %Y-%m-%d %H:%M:%S
    """
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())


def dict2df(my_dict, path):
    """
    Save python dictionary as csv using pandas dataframe
    :param my_dict: Dictionary like {"epoch": [1, 2], "accuracy": [0.5, 0.9]}
    :param path: /path/to/file.csv
    :return: (void) Save csv on specified path
    """
    df = pd.DataFrame.from_dict(my_dict, orient="columns")
    df.index.names = ['epoch']
    df.to_csv(path, index=True)


def convert_multiclass_mask(mask):
    """
    Transform multiclass mask [batch, num_classes, h, w] to [batch, h, w]
    :param mask: Mask to transform
    :return: Transformed multiclass mask
    """
    return mask.max(1)[1]


def reshape_masks(ndarray, to_shape):
    """
    Reshapes a center cropped (or padded) array back to its original shape.
    :param ndarray: (np.array) Mask Array to reshape
    :param to_shape: (tuple) Final desired shape
    :return: (np.array) Reshaped array to desired shape
    """
    h_in, w_in = ndarray.shape
    h_out, w_out = to_shape

    if h_in > h_out:  # center crop along h dimension
        h_offset = math.ceil((h_in - h_out) / 2)
        ndarray = ndarray[h_offset:(h_offset + h_out), :]
    else:  # zero pad along h dimension
        pad_h = (h_out - h_in)
        rem = pad_h % 2
        pad_dim_h = (math.ceil(pad_h / 2), math.ceil(pad_h / 2 + rem))
        # npad is tuple of (n_before, n_after) for each (h,w,d) dimension
        npad = (pad_dim_h, (0, 0))
        ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)

    if w_in > w_out:  # center crop along w dimension
        w_offset = math.ceil((w_in - w_out) / 2)
        ndarray = ndarray[:, w_offset:(w_offset + w_out)]
    else:  # zero pad along w dimension
        pad_w = (w_out - w_in)
        rem = pad_w % 2
        pad_dim_w = (math.ceil(pad_w / 2), math.ceil(pad_w / 2 + rem))
        npad = ((0, 0), pad_dim_w)
        ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)

    return ndarray  # reshaped


def get_current_lr(optimizer):
    """
    Gives the current learning rate of optimizer
    :param optimizer: Optimizer instance
    :return: Learning rate of specified optimizer
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_optimizer(optmizer_type, model, lr=0.1):
    """
    Create an instance of optimizer
    :param optmizer_type: (string) Optimizer name
    :param model: Model that optimizer will use
    :param lr: Learning rate
    :return: Instance of specified optimizer
    """
    if optmizer_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif optmizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optmizer_type == "over9000":
        return Over9000(filter(lambda p: p.requires_grad, model.parameters()))

    assert False, "No optimizer named: {}".format(optmizer_type)


def get_scheduler(scheduler_name, optimizer, epochs=40, min_lr=0.002, max_lr=0.01, scheduler_steps=None):
    """
    Gives the specified learning rate scheduler
    :param scheduler_name: Scheduler name
    :param optimizer: Optimizer which is changed by the scheduler
    :param epochs: Total training epochs
    :param min_lr: Minimum learning rate for OneCycleLR Scheduler
    :param max_lr: Maximum learning rate for OneCycleLR Scheduler
    :param scheduler_steps: If scheduler steps is selected, which steps perform
    :return: Instance of learning rate scheduler
    """
    if scheduler_name == "steps":
        if scheduler_steps is None:
            assert False, "Please specify scheduler steps."
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_steps, gamma=0.1)
    elif scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=6, factor=0.1, patience=12)
    elif scheduler_name == "one_cycle_lr":
        return OneCycleLR(optimizer, num_steps=epochs, lr_range=(min_lr, max_lr))
    elif scheduler_name == "constant":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9999], gamma=0.1)
    else:
        assert False, "Unknown scheduler: {}".format(scheduler_name)


def scheduler_step(optimizer, scheduler, metric, args):
    """
    Perform a step of a scheduler
    :param optimizer: Optimizer used during training
    :param scheduler: Scheduler instance
    :param metric: Metric to minimize
    :param args: Training list of arguments with required fields (Bool: apply_swa, String: scheduler_name)
    :return: (void) Apply scheduler step
    """
    if args.apply_swa:
        optimizer.step()
    if args.scheduler == "steps":
        scheduler.step()
    elif args.scheduler == "plateau":
        scheduler.step(metric)
    elif args.scheduler == "one_cycle_lr":
        scheduler.step()
    elif args.scheduler == "constant":
        pass  # No modify learning rate


def get_criterion(criterion_type, weights_criterion='default'):
    """
    Gives a list of subcriterions and their corresponding weight
    :param criterion_type: Name of created criterion
    :param weights_criterion: (optional) Weight for each subcriterion
    :return:
        (list) Subcriterions
        (list) Weights for each criterion
    """
    if criterion_type == "bce":
        criterion1 = nn.BCEWithLogitsLoss().cuda()
        criterion = [criterion1]
        multiclass = [False]
        default_weights_criterion = [1]
    elif criterion_type == "ce":
        criterion1 = nn.CrossEntropyLoss().cuda()
        criterion = [criterion1]
        multiclass = [True]
        default_weights_criterion = [1]
    elif criterion_type == "bce_dice":
        criterion1 = nn.BCEWithLogitsLoss().cuda()
        criterion2 = SoftDiceLoss().cuda()
        criterion3 = SoftInvDiceLoss().cuda()
        criterion = [criterion1, criterion2, criterion3]
        multiclass = [False, False, False]
        default_weights_criterion = [0.55, 0.35, 0.1]
    elif criterion_type == "bce_dice_border":
        criterion1 = nn.BCEWithLogitsLoss().cuda()
        criterion2 = SoftDiceLoss().cuda()
        criterion3 = SoftInvDiceLoss().cuda()
        criterion4 = BCEDicePenalizeBorderLoss().cuda()
        criterion = [criterion1, criterion2, criterion3, criterion4]
        multiclass = [False, False, False, False]
        default_weights_criterion = [0.5, 0.2, 0.1, 0.2]
    elif criterion_type == "bce_dice_ac":
        criterion1 = nn.BCEWithLogitsLoss().cuda()
        criterion2 = SoftDiceLoss().cuda()
        criterion3 = SoftInvDiceLoss().cuda()
        criterion4 = ActiveContourLoss().cuda()
        criterion = [criterion1, criterion2, criterion3, criterion4]
        multiclass = [False, False, False, False]
        default_weights_criterion = [0.3, 0.4, 0.2, 0.3]
    elif criterion_type == "bce_dice_border_ce":
        criterion1 = nn.BCEWithLogitsLoss().cuda()
        criterion2 = SoftDiceLoss().cuda()
        criterion3 = SoftInvDiceLoss().cuda()
        criterion4 = BCEDicePenalizeBorderLoss().cuda()
        criterion5 = nn.CrossEntropyLoss().cuda()
        criterion = [criterion1, criterion2, criterion3, criterion4, criterion5]
        multiclass = [False, False, False, False, True]
        default_weights_criterion = [0.5, 0.2, 0.2, 0.2, 0.5]
    elif criterion_type == "bce_dice_border_haus_ce":
        criterion1 = nn.BCEWithLogitsLoss().cuda()
        criterion2 = SoftDiceLoss().cuda()
        criterion3 = SoftInvDiceLoss().cuda()
        criterion4 = BCEDicePenalizeBorderLoss().cuda()
        criterion5 = HDDTBinaryLoss().cuda()
        criterion6 = nn.CrossEntropyLoss().cuda()
        criterion = [criterion1, criterion2, criterion3, criterion4, criterion5, criterion6]
        multiclass = [False, False, False, False, False, True]
        default_weights_criterion = [0.5, 0.2, 0.2, 0.2, 0.2, 0.5]
    elif criterion_type == "bce_dice_ce":
        criterion1 = nn.BCEWithLogitsLoss().cuda()
        criterion2 = SoftDiceLoss().cuda()
        criterion3 = SoftInvDiceLoss().cuda()
        criterion4 = nn.CrossEntropyLoss().cuda()
        criterion = [criterion1, criterion2, criterion3, criterion4]
        multiclass = [False, False, False, True]
        default_weights_criterion = [0.3, 0.4, 0.2, 0.65]
    else:
        assert False, "Unknown criterion: {}".format(criterion_type)

    if weights_criterion == "default":
        return criterion, default_weights_criterion, multiclass
    else:
        weights_criterion = [float(i) for i in weights_criterion.split(',')]
        if len(weights_criterion) != len(default_weights_criterion):
            assert False, "We need a weight for each subcriterion"
        return criterion, weights_criterion, multiclass


def defrost_model(model):
    """
    Unfreeze model parameters
    :param model: Instance of model
    :return: (void)
    """
    for param in model.parameters():  # Defrost model
        param.requires_grad = True


def check_defrost(model, defrosted, current_epoch, args):
    """
    Defrost model if given conditions
    :param model: (Pytorch model) Model to defrost
    :param defrosted: (bool) Current model status. Defrosted (True) or Not (False)
    :param current_epoch: (int) Current training epoch
    :param args: Training list of arguments with required fields (int: defrost_epoch, String: model_name)
    :return: (bool) True if model is defrosted or contrary False
    """
    if not defrosted and current_epoch >= args.defrost_epoch and "scratch" not in args.model_name:
        print("\n---- Unfreeze Model Weights! ----")
        defrost_model(model)
        defrosted = True
    return defrosted


def calculate_loss_accuracy(y_true, y_pred, criterion, weights_criterion):
    # ToDo: Fix -> https://github.com/MarioProjects/transfer_learning_experiments/blob/master/utils/utils_training.py#L223
    """
    Calculate the loss of generated predictions
    :param y_true: Expected prediction values
    :param y_pred: Model logits
    :param criterion: (list) Criterions to apply
    :param weights_criterion: (list) Weights for each subcriterion
    :return: Loss given by the criterions
    """

    loss = 0
    for indx, crit in enumerate(criterion):
        loss += (weights_criterion[indx] * crit(y_pred, y_true))

    return loss


def calculate_loss(y_true, y_pred, criterion, weights_criterion, multiclass_criterion):
    num_classes = len(np.unique(y_true.data.cpu().numpy()))
    loss = 0

    # if num_classes == 1:  # Single class case
    #     for indx, crit in enumerate(criterion):
    #         loss += (weights_criterion[indx] * crit(y_pred, y_true))

    # else:  # Multiclass case

    # Case Multiclass criterions
    multiclass_indices = [i for i, x in enumerate(multiclass_criterion) if x]
    if len(multiclass_indices) > 0:
        for indx_multiclass in multiclass_indices:
            loss += weights_criterion[indx_multiclass] * criterion[indx_multiclass](y_pred, y_true)

    # Case Multiclass as SingleClass problem => calculate criterions per class
    for current_class in y_true.unique():
        tmp_loss = 0
        tmp_mask = 1 - (y_true != current_class) * 1.0
        for indx, crit in enumerate(criterion):  # Acumulamos todos los losses para una clase
            if not multiclass_criterion[indx]:
                tmp_loss += (weights_criterion[indx] * crit(y_pred[:, current_class, :, :], tmp_mask))
        # Promediamos entre el numero de clases que participan
        loss += (tmp_loss / len(y_true.unique()))

    return loss


def train_step(train_loader, model, criterion, weights_criterion,
               multiclass_criterion, optimizer, task="segmentation"):
    """
    Perform a train step
    :param train_loader: (Dataset) Train dataset loader
    :param model: Model to train
    :param criterion: Choosed criterion
    :param weights_criterion: Choosed criterion weights
    :param multiclass_criterion:
    :param optimizer: Choosed optimizer
    :param task: One of "segmentation" - "classification"
    :return: Mean train loss
    """
    train_loss = []
    model.train()
    for indx, (image, label) in enumerate(train_loader):
        label = label.cuda()
        if task == "binary_classification":
            label = label.unsqueeze(1).float()
        elif task == "classification":
            label = label.long()
        image = image.type(torch.float).cuda()
        y_pred = model(image)

        if "classification" in task:
            loss = calculate_loss_accuracy(label, y_pred, criterion, weights_criterion)
        else:
            loss = calculate_loss(label, y_pred, criterion, weights_criterion, multiclass_criterion)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        train_loss.append(loss.item())

    return np.mean(train_loss)


def plot_save_pred(original_img, original_mask, pred_mask, save_dir, img_id):
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 16))
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    ax1.imshow(original_img, cmap="gray")
    ax1.set_title("Original Image")

    masked = np.ma.masked_where(original_mask == 0, original_mask)
    ax2.imshow(original_img, cmap="gray")
    ax2.imshow(masked, 'jet', interpolation='bilinear', alpha=0.25)
    ax2.set_title("Original Overlay")

    masked = np.ma.masked_where(pred_mask == 0, pred_mask)
    ax3.imshow(original_img, cmap="gray")
    ax3.imshow(masked, 'jet', interpolation='bilinear', alpha=0.25)
    ax3.set_title("Prediction Overlay")

    os.makedirs(os.path.join(save_dir, "preds_overlays"), exist_ok=True)

    pred_filename = os.path.join(
        save_dir,
        "preds_overlays",
        "mask_pred_{}.png".format(img_id),
    )
    plt.savefig(pred_filename, dpi=200, pad_inches=0.2, bbox_inches='tight')
    plt.close()


def val_step(val_loader, model, criterion, weights_criterion, multiclass_criterion,
             binary_threshold, num_classes=4, generate_stats=False, generate_overlays=False,
             save_path="", swap_values=None):
    """
    Perform a validation step
    :param val_loader: (Dataloader) Validation dataset loader
    :param model: Model used in training
    :param criterion: Choosed criterion
    :param weights_criterion: (list -> float) Choosed criterion weights
    :param multiclass_criterion:
    :param binary_threshold: (float) Threshold to set class as class 1. Typically 0.5
    :param num_classes: Num total classes in predictions
    :param generate_stats: (bool) If true generate predictions stats via pandas
    :param generate_overlays: (bool) If true save mask predictions
    :param save_path: (string) If save_preds then which directory to save mask predictions
    :param swap_values: (list of lists) In predicted mask, swaps first item and second in list. Example: [[1,2]]
    :return: Intersection Over Union and Dice Metrics, Mean validation loss
    """
    # https://stackoverflow.com/questions/8713620/appending-items-to-a-list-of-lists-in-python
    ious, dices = [[] for _ in range(num_classes)], [[] for _ in range(num_classes)]
    hausdorffs, assds = [[] for _ in range(num_classes)], [[] for _ in range(num_classes)]
    val_loss, df_info = [], []
    model.eval()

    if save_path != "":
        os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for sample_indx, (image, original_img, original_mask, mask, img_id) in enumerate(val_loader):

            original_mask = original_mask.cuda()
            image = image.type(torch.float).cuda()
            prob_pred = model(image)

            # ToDo: Fix val loss
            # val_loss.append(calculate_loss(mask, prob_pred, criterion, weights_criterion, multiclass_criterion).item())
            val_loss.append(0.0)

            for indx, single_pred in enumerate(prob_pred):

                original_mask = original_mask[indx].data.cpu().numpy().astype(np.uint8).squeeze()

                # Calculate metrics resizing prediction to original mask shape
                pred_mask = convert_multiclass_mask(single_pred.unsqueeze(0)).data.cpu().numpy()
                pred_mask = reshape_masks(pred_mask.squeeze(0), original_mask.shape)
                if swap_values is not None:
                    for swap_val in swap_values:
                        pred_mask = np.where(
                            pred_mask == swap_val[0], swap_val[1],
                            np.where(pred_mask == swap_val[1], swap_val[0], pred_mask)
                        )

                patient_iou = [-1, -1, -1]  # LV, MYO, RV
                patient_dice = [-1, -1, -1]  # LV, MYO, RV
                patient_hausddorf = [999, 999, 999]  # LV, MYO, RV
                patient_assd = [999, 999, 999]  # LV, MYO, RV

                for current_class in np.unique(np.concatenate((original_mask, pred_mask))):

                    binary_ground_truth = np.where(original_mask == current_class, 1, 0).astype(np.int32)
                    binary_pred_mask = np.where(pred_mask == current_class, 1, 0).astype(np.int32)

                    tmp_iou = jaccard_coef(binary_ground_truth, binary_pred_mask)
                    tmp_dice = dice_coef(binary_ground_truth, binary_pred_mask)
                    # https://github.com/loli/medpy/blob/master/medpy/metric/binary.py#L1212
                    if not np.count_nonzero(binary_pred_mask) and not np.count_nonzero(binary_ground_truth):
                        # The same, distances 0
                        tmp_assd = 0
                        tmp_hausdorff = 0
                    elif not np.count_nonzero(binary_pred_mask) or not np.count_nonzero(binary_ground_truth):
                        # ToDo: equivalent worst value for Hausdorff and surface distances
                        tmp_assd = 999
                        tmp_hausdorff = 999
                    else:
                        tmp_assd = medmetrics.assd(binary_pred_mask, binary_ground_truth)
                        tmp_hausdorff = medmetrics.hd(binary_pred_mask, binary_ground_truth)

                    ious[current_class].append(tmp_iou)
                    dices[current_class].append(tmp_dice)
                    hausdorffs[current_class].append(tmp_hausdorff)
                    assds[current_class].append(tmp_assd)

                    # -1 Due index 0 is background
                    if current_class != 0:
                        patient_iou[current_class - 1] = tmp_iou
                        patient_dice[current_class - 1] = tmp_dice
                        patient_hausddorf[current_class - 1] = tmp_hausdorff
                        patient_assd[current_class - 1] = tmp_assd

                if generate_stats:
                    patient_info = img_id[0].split("_")
                    df_info.append({
                        "patient": patient_info[0], "slice": patient_info[1][5:], "phase": patient_info[2][5:],
                        "IOU_LV": patient_iou[0], "IOU_MYO": patient_iou[1], "IOU_RV": patient_iou[2],
                        "IOU_MEAN": np.mean(np.array([value for value in patient_iou if value != -1])),
                        "DICE_LV": patient_dice[0], "DICE_MYO": patient_dice[1], "DICE_RV": patient_dice[2],
                        "DICE_MEAN": np.mean(np.array([value for value in patient_dice if value != -1])),
                        "HAUSDORFF_LV": patient_hausddorf[0], "HAUSDORFF_MYO": patient_hausddorf[1],
                        "HAUSDORFF_RV": patient_hausddorf[2],
                        "HAUSDORFF_MEAN": np.mean(np.array([value for value in patient_hausddorf if value != -999])),
                        "ASSD_LV": patient_assd[0], "ASSD_MYO": patient_assd[1], "ASSD_RV": patient_assd[2],
                        "ASSD_MEAN": np.mean(np.array([value for value in patient_assd if value != -999])),
                    })

                if generate_overlays:
                    plot_save_pred(
                        original_img.data.cpu().numpy().squeeze(), original_mask,
                        pred_mask, save_path, img_id[0]
                    )

    stats = None
    if generate_stats:
        stats = pd.DataFrame(df_info)
        stats.to_csv(os.path.join(save_path, "val_patient_stats.csv"), index=False)

    iou = [np.mean(i) for i in ious]  # Class mean, not global
    iou = np.append(np.mean(iou[1:]), iou)  # Add as first value global class mean (without background)

    dice = [np.mean(i) for i in dices]
    dice = np.append(np.mean(dice[1:]), dice)

    hausdorff = [np.mean(i) for i in hausdorffs]
    hausdorff = np.append(np.mean(hausdorff[1:]), hausdorff)

    assd = [np.mean(i) for i in assds]
    assd = np.append(np.mean(assd[1:]), assd)

    return iou, dice, hausdorff, assd, np.mean(val_loss), stats


def val_step_accuracy(val_loader, model, criterion, weights_criterion, task="classification"):
    """
    Perform a validation step
    :param val_loader: (Dataloader) Validation dataset loader
    :param model: Model used in training
    :param criterion: Choosed criterion
    :param weights_criterion: (list -> float) Choosed criterion weights
    :param task:
    :return: Accuracy Metric, Mean validation loss
    """
    total, correct, val_loss = 0, 0, 0
    val_loss, df_info = [], []
    model.eval()

    with torch.no_grad():
        for sample_indx, (image, targets, _, img_id) in enumerate(val_loader):
            if task == "binary_classification":
                targets = targets.unsqueeze(1).float().cuda()
            elif task == "classification":
                targets = targets.long().cuda()
            image = image.type(torch.float).cuda()
            prob_pred = model(image)

            val_loss.append(
                calculate_loss_accuracy(targets, prob_pred, criterion, weights_criterion).item()
            )

            total += targets.size(0)
            if task == "binary_classification":
                # Assuming binary classification
                predicted = torch.round(nn.Sigmoid()(prob_pred))
                correct += predicted.eq(targets).sum().item()
            elif task == "classification":
                _, predicted = prob_pred.max(1)
                correct += predicted.eq(targets).sum().item()

    acc = correct / total

    return acc, np.mean(np.array(val_loss))


def clean_stats(df, train_df):
    df = df.fillna(1)
    df["Vendor"] = "Z"
    df["Centre"] = 999
    df["Type"] = "XX"

    for i, row in df.iterrows():
        patient = row["patient"]
        c_phase = row["phase"]

        centre = train_df.loc[train_df["External code"] == patient].iloc[0]["Centre"]
        vendor = train_df.loc[train_df["External code"] == patient].iloc[0]["Vendor"]
        c_type = train_df.loc[(train_df["External code"] == patient) & (train_df["Phase"] == int(c_phase))].iloc[0][
            "Type"]

        df.at[i, 'Vendor'] = vendor
        df.at[i, 'Centre'] = centre
        df.at[i, 'Type'] = c_type

    return df


def val_step_experiments(segmentation_loader, segmentator, val_same_patients, train_csv, verbose=True,
                         num_classes=4, generate_imgs=False, image_modificator_fn=None, save_dir="val_outs", ):
    if image_modificator_fn is None and generate_imgs:
        assert False, "Why generate images if not modifying them (not image_modificator_fn)?!"

    ious, dices = [[] for _ in range(num_classes)], [[] for _ in range(num_classes)]
    df_info = []
    segmentator.eval()

    for sample_indx, (image, original_image, original_mask, mask, img_id) in enumerate(segmentation_loader):

        c_patient = img_id[0].split("_")[0]
        c_vendor = train_csv.loc[train_csv["External code"] == c_patient].iloc[0]["Vendor"]

        if c_patient in val_same_patients and c_vendor == "B":

            img_id_str = img_id[0]
            original_mask = original_mask.cuda()

            image = image.type(torch.float).cuda()
            original_img = copy.deepcopy(image).detach()
            if image_modificator_fn is not None:
                image, init_discriminator_probs, new_discriminator_probs = image_modificator_fn.apply(image)
                original_img = apply_batch_local_torch_normalization(original_img, "standardize")

            prob_pred = segmentator(image)

            for indx, single_pred in enumerate(prob_pred):

                original_mask = original_mask[indx].data.cpu().numpy().astype(np.uint8).squeeze()

                # Calculate metrics resizing prediction to original mask shape
                pred_mask = convert_multiclass_mask(single_pred.unsqueeze(0)).data.cpu().numpy()
                pred_mask = reshape_masks(pred_mask.squeeze(0), original_mask.shape)

                patient_iou = [-1, -1, -1]  # LV, MYO, RV
                patient_dice = [-1, -1, -1]  # LV, MYO, RV
                for current_class in np.unique(np.concatenate((original_mask, pred_mask))):
                    binary_ground_truth = np.where(original_mask == current_class, 1, 0).astype(np.int32)
                    binary_pred_mask = np.where(pred_mask == current_class, 1, 0).astype(np.int32)
                    tmp_iou = jaccard_coef(binary_ground_truth, binary_pred_mask)
                    ious[current_class].append(tmp_iou)
                    tmp_dice = dice_coef(binary_ground_truth, binary_pred_mask)
                    dices[current_class].append(tmp_dice)

                    # -1 Due index 0 is background
                    if current_class != 0:
                        patient_iou[current_class - 1] = tmp_iou
                        patient_dice[current_class - 1] = tmp_dice

                patient_info = img_id[0].split("_")
                df_info.append({
                    "patient": patient_info[0], "slice": patient_info[1][5:], "phase": patient_info[2][5:],
                    "IOU_LV": patient_iou[0], "IOU_MYO": patient_iou[1], "IOU_RV": patient_iou[2],
                    "IOU_MEAN": np.mean(np.array([value for value in patient_iou if value != -1])),
                    "DICE_LV": patient_dice[0], "DICE_MYO": patient_dice[1], "DICE_RV": patient_dice[2],
                    "DICE_MEAN": np.mean(np.array([value for value in patient_dice if value != -1])),
                })

                diferencia = (original_img - image)
                diferencia_np = diferencia.cpu().numpy()

                if verbose:
                    print(img_id_str)
                    print(f"IOU: {np.mean(np.array([value for value in patient_iou if value != -1]))}")
                    print("Original Max: {}".format(np.absolute(original_img.cpu().numpy()).max()))
                    print("Original Mean: {}".format(original_img.cpu().numpy().mean()))
                    print("Original STD: {}".format(original_img.cpu().numpy().std()))

                    print("Modified Max: {}".format(np.absolute(image.cpu().numpy()).max()))
                    print("Modified Mean: {}".format(image.cpu().numpy().mean()))
                    print("Modified STD: {}".format(image.cpu().numpy().std()))

                    print("Difference Max: {}".format(np.absolute(diferencia_np).max()))
                    print("Difference Mean: {}".format(diferencia_np.mean()))
                    print("Difference STD: {}".format(diferencia_np.std()))

                if generate_imgs:
                    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(25, 40))

                    ax1.set_title("Original {} - {}".format(c_vendor, ['%.3f' % elem for elem in
                                                                       init_discriminator_probs.tolist()[0]]))
                    ax1.imshow(original_img.cpu().detach().squeeze(0).squeeze(0), cmap="gray")
                    ax1.axis("off")

                    ax2.set_title(
                        "Transformed - {}".format(['%.3f' % elem for elem in new_discriminator_probs.tolist()[0]]))
                    ax2.imshow(image.cpu().detach().squeeze(0).squeeze(0), cmap="gray")
                    ax2.axis("off")

                    ax3.set_title("Difference")
                    ax3.imshow(diferencia.cpu().detach().squeeze(0).squeeze(0), cmap="gray")
                    ax3.axis("off")

                    ax4.set_title("Original Mask")
                    ax4.imshow(original_mask, cmap="gray")
                    ax4.axis("off")

                    ax5.set_title("Predicted Mask - Mean IOU: {:.4f}".format(
                        np.mean(np.array([value for value in patient_iou if value != -1]))))
                    ax5.imshow(pred_mask, cmap="gray")
                    ax5.axis("off")

                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(
                        os.path.join(save_dir, "{}_{}.jpg".format(img_id_str, str(uuid.uuid4().hex))),
                        dpi=200, pad_inches=0.2, bbox_inches='tight'
                    )
                    plt.close()

    stats = pd.DataFrame(df_info)

    stats = clean_stats(stats, train_csv)
    same_stats = stats[stats['patient'].isin(val_same_patients)]

    return same_stats
