#!/usr/bin/env python
# coding: utf-8

# ---- Library import ----
from torch.utils.data import DataLoader

# ---- My utils ----
from models import *
from utils.arguments import *
from utils.data_augmentation import data_augmentation_selector
from utils.dataload import dataset_selector
from utils.training import *

np.set_printoptions(precision=4)
_, _, val_aug = data_augmentation_selector(args.data_augmentation, args.img_size, args.crop_size)

val_dataset = dataset_selector("validation", val_aug, [], args)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)

model = model_selector(args.model_name, in_size=(args.crop_size, args.crop_size))
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
if args.model_checkpoint != "":
    print("Load from pretrained checkpoint: {}".format(args.model_checkpoint))
    model.load_state_dict(torch.load(args.model_checkpoint))
else:
    model_checkpoint = args.output_dir + "/model_" + args.model_name + "_best_iou.pt"
    if not os.path.exists(model_checkpoint):
        assert False, "Evaluating without model checkpoint?!"
    else:
        print("No checkpoint provided! Loading from best checkpoint: {}".format(model_checkpoint))
        model.load_state_dict(torch.load(model_checkpoint))

criterion, weights_criterion, multiclass_criterion = get_criterion(args.criterion, args.weights_criterion)

save_path = os.path.join(
    args.output_dir if args.evaluation_folder == "" else args.evaluation_folder,
    args.evaluation_descriptor
)
print("Generated evaluation files will be saved in: '{}'".format(save_path))

iou, dice, val_loss, stats = val_step(
    val_loader, model, criterion, weights_criterion, multiclass_criterion, args.binary_threshold,
    generate_stats=True, generate_overlays=args.eval_overlays, save_path=save_path
)

iou_str, dice_str = ['%.4f' % elem for elem in iou], ['%.4f' % elem for elem in dice]
print("Val IOU: {}, Val DICE: {}".format(iou_str, dice_str))
