#!/usr/bin/env python
# coding: utf-8

# ---- Library import ----
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA

# ---- My utils ----
from models import *
from utils.arguments import *
from utils.data_augmentation import data_augmentation_selector
from utils.dataload import dataset_selector
from utils.training import *

np.set_printoptions(precision=4)
train_aug, train_aug_img, val_aug = data_augmentation_selector(args.data_augmentation, args.img_size, args.crop_size)

train_dataset, val_dataset = dataset_selector(train_aug, train_aug_img, val_aug, args)
if args.dataset == "mnms_and_entropy" or args.dataset == "mnms_and_entropy_and_weakly":
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, collate_fn=train_dataset.masks_collate
    )
else:
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)

in_channels = 3 if args.add_depth else 1

model = model_selector(args.model_name, num_classes=4, in_channels=in_channels)
model_total_params = sum(p.numel() for p in model.parameters())
print("Model total number of parameters: {}".format(model_total_params))
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
if args.model_checkpoint != "":
    print("Load from pretrained checkpoint: {}".format(args.model_checkpoint))
    model.load_state_dict(torch.load(args.model_checkpoint))

criterion, weights_criterion, multiclass_criterion = get_criterion(args.criterion, args.weights_criterion)

optimizer = get_optimizer(args.optimizer, model, lr=args.learning_rate)
if args.apply_swa:
    print("--- Applying SWA ---")
    optimizer = SWA(optimizer, swa_start=args.swa_start, swa_freq=args.swa_freq, swa_lr=args.swa_lr)

scheduler = get_scheduler(
    args.scheduler, optimizer, epochs=args.epochs,
    min_lr=args.min_lr, max_lr=args.max_lr,
    scheduler_steps=args.scheduler_steps
)

progress = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": [], "val_hausdorff": [], "val_assd": []}
best_iou, best_dice, best_hausdorff, best_assd = -1, -1, 999, 999

print("\n-------------- START TRAINING -------------- ")
for current_epoch in range(args.epochs):

    train_loss = train_step(train_loader, model, criterion, weights_criterion, multiclass_criterion, optimizer)

    iou, dice, hausdorff, assd, val_loss, stats = val_step(
        val_loader, model, criterion, weights_criterion, multiclass_criterion, args.binary_threshold,
        generate_stats=((current_epoch + 1) == args.epochs), save_path=args.output_dir,
        generate_overlays=(((current_epoch + 1) == args.epochs) and args.eval_overlays),
    )

    iou_str, dice_str = ['%.4f' % elem for elem in iou], ['%.4f' % elem for elem in dice]
    hausdorff_str, assd_str = ['%.4f' % elem for elem in hausdorff], ['%.4f' % elem for elem in assd]
    # metrics is a list of [avg, Background, LV, MYO, RV] -> avg = mean(LV, MYO, RV)
    iou = iou[0]; dice = dice[0]; hausdorff = hausdorff[0]; assd = assd[0]

    print("[" + current_time() + "] Epoch: %d, LR: %.8f, Train: %.6f, Val: %.6f, "
                                 "Val IOU: %s, Val Dice: %s, Val Hausdorff: %s, Val ASSD: %s" % (
              current_epoch + 1, get_current_lr(optimizer), train_loss, val_loss,
              iou_str, dice_str, hausdorff_str, assd_str
          ))

    if iou > best_iou and not args.apply_swa:
        torch.save(model.state_dict(), args.output_dir + "/model_" + args.model_name + "_best_iou.pt")
        best_iou = iou

    if dice > best_dice and not args.apply_swa:
        torch.save(model.state_dict(), args.output_dir + "/model_" + args.model_name + "_best_dice.pt")
        best_dice = dice

    if hausdorff < best_hausdorff and not args.apply_swa:
        torch.save(model.state_dict(), args.output_dir + "/model_" + args.model_name + "_best_hausdorff.pt")
        best_hausdorff = hausdorff

    if assd < best_assd and not args.apply_swa:
        torch.save(model.state_dict(), args.output_dir + "/model_" + args.model_name + "_best_assd.pt")
        best_assd = assd

    if not args.apply_swa:
        torch.save(model.state_dict(), args.output_dir + "/model_" + args.model_name + "_last.pt")

    progress["train_loss"].append(np.mean(train_loss))
    progress["val_loss"].append(np.mean(val_loss))
    progress["val_iou"].append(iou)
    progress["val_dice"].append(dice)
    progress["val_hausdorff"].append(hausdorff)
    progress["val_assd"].append(assd)

    dict2df(progress, args.output_dir + 'progress.csv')

    scheduler_step(optimizer, scheduler, iou, args)

# --------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------------------- #

if args.apply_swa:
    torch.save(optimizer.state_dict(), args.output_dir + "/optimizer_" + args.model_name + "_before_swa_swap.pt")
    optimizer.swap_swa_sgd()  # Set the weights of your model to their SWA averages
    optimizer.bn_update(train_loader, model, device='cuda')

    torch.save(
        model.state_dict(),
        args.output_dir + "/swa_checkpoint_last_bn_update_{}epochs_lr{}.pt".format(args.epochs, args.swa_lr)
    )

    iou, dice, hausdorff, assd, val_loss, stats = val_step(
        val_loader, model, criterion, weights_criterion, multiclass_criterion, args.binary_threshold,
        generate_stats=True, generate_overlays=args.eval_overlays, save_path=os.path.join(args.output_dir, "swa_preds")
    )

    print("[SWA] Val IOU: %s, Val Dice: %s" % (iou, dice))

print("\n---------------")
val_iou = np.array(progress["val_iou"])
val_dice = np.array(progress["val_dice"])
print("Best IOU {:.4f} at epoch {}".format(val_iou.max(), val_iou.argmax() + 1))
print("Best DICE {:.4f} at epoch {}".format(val_dice.max(), val_dice.argmax() + 1))
print("---------------\n")
