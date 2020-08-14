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

train_dataset = dataset_selector("train", train_aug, train_aug_img, args)
val_dataset = dataset_selector("validation", val_aug, [], args)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

in_channels = 3 if args.add_depth else 1

model = model_selector(args.model_name, num_classes=args.num_classes, in_channels=in_channels)
model_total_params = sum(p.numel() for p in model.parameters())
print("Model total number of parameters: {}".format(model_total_params))
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
if args.model_checkpoint != "":
    print("Load from pretrained checkpoint: {}".format(args.model_checkpoint))
    model.load_state_dict(torch.load(args.model_checkpoint))

criterion, weights_criterion, multiclass_criterion = get_criterion(args.criterion, args.weights_criterion)

optimizer = get_optimizer(args.optimizer, model, lr=args.learning_rate)
if args.apply_swa:
    optimizer = SWA(optimizer, swa_start=args.swa_start, swa_freq=args.swa_freq, swa_lr=args.swa_lr)

scheduler = get_scheduler(
    args.scheduler, optimizer, epochs=args.epochs,
    min_lr=args.min_lr, max_lr=args.max_lr,
    scheduler_steps=args.scheduler_steps
)

progress = {"train_loss": [], "val_loss": [], "val_accuracy": []}
best_accuracy, best_model = -1, None

task = "binary_classification" if "binary" in args.label_type else "classification"

print("\n-------------- START TRAINING -------------- ")
for current_epoch in range(args.epochs):

    train_loss = train_step(
        train_loader, model, criterion, weights_criterion, multiclass_criterion, optimizer, task=task
    )

    accuracy, val_loss = val_step_accuracy(
        val_loader, model, criterion, weights_criterion, multiclass_criterion, task=task
    )

    print("[" + current_time() + "] Epoch: %d, LR: %.8f, Train: %.6f, Val: %.6f, Val Accuracy: %.4f" % (
        current_epoch + 1, get_current_lr(optimizer), train_loss, val_loss, accuracy))

    if accuracy > best_accuracy and not args.apply_swa:
        torch.save(model.state_dict(), args.output_dir + "/model_" + args.model_name + "_best_accuracy.pt")
        best_accuracy = accuracy
        best_model = model.state_dict()

    if not args.apply_swa:
        torch.save(model.state_dict(), args.output_dir + "/model_" + args.model_name + "_last.pt")

    progress["train_loss"].append(np.mean(train_loss))
    progress["val_loss"].append(np.mean(val_loss))
    progress["val_accuracy"].append(accuracy)

    dict2df(progress, args.output_dir + 'progress.csv')

    scheduler_step(optimizer, scheduler, accuracy, args)

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

    accuracy, val_loss = val_step_accuracy(
        val_loader, model, criterion, weights_criterion, multiclass_criterion, task=task
    )

    print("[SWA] Val Accuracy: %s" % (accuracy))

print("\n---------------")
val_accuracy = np.array(progress["val_accuracy"])
print("Best Accuracy {:.4f} at epoch {}".format(val_accuracy.max(), val_accuracy.argmax() + 1))
print("---------------\n")
