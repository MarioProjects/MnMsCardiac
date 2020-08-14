#!/usr/bin/env python
# coding: utf-8

# ### Import libraries
# ---- My utils ----
import argparse

from utils.data_augmentation import data_augmentation_selector
from utils.dataload import *
from utils.training import *
from models import *

from pylab import rcParams

rcParams['figure.figsize'] = 8, 5
plt.rc('grid', linestyle="--", color='gray')

# https://learnui.design/tools/data-color-picker.html#palette
colors = ['#33508f', '#ff5d68', '#ffa600', '#af4f9b']


def get_args():
    parser = argparse.ArgumentParser(description='M&Ms 2020 Challenge - Input entropy modification')
    parser.add_argument('--target', type=str, default='A', help='Desired domain to transform')
    parser.add_argument('--out_threshold', type=float, default=0.01, help='Difference stop condition')
    parser.add_argument('--max_iters', type=int, default=100, help='Maximum number of iters to apply entropy')

    parser.add_argument('--entropy_lambda', type=float, default=0.99, help='Learning rate')

    parser.add_argument('--add_l1', action='store_true', help='If add L1 loss or not')
    parser.add_argument('--l1_lambda', type=float, default=0.0, help='L1 impact factor')

    parser.add_argument('--add_blur_param', action='store_true', help='Add blur matrix param or not')
    parser.add_argument('--blur_lambda', type=float, default=0.0, help='Blur param impact factor')

    parser.add_argument('--add_unblur_param', action='store_true', help='Add unblur matrix param or not')
    parser.add_argument('--unblur_lambda', type=float, default=0.0, help='Unblur param impact factor')

    parser.add_argument('--add_gamma_param', action='store_true', help='Add gamma param or not')
    parser.add_argument('--gamma_lambda', type=float, default=0.0, help='Gamma param impact factor')

    parser.add_argument('--generate_images', action='store_true', help='Generate images')
    parser.add_argument('--verbose', action='store_true', help='Add verbosity')

    arguments = parser.parse_args()
    return arguments


print("\n\n ----------------------------------------------")
args = get_args()
for argument in args.__dict__:
    print("{}: {}".format(argument, args.__dict__[argument]))

# --------------------------------
# ### Load Data

train_aug, train_aug_img, val_aug = data_augmentation_selector("none", 224, 224)

data_partition = "validation"
general_aug, img_aug = train_aug, train_aug_img
normalization = "standardize"
fold_system = "patient"
label_type = "vendor_label_full"
data_fold = 0
add_depth = False
in_channels = 3 if add_depth else 1
data_fold_validation = None

discriminator_val_dataset = MMsDataset(
    mode=data_partition, transform=train_aug, img_transform=train_aug_img,
    folding_system=fold_system, normalization=normalization, label_type=label_type,
    train_fold=data_fold, val_fold=data_fold_validation, add_depth=add_depth
)

discriminator_loader = DataLoader(discriminator_val_dataset, batch_size=1, shuffle=False, drop_last=False)

data_partition = "validation"
general_aug, img_aug = train_aug, train_aug_img
normalization = "none"  # "standardize" no normalization, we will apply it later on apply() at ImageBackwardEntropy
fold_system = "vendor"
label_type = "mask"

segmentation_train_fold = 'A'
segmentation_val_fold = 'B'

segmentation_val_dataset = MMsDataset(
    mode=data_partition, transform=general_aug, img_transform=img_aug,
    folding_system=fold_system, normalization=normalization, label_type=label_type,
    train_fold=segmentation_train_fold, val_fold=segmentation_val_fold,
)

segmentation_loader = DataLoader(segmentation_val_dataset, batch_size=1, shuffle=False, drop_last=False)

val_same_patients = np.intersect1d(
    discriminator_val_dataset.df["External code"],
    segmentation_val_dataset.df["External code"]
)

# ------------------------------------------
# ### Load Models

num_classes, crop_size, model_name = 3, 224, "resnet34_unet_scratch_classification"

discriminator = model_selector(model_name, num_classes=num_classes, in_channels=in_channels)
model_total_params = sum(p.numel() for p in discriminator.parameters())
print("Model total number of parameters: {}".format(model_total_params))
discriminator = torch.nn.DataParallel(discriminator, device_ids=range(torch.cuda.device_count()))

model_checkpoint = "checkpoints/full_discriminator_{}channel_fold{}.pt".format(in_channels, data_fold)
discriminator.load_state_dict(torch.load(model_checkpoint))
print("Discriminator checkpoint loaded correctly!")

###########################################################################################

num_classes, crop_size, model_name = 4, 224, "resnet34_unet_scratch"

segmentator = model_selector(model_name, num_classes=num_classes, in_channels=in_channels)
model_total_params = sum(p.numel() for p in segmentator.parameters())
print("Model total number of parameters: {}".format(model_total_params))
segmentator = torch.nn.DataParallel(segmentator, device_ids=range(torch.cuda.device_count()))

model_checkpoint = f"checkpoints/segmentator_{segmentation_train_fold}vs{segmentation_val_fold}_{in_channels}channel.pt"
segmentator.load_state_dict(torch.load(model_checkpoint))
print("Segmentator checkpoint loaded correctly!")

###########################################################################################

criterion, weights_criterion = "ce", "default"
criterion, weights_criterion, multiclass_criterion = get_criterion(criterion, weights_criterion)
task = "classification"  # binary_classification or classification

accuracy, val_loss = val_step_accuracy(
   discriminator_loader, discriminator, criterion, weights_criterion, multiclass_criterion, task=task
)

print(f"Discriminator accuracy: {accuracy}")


###########################################################################################


# Image modification using entropy
class ImageBackwardEntropy:

    def __init__(self, discriminator_model, target, max_iters=500, out_threshold=0.01, entropy_lambda=0.9,
                 add_l1=False, l1_lambda=0.0, add_blur_param=False, blur_lambda=0.0,
                 add_unblur_param=False, unblur_lambda=0.0, add_gamma_param=False, gamma_lambda=0.0, verbose=False):

        self.discriminator_model = discriminator_model
        self.target = target
        self.max_iters = max_iters
        self.out_threshold = out_threshold
        self.entropy_lambda = entropy_lambda
        self.verbose = verbose

        self.add_l1 = add_l1
        self.l1_lambda = l1_lambda

        self.add_blur_param = add_blur_param
        self.blur_lambda = blur_lambda

        self.add_unblur_param = add_unblur_param
        self.unblur_lambda = unblur_lambda

        self.add_gamma_param = add_gamma_param
        self.gamma_lambda = gamma_lambda

    @staticmethod
    def cxe_loss(predicted, real):
        # https://discuss.pytorch.org/t/catrogircal-cross-entropy-with-soft-classes/50871
        return -(real * torch.log(predicted)).sum(dim=1).mean()

    def apply(self, image):

        x = copy.deepcopy(image).detach()

        blur_param = torch.tensor(0.0, requires_grad=False)

        blurring_matrix = torch.tensor(
            [[blur_param, blur_param, blur_param],
             [blur_param, 1, blur_param],
             [blur_param, blur_param, blur_param]],
            requires_grad=False)

        unblur_param = torch.tensor(0.0, requires_grad=False)

        unblurring_matrix = torch.tensor(
            [[-unblur_param, -unblur_param, -unblur_param],
             [-unblur_param, 1 + 8 * unblur_param, -unblur_param],
             [-unblur_param, -unblur_param, -unblur_param]],
            requires_grad=False)

        gamma_param = torch.tensor(1.0, requires_grad=False).cuda()

        with torch.no_grad():
            x_ini = apply_batch_local_torch_normalization(copy.deepcopy(x).detach(), "standardize")
            initial_y = torch.nn.functional.softmax(self.discriminator_model(x_ini), dim=1)

        for iteration in range(self.max_iters):

            with torch.autograd.detect_anomaly():
                x.requires_grad_(True)

                # Apply blurring matrix conv2d
                if self.add_blur_param:
                    blurring_matrix = blurring_matrix.detach()
                    blur_sum = blurring_matrix.sum()
                    blurring_matrix = blurring_matrix / blur_sum
                    blurring_matrix.requires_grad_(True)
                    x2 = torch.nn.functional.conv2d(
                        x, blurring_matrix.unsqueeze(0).unsqueeze(0).cuda(),
                        padding=1, stride=1
                    )
                else:
                    x2 = x

                # Apply unblurring matrix conv2d
                if self.add_unblur_param:
                    unblurring_matrix = unblurring_matrix.detach()
                    unblur_sum = unblurring_matrix.sum()
                    unblurring_matrix = unblurring_matrix / unblur_sum
                    unblurring_matrix.requires_grad_(True)
                    x3 = torch.nn.functional.conv2d(
                        x2, unblurring_matrix.unsqueeze(0).unsqueeze(0).cuda(),
                        padding=1, stride=1
                    ).clamp(0) + 1e-10
                else:
                    x3 = x2

                # Gamma Correction => C = (Max*((Image/Max)^gammaFactor))
                if self.add_gamma_param:
                    gamma_param = gamma_param.detach()
                    gamma_param.requires_grad_(True)
                    x4 = x3.max().detach() * (torch.pow(x3 / (x3.max().detach()), gamma_param))
                else:
                    x4 = x3

                x5 = apply_batch_local_torch_normalization(x4, "standardize")

                y = torch.nn.functional.softmax(self.discriminator_model(x5), dim=1)

                # Check if difference is too small => Break
                if (y.cuda() - self.target.cuda()).abs().max() <= self.out_threshold:
                    x.requires_grad_(False)
                    break

                error = self.cxe_loss(y.cuda(), self.target.cuda())

                if self.add_l1:  # ToDo: no seria x5? las primeras veces image===x
                    error = error - (torch.nn.L1Loss()(image.detach(), x) * self.l1_lambda)

                error.backward()

                if self.add_blur_param:
                    blur_param_grad = (blurring_matrix.grad.sum() - blurring_matrix.grad[1, 1]) / blur_sum
                    blur_param = blur_param - self.blur_lambda * blur_param_grad
                    blur_param = torch.clamp(blur_param, 0, 1)
                    blurring_matrix = torch.tensor(
                        [[blur_param, blur_param, blur_param],
                         [blur_param, 1, blur_param],
                         [blur_param, blur_param, blur_param]],
                        requires_grad=False)
                    blurring_matrix /= blurring_matrix.sum()

                if self.add_unblur_param:
                    unblur_param_grad = (-unblurring_matrix.grad.sum() + (9 * unblurring_matrix.grad[1, 1])) / unblur_sum
                    unblur_param = unblur_param - self.unblur_lambda * unblur_param_grad
                    unblur_param = torch.clamp(unblur_param, 0, 0.05)
                    unblurring_matrix = torch.tensor(
                        [[-unblur_param, -unblur_param, -unblur_param],
                         [-unblur_param, 1 + 8 * unblur_param, -unblur_param],
                         [-unblur_param, -unblur_param, -unblur_param]],
                        requires_grad=False)

                if self.add_gamma_param:
                    gamma_param = (gamma_param.detach() - self.gamma_lambda * gamma_param.grad).clamp(0.8, 1.2)

                x = (x.detach() - self.entropy_lambda * x.grad).clamp(0)

        if self.verbose:
            print("")
            if (iteration + 1) < self.max_iters:
                print(f"----- Early stopping at iteration {iteration} -----")
            if self.add_blur_param:
                print("Blur matrix: \n{}".format(blurring_matrix))
            if self.add_unblur_param:
                print("Unblur matrix: \n{}".format(unblurring_matrix))
            if self.add_gamma_param:
                print("Gamma param: {}".format(gamma_param))
            print("Target: {}".format(self.target))
            print("Initial y: {}".format(['%.4f' % elem for elem in initial_y.tolist()[0]]))
            print("Final y: {}".format(['%.4f' % elem for elem in y.tolist()[0]]))
            print("")

        return x5.detach(), initial_y, y


if args.target == 'A':
    target_tensor = torch.from_numpy(np.array([1.0, 0.0, 0.0]))
elif args.target == 'B':
    target_tensor = torch.from_numpy(np.array([0.0, 1.0, 0.0]))
elif args.target == 'C':
    target_tensor = torch.from_numpy(np.array([0.0, 0.0, 1.0]))
elif args.target == 'equal':
    target_tensor = torch.from_numpy(np.array([0.333, 0.333, 0.333]))
else:
    assert False, "Unknown target '{}'".format(args.target)

entropy_descriptor = ""  # "simple/"

save_dir = "entropy_images/{}vs{}/{}outThreshold{}_learningRate{}_maxIters{}_target{}".format(
    segmentation_train_fold, segmentation_val_fold, entropy_descriptor,
    args.out_threshold, args.entropy_lambda, args.max_iters, args.target
)

if args.add_l1: save_dir += "_usingL1lambda{}".format(args.l1_lambda)
if args.add_blur_param: save_dir += "_usingBlurParamlambda{}".format(args.blur_lambda)
if args.add_unblur_param: save_dir += "_usingUnblurParamlambda{}".format(args.unblur_lambda)
if args.add_gamma_param: save_dir += "_usingGammaParamlambda{}".format(args.gamma_lambda)

image_modificator_fn = ImageBackwardEntropy(
    discriminator, target_tensor, max_iters=args.max_iters,
    out_threshold=args.out_threshold, entropy_lambda=args.entropy_lambda, verbose=args.verbose,
    add_l1=args.add_l1, l1_lambda=args.l1_lambda, add_blur_param=args.add_blur_param, blur_lambda=args.blur_lambda,
    add_unblur_param=args.add_unblur_param, unblur_lambda=args.unblur_lambda,
    add_gamma_param=args.add_gamma_param, gamma_lambda=args.gamma_lambda,
)

print("Start validation evaluation...")
train_csv = pd.read_csv("utils/data/train.csv")
stats = val_step_experiments(
    segmentation_loader, segmentator, val_same_patients, train_csv,
    num_classes=4, generate_imgs=args.generate_images, image_modificator_fn=image_modificator_fn,
    save_dir=save_dir, verbose=args.verbose,
)

print(stats.groupby("Vendor")["IOU_MEAN"].mean())

stats.groupby("Vendor")["IOU_MEAN"].mean().plot.bar(color=colors)
# -------------------------------------------------------------- #
plt.ylabel("Mean IOU")
plt.xticks(rotation='horizontal')
plt.yticks(np.arange(0, stats.groupby("Vendor")["IOU_MEAN"].mean().max() + 0.05, .05))
plt.title("Mean IOU by Vendor")
plt.grid()
plt.savefig(os.path.join(save_dir, 'iou_vendor.png'), bbox_inches='tight', dpi=160)
