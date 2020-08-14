from utils.training import *
from models import *


# -----

# Image modification using entropy
class ImageBackwardEntropy:

    def __init__(self, discriminator_model, target, max_iters=500, out_threshold=0.01, entropy_lambda=0.9,
                 add_l1=False, l1_lambda=0.0, add_blur_param=False, blur_lambda=0.0,
                 add_unblur_param=False, unblur_lambda=0.0, add_gamma_param=False, gamma_lambda=0.0, verbose=True):

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

    def apply(self, image, target=None):

        if target is None:
            target = self.target

        x = copy.deepcopy(image).detach()

        blur_param = torch.tensor(0.0, requires_grad=False)
        blur_sum, unblur_sum, iteration = 0, 0, 0

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
            if (y.cuda() - target.cuda()).abs().max() <= self.out_threshold:
                x.requires_grad_(False)
                break

            error = self.cxe_loss(y.cuda(), target.cuda())

            if self.add_l1:
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
                unblur_param = torch.clamp(unblur_param, 0, 0.15)
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

        return x5.detach()

    def apply_volume(self, volume):
        output_volume = torch.empty(volume.shape)  # .float().cuda()

        for indx, vslice in enumerate(volume):
            fn_volume = self.apply(vslice.unsqueeze(0))
            output_volume[indx, ...] = fn_volume

        return output_volume


def target_generator(target):
    if target == 'A':
        target_tensor = torch.from_numpy(np.array([1.0, 0.0, 0.0]))
    elif target == 'B':
        target_tensor = torch.from_numpy(np.array([0.0, 1.0, 0.0]))
    elif target == 'C':
        target_tensor = torch.from_numpy(np.array([0.0, 0.0, 1.0]))
    elif target == 'equal':
        target_tensor = torch.from_numpy(np.array([0.333, 0.333, 0.333]))
    else:
        assert False, "Unknown target '{}'".format(target)
    return target_tensor
