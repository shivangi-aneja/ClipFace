import torch
import numpy as np
from torchmetrics import MeanMetric

from model.stylegan_ada import SmoothUpsample, SmoothDownsample, identity


def matrix(*rows, device):
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, torch.Tensor)]
    if len(ref) == 0:
        return torch.tensor(rows, device=device).float()
    elems = [x if isinstance(x, torch.Tensor) else x * torch.ones(ref[0].shape, device=device).float() for x in elems]
    return torch.stack(elems, dim=-1).reshape(ref[0].shape + (len(rows), -1))


def translate2d(tx, ty, device=torch.device("cpu")):
    return matrix([1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1], device=device)


def translate3d(tx, ty, tz, device=torch.device("cpu")):
    return matrix([1, 0, 0, tx],
                  [0, 1, 0, ty],
                  [0, 0, 1, tz],
                  [0, 0, 0, 1], device=device)


def scale2d(sx, sy, device=torch.device("cpu")):
    return matrix([sx, 0, 0],
                  [0, sy, 0],
                  [0, 0, 1], device=device)


def scale3d(sx, sy, sz, device=torch.device("cpu")):
    return matrix([sx, 0, 0, 0],
                  [0, sy, 0, 0],
                  [0, 0, sz, 0],
                  [0, 0, 0, 1], device=device)


def rotate2d(theta, device=torch.device("cpu")):
    return matrix([torch.cos(theta), torch.sin(-theta), 0],
                  [torch.sin(theta), torch.cos(theta), 0],
                  [0, 0, 1], device=device)


def rotate3d(v, theta, device=torch.device("cpu")):
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    s = torch.sin(theta)
    c = torch.cos(theta)
    cc = 1 - c
    return matrix([vx * vx * cc + c, vx * vy * cc - vz * s, vx * vz * cc + vy * s, 0],
                  [vy * vx * cc + vz * s, vy * vy * cc + c, vy * vz * cc - vx * s, 0],
                  [vz * vx * cc - vy * s, vz * vy * cc + vx * s, vz * vz * cc + c, 0],
                  [0, 0, 0, 1], device=device)


def translate2d_inv(tx, ty, device=torch.device("cpu")):
    return translate2d(-tx, -ty, device)


def scale2d_inv(sx, sy, device=torch.device("cpu")):
    return scale2d(1 / sx, 1 / sy, device)


def rotate2d_inv(theta, device=torch.device("cpu")):
    return rotate2d(-theta, device)


class AugmentPipe(torch.nn.Module):
    def __init__(self, start_p, target, interval, fixed, batch_size):
        super().__init__()

        self.register_buffer('p', torch.ones([1]) * start_p)  # Overall multiplier for augmentation probability.
        self.p_real_signs = MeanMetric(dist_sync_on_step=True)

        self.ada_target = target
        self.batch_size = batch_size
        self.ada_interval = interval
        self.ada_kimg = 500

        # Pixel blitting.
        self.xflip = 1.  # Probability multiplier for x-flip.
        self.rotate90 = 1.  # Probability multiplier for 90 degree rotations.
        self.xint = 1.  # Probability multiplier for integer translation.
        self.xint_max = 0.125  # Range of integer translation, relative to image dimensions.

        # General geometric transformations.
        self.scale = 1.  # Probability multiplier for isotropic scaling.
        self.rotate = 1.  # Probability multiplier for arbitrary rotation.
        self.aniso = 1.  # Probability multiplier for anisotropic scaling.
        self.xfrac = 0.75  # Probability multiplier for fractional translation.
        self.scale_std = 0.2  # Log2 standard deviation of isotropic scaling.
        self.rotate_max = 1.  # Range of arbitrary rotation, 1 = full circle.
        self.aniso_std = 0.2  # Log2 standard deviation of anisotropic scaling.
        self.xfrac_std = 0.125  # Standard deviation of frational translation, relative to image dimensions.

        # Color transformations.
        self.brightness = 1.  # Probability multiplier for brightness.
        self.contrast = 1.  # Probability multiplier for contrast.
        self.lumaflip = 0.5  # Probability multiplier for luma flip.
        self.hue = 1.  # Probability multiplier for hue rotation.
        self.saturation = 1.  # Probability multiplier for saturation.
        self.brightness_std = 0.2  # Standard deviation of brightness.
        self.contrast_std = 0.5  # Log2 standard deviation of contrast.
        self.hue_max = 1  # Range of hue rotation, 1 = full circle.
        self.saturation_std = 1  # Log2 standard deviation of saturation.

        self.upsampler = SmoothUpsample()
        self.downsampler = SmoothDownsample()
        self.forward = self.forward if start_p >= 0 else identity
        self.accumulate_real_sign = self.accumulate_real_sign if not fixed else self.accumulate_real_sign_no_op
        self.heuristic_update = self.heuristic_update if not fixed else self.heuristic_update_no_op

    def accumulate_real_sign(self, sign):
        self.p_real_signs.update(sign)

    def heuristic_update(self):
        adjust = torch.sign(self.p_real_signs.compute() - self.ada_target) * (self.batch_size * self.ada_interval) / (self.ada_kimg * 1000)
        self.p.copy_((self.p + adjust).max(torch.tensor(0, device=self.p.device).float()))
        self.p_real_signs.reset()

    def forward(self, images, disable_grid_sampling=False):
        device = images.device
        batch_size, num_channels, height, width = images.shape

        # pixel blitting params

        # Initialize inverse homogeneous 2D transform: G_inv @ pixel_out ==> pixel_in
        I_3 = torch.eye(3, device=device)
        G_inv = I_3

        # Apply x-flip with probability (xflip * strength).
        if self.xflip > 0:
            i = torch.floor(torch.rand([batch_size], device=device) * 2)
            i = torch.where(torch.rand([batch_size], device=device) < self.xflip * self.p, i, torch.zeros_like(i))
            G_inv = G_inv @ scale2d_inv(1 - 2 * i, 1, device=device)

        # Apply 90 degree rotations with probability (rotate90 * strength).
        if self.rotate90 > 0:
            i = torch.floor(torch.rand([batch_size], device=device) * 4)
            i = torch.where(torch.rand([batch_size], device=device) < self.rotate90 * self.p, i, torch.zeros_like(i))
            G_inv = G_inv @ rotate2d_inv(-np.pi / 2 * i, device=device)

        # Apply integer translation with probability (xint * strength).
        if self.xint > 0:
            t = (torch.rand([batch_size, 2], device=device) * 2 - 1) * self.xint_max
            t = torch.where(torch.rand([batch_size, 1], device=device) < self.xint * self.p, t, torch.zeros_like(t))
            G_inv = G_inv @ translate2d_inv(torch.round(t[:, 0] * width), torch.round(t[:, 1] * height), device=device)

        # geometric transformations parameters

        # Apply isotropic scaling with probability (scale * strength).
        if self.scale > 0:
            s = torch.exp2(torch.randn([batch_size], device=device) * self.scale_std)
            s = torch.where(torch.rand([batch_size], device=device) < self.scale * self.p, s, torch.ones_like(s))
            G_inv = G_inv @ scale2d_inv(s, s, device=device)

        # Apply pre-rotation with probability p_rot.
        p_rot = 1 - torch.sqrt((1 - self.rotate * self.p).clamp(0, 1))  # P(pre OR post) = p
        if self.rotate > 0:
            theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.rotate_max
            theta = torch.where(torch.rand([batch_size], device=device) < p_rot, theta, torch.zeros_like(theta))
            G_inv = G_inv @ rotate2d_inv(-theta, device=device)  # Before anisotropic scaling.

        # Apply anisotropic scaling with probability (aniso * strength).
        if self.aniso > 0:
            s = torch.exp2(torch.randn([batch_size], device=device) * self.aniso_std)
            s = torch.where(torch.rand([batch_size], device=device) < self.aniso * self.p, s, torch.ones_like(s))
            G_inv = G_inv @ scale2d_inv(s, 1 / s, device=device)

        # Apply post-rotation with probability p_rot.
        if self.rotate > 0:
            theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.rotate_max
            theta = torch.where(torch.rand([batch_size], device=device) < p_rot, theta, torch.zeros_like(theta))
            G_inv = G_inv @ rotate2d_inv(-theta, device=device)  # After anisotropic scaling.

        # Apply fractional translation with probability (xfrac * strength).
        if self.xfrac > 0:
            t = torch.randn([batch_size, 2], device=device) * self.xfrac_std
            t = torch.where(torch.rand([batch_size, 1], device=device) < self.xfrac * self.p, t, torch.zeros_like(t))
            G_inv = G_inv @ translate2d_inv(t[:, 0] * width, t[:, 1] * height, device=device)

        # Execute geometric transformations.

        # Execute if the transform is not identity.
        if G_inv is not I_3 and not disable_grid_sampling:
            # Upsample.
            images = self.upsampler(images)
            G_inv = scale2d(2, 2, device=device) @ G_inv @ scale2d_inv(2, 2, device=device)
            G_inv = translate2d(-0.5, -0.5, device=device) @ G_inv @ translate2d_inv(-0.5, -0.5, device=device)
            G_inv = scale2d(2 / images.shape[3], 2 / images.shape[2], device=device) @ G_inv @ scale2d_inv(2 / images.shape[3], 2 / images.shape[2], device=device)

            # Execute transformation.
            grid = torch.nn.functional.affine_grid(theta=G_inv[:, :2, :], size=images.shape, align_corners=False)
            images = torch.nn.functional.grid_sample(images, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

            # Downsample and crop.
            images = self.downsampler(images)

        # color transformation parameters

        I_4 = torch.eye(4, device=device)
        C = I_4

        # Apply brightness with probability (brightness * strength).
        if self.brightness > 0:
            b = torch.randn([batch_size], device=device) * self.brightness_std
            b = torch.where(torch.rand([batch_size], device=device) < self.brightness * self.p, b, torch.zeros_like(b))
            C = translate3d(b, b, b, device=device) @ C

        # Apply contrast with probability (contrast * strength).
        if self.contrast > 0:
            c = torch.exp2(torch.randn([batch_size], device=device) * self.contrast_std)
            c = torch.where(torch.rand([batch_size], device=device) < self.contrast * self.p, c, torch.ones_like(c))
            C = scale3d(c, c, c, device=device) @ C

        # Apply luma flip with probability (lumaflip * strength).
        v = torch.tensor([1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3), 0], device=device).float()  # Luma axis.
        if self.lumaflip > 0:
            i = torch.floor(torch.rand([batch_size, 1, 1], device=device) * 2)
            i = torch.where(torch.rand([batch_size, 1, 1], device=device) < self.lumaflip * self.p, i, torch.zeros_like(i))
            C = (I_4 - 2 * v.ger(v) * i) @ C  # Householder reflection.

        # Apply hue rotation with probability (hue * strength).
        if self.hue > 0 and num_channels > 1:
            theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.hue_max
            theta = torch.where(torch.rand([batch_size], device=device) < self.hue * self.p, theta, torch.zeros_like(theta))
            C = rotate3d(v, theta, device=device) @ C  # Rotate around v.

        # Apply saturation with probability (saturation * strength).
        if self.saturation > 0 and num_channels > 1:
            s = torch.exp2(torch.randn([batch_size, 1, 1], device=device) * self.saturation_std)
            s = torch.where(torch.rand([batch_size, 1, 1], device=device) < self.saturation * self.p, s, torch.ones_like(s))
            C = (v.ger(v) + (I_4 - v.ger(v)) * s) @ C

        # Execute color transformations.

        # Execute if the transform is not identity.
        if C is not I_4:
            images = images.reshape([batch_size, num_channels, height * width])
            images = C[:, :3, :3] @ images + C[:, :3, 3:]
            images = images.reshape([batch_size, num_channels, height, width])

        return images

    def accumulate_real_sign_no_op(self, _sign):
        pass

    def heuristic_update_no_op(self):
        pass
