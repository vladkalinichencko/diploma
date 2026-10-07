"""SimCLR augmentations on a GPU batch: crop, flip, color jitter, grayscale.

Applied to RGB in [0, 1] before the bits are appended, so the bits never change.
"""

import torch
import torch.nn.functional as F


def random_resized_crop(image_NCHW, scale=(0.2, 1.0)):
    n = image_NCHW.shape[0]
    device = image_NCHW.device
    area = torch.empty(n, device=device).uniform_(*scale)
    log_ratio = torch.empty(n, device=device).uniform_(-0.288, 0.288)  # aspect 3/4..4/3
    ratio = log_ratio.exp()
    width = (area * ratio).sqrt().clamp(max=1.0)
    height = (area / ratio).sqrt().clamp(max=1.0)
    center_x = (torch.rand(n, device=device) * 2 - 1) * (1 - width)
    center_y = (torch.rand(n, device=device) * 2 - 1) * (1 - height)
    flip = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
    theta_N23 = torch.zeros(n, 2, 3, device=device)
    theta_N23[:, 0, 0] = width * flip
    theta_N23[:, 0, 2] = center_x
    theta_N23[:, 1, 1] = height
    theta_N23[:, 1, 2] = center_y
    grid = F.affine_grid(theta_N23, list(image_NCHW.shape), align_corners=False)
    return F.grid_sample(image_NCHW, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def gray(image_NCHW):
    weight_C = torch.tensor([0.299, 0.587, 0.114], device=image_NCHW.device).view(1, 3, 1, 1)
    return (image_NCHW * weight_C).sum(1, keepdim=True)


def color_jitter(image_NCHW, strength=0.4, p=0.8):
    n = image_NCHW.shape[0]
    device = image_NCHW.device

    def factor():
        on = (torch.rand(n, 1, 1, 1, device=device) < p).float()
        return 1 + on * torch.empty(n, 1, 1, 1, device=device).uniform_(-strength, strength)

    out = image_NCHW * factor()  # brightness
    mean = gray(out).mean(dim=(2, 3), keepdim=True)
    out = (out - mean) * factor() + mean  # contrast
    out = (out - gray(out)) * factor() + gray(out)  # saturation
    return out.clamp(0, 1)


def random_grayscale(image_NCHW, p=0.2):
    on = (torch.rand(image_NCHW.shape[0], 1, 1, 1, device=image_NCHW.device) < p).float()
    return on * gray(image_NCHW).expand_as(image_NCHW) + (1 - on) * image_NCHW


def simclr_view(image_NCHW):
    return random_grayscale(color_jitter(random_resized_crop(image_NCHW)))
