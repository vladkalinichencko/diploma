"""STL-10 at 32x32 with n shared random bits, after Chen, Luo, Li 2021 (RandBit).

Each image gets a fixed random n-bit code. The code is appended as n constant
channels after augmentation, so both views of one image carry the same bits.
The bits are useless for the downstream class and solve instance discrimination
once 2**n is large compared to the batch.

Axes: N images, C channels, H, W pixels, K bits.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

STL_DIR = Path.home() / "VSCodeProjects/ssl-robinson-ifm/data/stl10_binary"
CACHE_DIR = Path(__file__).parent / "cache"
SIDE = 32
NUM_UNLABELED = 20_000


def load_stl_split(name, count=None):
    """uint8 STL-10 split, downsampled to 32x32, cached as .npy."""
    cache = CACHE_DIR / f"{name}_{SIDE}_{count}.npy"
    if cache.exists():
        return np.load(cache)
    raw = np.memmap(STL_DIR / f"{name}_X.bin", dtype=np.uint8, mode="r").reshape(-1, 3, 96, 96)
    raw = raw[:count] if count else raw
    image_NCHW = torch.from_numpy(np.ascontiguousarray(raw)).permute(0, 1, 3, 2).float()  # STL is column-major
    small_NCHW = F.interpolate(image_NCHW, size=(SIDE, SIDE), mode="area").round().clamp(0, 255).byte().numpy()
    CACHE_DIR.mkdir(exist_ok=True)
    np.save(cache, small_NCHW)
    return small_NCHW


def load_stl_labels(name):
    return np.fromfile(STL_DIR / f"{name}_y.bin", dtype=np.uint8).astype(np.int64) - 1


def random_bits(count, num_bits, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (count, num_bits), generator=g).float()


def load_randbit(num_bits, seed, device):
    """Everything the run needs, on device: SSL pool, labeled train, test, and their bits."""
    unlabeled_NCHW = load_stl_split("unlabeled", NUM_UNLABELED)
    train_NCHW, test_NCHW = load_stl_split("train"), load_stl_split("test")

    def to_device(x):
        return torch.from_numpy(x).to(device).float() / 255.0

    return {
        "ssl_image": to_device(unlabeled_NCHW),
        "ssl_bits": random_bits(len(unlabeled_NCHW), num_bits, seed).to(device),
        "train_image": to_device(train_NCHW),
        "train_bits": random_bits(len(train_NCHW), num_bits, seed + 1).to(device),
        "train_label": torch.from_numpy(load_stl_labels("train")).to(device),
        "test_image": to_device(test_NCHW),
        "test_bits": random_bits(len(test_NCHW), num_bits, seed + 2).to(device),
        "test_label": torch.from_numpy(load_stl_labels("test")).to(device),
    }


MEAN_C = torch.tensor([0.447, 0.440, 0.405]).view(1, 3, 1, 1)
STD_C = torch.tensor([0.260, 0.256, 0.271]).view(1, 3, 1, 1)


def with_bits(image_NCHW, bits_NK):
    """Normalize RGB and append each bit as a constant channel (bits in {-1, +1})."""
    rgb_NCHW = (image_NCHW - MEAN_C.to(image_NCHW.device)) / STD_C.to(image_NCHW.device)
    n, _, h, w = image_NCHW.shape
    bit_NKHW = (2 * bits_NK - 1).view(n, -1, 1, 1).expand(n, bits_NK.shape[1], h, w)
    return torch.cat([rgb_NCHW, bit_NKHW], dim=1)
