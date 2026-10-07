"""Figure of the RandBit input: STL-10 images, two SimCLR views of each, and the b constant bit channels
that both views share.

Run from experiments/randbit: python3 figure_views.py  ->  ../../figures/randbit_views.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from augment import simclr_view
from data import load_stl_split, random_bits

NUM_IMAGES = 4
NUM_BITS = 8
OUT = Path(__file__).resolve().parents[2] / "figures" / "randbit_views.png"

torch.manual_seed(3)
image_NCHW = torch.from_numpy(load_stl_split("test")[[0, 7, 21, 40]]).float() / 255.0
bits_NK = random_bits(NUM_IMAGES, NUM_BITS, seed=0)
view_a_NCHW, view_b_NCHW = simclr_view(image_NCHW), simclr_view(image_NCHW)


def show_image(ax, image_CHW):
    ax.imshow(image_CHW.permute(1, 2, 0).numpy())
    ax.set_xticks([]), ax.set_yticks([])


def show_bits(ax, bits_K):
    ax.imshow(bits_K.view(2, NUM_BITS // 2).numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([x - 0.5 for x in range(1, NUM_BITS // 2)], minor=True)
    ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color="#888888", lw=1)
    ax.tick_params(which="both", length=0)
    ax.set_xticks([]), ax.set_yticks([])


titles = ["image", "view 1: RGB", "view 1: 8 bit channels", "view 2: RGB", "view 2: 8 bit channels"]
fig, axes = plt.subplots(NUM_IMAGES, 5, figsize=(9, 7.4), gridspec_kw={"width_ratios": [1, 1, 1.4, 1, 1.4]})
for row in range(NUM_IMAGES):
    show_image(axes[row, 0], image_NCHW[row])
    show_image(axes[row, 1], view_a_NCHW[row])
    show_bits(axes[row, 2], bits_NK[row])
    show_image(axes[row, 3], view_b_NCHW[row])
    show_bits(axes[row, 4], bits_NK[row])
for ax, title in zip(axes[0], titles):
    ax.set_title(title, fontsize=10)
fig.text(0.5, 0.01, "each square is one input channel, constant over all 32×32 pixels (white = +1, black = −1)", ha="center", fontsize=9)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT, dpi=160)
print(OUT)
