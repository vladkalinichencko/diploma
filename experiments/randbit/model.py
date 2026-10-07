"""Small ConvNet encoder for 32x32 inputs and an MLP projector.

encoder: (3 + K) x 32 x 32 -> h, 512 numbers (global average pool after four conv stages)
projector: h 512 -> 512 -> z 128
Probes read h, the losses read z.
"""

import torch.nn as nn

H_DIM = 512
Z_DIM = 128


def conv_stage(c_in, c_out, stride):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


def make_encoder(num_bits):
    return nn.Sequential(
        conv_stage(3 + num_bits, 64, 1),
        conv_stage(64, 128, 2),
        conv_stage(128, 256, 2),
        conv_stage(256, H_DIM, 2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )


def make_projector():
    return nn.Sequential(
        nn.Linear(H_DIM, H_DIM, bias=False),
        nn.BatchNorm1d(H_DIM),
        nn.ReLU(inplace=True),
        nn.Linear(H_DIM, Z_DIM),
    )
