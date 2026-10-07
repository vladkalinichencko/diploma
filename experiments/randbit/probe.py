"""Frozen-encoder readouts on h.

class_accuracy: logistic regression STL-10 class, fit on the 5000 train images, scored on 8000 test.
bit_accuracy:   ridge from h to the bits, fit on train, sign accuracy on test (0.5 = no bits in h).
Both use the clean, un-augmented images with their own random bits.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from data import with_bits


@torch.no_grad()
def features(encoder, image_NCHW, bits_NK, batch=1000):
    encoder.eval()
    h = [encoder(with_bits(image_NCHW[i : i + batch], bits_NK[i : i + batch])) for i in range(0, len(image_NCHW), batch)]
    encoder.train()
    return torch.cat(h).cpu().numpy()


def readout(encoder, data):
    h_train = features(encoder, data["train_image"], data["train_bits"])
    h_test = features(encoder, data["test_image"], data["test_bits"])
    scaler = StandardScaler().fit(h_train)
    h_train, h_test = scaler.transform(h_train), scaler.transform(h_test)

    y_train, y_test = data["train_label"].cpu().numpy(), data["test_label"].cpu().numpy()
    classifier = LogisticRegression(max_iter=2000, C=0.1).fit(h_train, y_train)
    result = {"class_accuracy": float(classifier.score(h_test, y_test))}

    bits_train, bits_test = data["train_bits"].cpu().numpy(), data["test_bits"].cpu().numpy()
    if bits_train.shape[1] > 0:
        predicted = Ridge(alpha=1.0).fit(h_train, bits_train).predict(h_test)
        result["bit_accuracy"] = float(((predicted > 0.5) == (bits_test > 0.5)).mean())
    return result
