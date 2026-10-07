"""Every RandBit run of the first pass, one after another on one GPU, skipping runs already in the log.

1. Suppression curve: each loss x number of shared bits, plus the random-init readout per bit count.
2. Guidance at 16 bits: 100 labels mixed in jointly or through A-GEM, and the labels-only control.
3. The same guidance with fresh random bits on the labeled images, so labels cannot be fit through the codes.
"""

import json
import subprocess
import sys
from pathlib import Path

LOG = Path("logs/runs.jsonl")
EPOCHS = 15
BITS = [0, 2, 4, 6, 8, 16]
LOSSES = ["infonce", "vicreg", "sigreg"]

runs = [dict(loss="infonce", bits=k, guide="none", labels=0, epochs=0) for k in BITS]
runs += [dict(loss=loss, bits=k, guide="none", labels=0, epochs=EPOCHS) for loss in LOSSES for k in BITS]
runs += [dict(loss=loss, bits=16, guide=g, labels=100, epochs=EPOCHS) for loss in LOSSES for g in ["joint", "agem"]]
runs += [dict(loss="none", bits=k, guide="joint", labels=100, epochs=EPOCHS) for k in [0, 16]]
runs += [dict(loss=loss, bits=16, guide="joint", labels=100, epochs=EPOCHS, label_bits="fresh") for loss in LOSSES + ["none"]]


def key(run):
    return tuple(run.get(k, "fixed") for k in ["loss", "bits", "guide", "labels", "epochs", "label_bits"])


done = {key(json.loads(line)) for line in LOG.read_text().splitlines()} if LOG.exists() else set()
for run in runs:
    if key(run) in done:
        continue
    command = [sys.executable, "train_randbit.py", "--out", str(LOG)]
    command += [f"--{k.replace('_', '-')}={v}" for k, v in run.items()]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)
