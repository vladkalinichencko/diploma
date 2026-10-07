"""One RandBit run: does a joint-embedding loss learn the image or the shared bits?

Variants, chosen by arguments:
  --loss   infonce | vicreg | sigreg       which SSL objective
  --bits   K                               how many shared random bits ride along with each image
  --guide  none | joint | agem             how m labeled images enter pretraining
           joint: add the cross-entropy of a linear head on h to the SSL loss
           agem:  same, but the SSL gradient on the encoder is projected so it never
                  points against the label gradient (Chaudhry et al. 2019)
  --labels m                               how many labeled images (balanced over classes)
  --label-bits fixed | fresh               labeled images keep their own bits, or get new random
                                           bits every step (the label loss cannot be met by memorizing codes)

Result: one JSON line with linear class accuracy and bit decodability on the frozen encoder.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from augment import simclr_view
from data import load_randbit, with_bits
from losses import LOSSES
from model import H_DIM, make_encoder, make_projector
from probe import readout

parser = argparse.ArgumentParser()
parser.add_argument("--loss", default="infonce", choices=list(LOSSES))
parser.add_argument("--bits", type=int, default=0)
parser.add_argument("--guide", default="none", choices=["none", "joint", "agem"])
parser.add_argument("--labels", type=int, default=0)
parser.add_argument("--label-bits", default="fixed", choices=["fixed", "fresh"])
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="logs/runs.jsonl")
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
data = load_randbit(args.bits, args.seed, device)

encoder = make_encoder(args.bits).to(device)
projector = make_projector().to(device)
head = torch.nn.Linear(H_DIM, 10).to(device)
ssl_loss = LOSSES[args.loss]
optimizer = torch.optim.Adam([*encoder.parameters(), *projector.parameters(), *head.parameters()], lr=1e-3, weight_decay=1e-6)

per_class = args.labels // 10
labeled = torch.cat([torch.nonzero(data["train_label"] == c).flatten()[:per_class] for c in range(10)]) if args.labels else None


def ssl_step(index_B):
    image_BCHW, bits_BK = data["ssl_image"][index_B], data["ssl_bits"][index_B]
    z1_BD = projector(encoder(with_bits(simclr_view(image_BCHW), bits_BK)))
    z2_BD = projector(encoder(with_bits(simclr_view(image_BCHW), bits_BK)))
    return ssl_loss(z1_BD, z2_BD)


def label_step():
    index_L = labeled[torch.randint(len(labeled), (min(len(labeled), 128),), device=device)]
    image_LCHW, bits_LK = data["train_image"][index_L], data["train_bits"][index_L]
    if args.label_bits == "fresh":
        bits_LK = torch.randint(0, 2, bits_LK.shape, device=device).float()
    return F.cross_entropy(head(encoder(with_bits(simclr_view(image_LCHW), bits_LK))), data["train_label"][index_L])


def agem_backward(loss_ssl, loss_label):
    """Encoder gets proj(g_ssl) + g_label, where proj removes the part of g_ssl that opposes g_label."""
    params = list(encoder.parameters())
    g_ssl = torch.autograd.grad(loss_ssl, params, retain_graph=True)
    g_label = torch.autograd.grad(loss_label, params, retain_graph=True)
    dot = sum((a * b).sum() for a, b in zip(g_ssl, g_label))
    if dot < 0:
        norm = sum(b.square().sum() for b in g_label)
        g_ssl = [a - dot / norm * b for a, b in zip(g_ssl, g_label)]
    (loss_ssl + loss_label).backward(inputs=[*projector.parameters(), *head.parameters()])
    for p, a, b in zip(params, g_ssl, g_label):
        p.grad = a + b
    return float(dot < 0)


start = time.time()
steps_per_epoch = len(data["ssl_image"]) // args.batch
for epoch in range(args.epochs):
    order = torch.randperm(len(data["ssl_image"]), device=device)
    conflicts = 0.0
    for step in range(steps_per_epoch):
        loss = ssl_step(order[step * args.batch : (step + 1) * args.batch])
        optimizer.zero_grad()
        if args.guide == "none":
            loss.backward()
        elif args.guide == "joint":
            (loss + label_step()).backward()
        else:
            conflicts += agem_backward(loss, label_step())
        optimizer.step()
    print(f"epoch {epoch} loss {loss.item():.4f} conflicts {conflicts / steps_per_epoch:.2f} {time.time() - start:.0f}s", flush=True)

result = {**vars(args), **readout(encoder, data), "final_loss": loss.item() if args.epochs else None, "seconds": time.time() - start}
print(json.dumps(result), flush=True)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "a") as f:
    f.write(json.dumps(result) + "\n")
