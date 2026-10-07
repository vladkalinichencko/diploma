"""Figure and table from logs/runs.jsonl: class accuracy and bit decodability against the number of shared bits,
and what 100 labels during pretraining change at b = 16."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = [json.loads(line) for line in Path("logs/runs.jsonl").read_text().splitlines()]
COLORS = {"infonce": "#2a6fdb", "vicreg": "#d9822b", "sigreg": "#2f9e6e", "init": "#888888"}


def curve(loss, epochs, field):
    points = sorted((r["bits"], r[field]) for r in runs if r["loss"] == loss and r["guide"] == "none" and r["epochs"] == epochs and field in r)
    return [b for b, _ in points], [a for _, a in points]


def guided(loss, guide, labels, label_bits):
    match = [r for r in runs if r["loss"] == loss and r["bits"] == 16 and r["guide"] == guide and r["labels"] == labels and r.get("label_bits", "fixed") == label_bits and r["epochs"] == 15]
    return match[0]["class_accuracy"] if match else 0.0


fig, (left, right, bars) = plt.subplots(1, 3, figsize=(15, 3.8))
for loss in ["infonce", "vicreg", "sigreg"]:
    left.plot(*curve(loss, 15, "class_accuracy"), "o-", color=COLORS[loss], label=loss)
    right.plot(*curve(loss, 15, "bit_accuracy"), "o-", color=COLORS[loss], label=loss)
left.plot(*curve("infonce", 0, "class_accuracy"), "s--", color=COLORS["init"], label="random init")
left.axhline(0.1, color="black", lw=0.6, ls=":")
left.set(xlabel="shared random bits b", ylabel="STL-10 linear accuracy (test)", title="Image content in h")
right.set(xlabel="shared random bits b", ylabel="bit sign accuracy from h (test)", title="Shortcut in h", ylim=(0.45, 1.02))
left.legend(frameon=False)

conditions = [("no labels", "none", 0, "fixed"), ("joint", "joint", 100, "fixed"), ("A-GEM", "agem", 100, "fixed"), ("joint, fresh bits", "joint", 100, "fresh")]
losses = ["infonce", "vicreg", "sigreg", "none"]
for i, (name, guide, labels, label_bits) in enumerate(conditions):
    heights = [guided(loss, guide, labels, label_bits) for loss in losses]
    bars.bar([j + 0.2 * (i - 1.5) for j in range(len(losses))], heights, width=0.2, label=name)
bars.axhline(0.1, color="black", lw=0.6, ls=":")
bars.set_xticks(range(len(losses)), ["infonce", "vicreg", "sigreg", "labels only"])
bars.set(ylabel="STL-10 linear accuracy (test)", title="b = 16, 100 labels during pretraining")
bars.legend(frameon=False, fontsize=8)
fig.tight_layout()
Path("reports").mkdir(exist_ok=True)
fig.savefig("reports/randbit_suppression.png", dpi=160)

lines = ["| loss | bits | guide | labels | label bits | epochs | class acc | bit acc |", "|---|---|---|---|---|---|---|---|"]
for r in sorted(runs, key=lambda r: (r["guide"] != "none", r["epochs"] > 0, r["loss"], r["bits"])):
    bit = f"{r['bit_accuracy']:.3f}" if "bit_accuracy" in r else "–"
    lines.append(f"| {r['loss']} | {r['bits']} | {r['guide']} | {r['labels']} | {r.get('label_bits', 'fixed') if r['labels'] else '–'} | {r['epochs']} | {r['class_accuracy']:.3f} | {bit} |")
Path("reports/randbit_table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
