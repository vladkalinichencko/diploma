"""Figures 1 and 2 of the draft as hand-written SVG.

system.svg: the levels of a joint-embedding training system and the arrows between them.
atlas.svg:  for each result, which level it concludes about, which it moves, which it assumes.

Run: python3 figures/atlas.py   (writes both files next to this script)
Check: rsvg-convert -z 2 figures/atlas.svg -o /tmp/atlas.png
"""

import re
from pathlib import Path

OUT = Path(__file__).parent
FONT = "Helvetica, Arial, sans-serif"

GROUP_COLOR = {
    "Data": ("#f6efe0", "#b08a3e"),
    "Network": ("#e6eef8", "#3d6ea8"),
    "Sample": ("#e7f3ea", "#3f8a55"),
    "Objective": ("#fbebe0", "#c0662b"),
    "Time": ("#efe8f6", "#6f4fa3"),
    "Control": ("#f3f3f3", "#666666"),
    "Use": ("#e5f3f3", "#2d7f80"),
}


def rich(text):
    """Turn _{...} and ^{...} into SVG subscript and superscript spans."""
    out, pos = [], 0
    for m in re.finditer(r"([_^])\{([^}]*)\}", text):
        out.append(esc(text[pos:m.start()]))
        shift = "sub" if m.group(1) == "_" else "super"
        out.append(f'<tspan baseline-shift="{shift}" font-size="75%">{esc(m.group(2))}</tspan>')
        pos = m.end()
    out.append(esc(text[pos:]))
    return "".join(out)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=12, anchor="start", weight="normal", color="#222", rotate=None, italic=False):
    tr = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    style = ' font-style="italic"' if italic else ""
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{color}"{style}{tr}>{rich(s)}</text>')


def rect(x, y, w, h, fill="#fff", stroke="#444", dash=False, r=6, sw=1.2):
    d = ' stroke-dasharray="5 4"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'


def box(x, y, w, h, lines, group, dash=False, size=12):
    fill, stroke = GROUP_COLOR[group]
    parts = [rect(x, y, w, h, fill="#ffffff", stroke=stroke, dash=dash)]
    step = size + 3
    y0 = y + h / 2 - (len(lines) - 1) * step / 2 + size / 3
    for i, line in enumerate(lines):
        weight = "bold" if i == 0 else "normal"
        parts.append(text(x + w / 2, y0 + i * step, line, size=size if i == 0 else size - 1,
                          anchor="middle", weight=weight))
    return "\n".join(parts)


def frame(x, y, w, h, group, label=None):
    fill, stroke = GROUP_COLOR[group]
    return "\n".join([
        rect(x, y, w, h, fill=fill, stroke=stroke, r=10, sw=1.5),
        text(x + 4, y - 7, label or group, size=13, weight="bold", color=stroke),
    ])


def arrow(points, label=None, at=None, dash=False, color="#333", anchor="middle"):
    pts = " ".join(f"{x},{y}" for x, y in points)
    d = ' stroke-dasharray="5 4"' if dash else ""
    parts = [f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.4"{d} marker-end="url(#head)"/>']
    if label:
        lx, ly = at
        parts.append(text(lx, ly, label, size=11, anchor=anchor, color=color, italic=True))
    return "\n".join(parts)


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
            '<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>\n'
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>\n{body}\n</svg>\n')


def system():
    b = []
    b.append(frame(610, 50, 450, 80, "Control"))
    b.append(box(625, 63, 420, 54, ["temperature τ(Z), kernel width σ",
                                     "adaptive schedules: τ computed from the batch"], "Control"))

    b.append(frame(20, 160, 180, 240, "Data"))
    b.append(box(35, 178, 150, 46, ["data distribution", "P_{X}, inputs x"], "Data"))
    b.append(box(35, 244, 150, 140, ["relation G", "augmentations T", "graph W, Ā", "two-view operator 𝒯",
                                     "positive pairs (x, x⁺)"], "Data"))
    b.append(arrow([(110, 224), (110, 242)]))

    b.append(frame(225, 160, 360, 240, "Network"))
    b.append(text(240, 196, "parameters θ", size=12, color=GROUP_COLOR["Network"][1], weight="bold"))
    b.append('<polyline points="240,212 240,206 490,206 490,212" fill="none" stroke="#3d6ea8" stroke-width="1.4"/>')
    b.append(box(240, 225, 80, 50, ["encoder", "f_{θ}"], "Network"))
    b.append(box(335, 225, 60, 50, ["h", "repr."], "Network"))
    b.append(box(410, 225, 80, 50, ["projector", "h → z"], "Network"))
    b.append(box(505, 225, 64, 50, ["z", "embed."], "Network"))
    b.append(arrow([(320, 250), (333, 250)]))
    b.append(arrow([(395, 250), (408, 250)]))
    b.append(arrow([(490, 250), (503, 250)]))
    b.append(box(420, 325, 155, 54, ["target branch θ̄", "EMA of θ, stop-gradient"], "Network", dash=True, size=11))
    b.append(arrow([(450, 275), (450, 323)], "EMA", (456, 304), dash=True, anchor="start"))
    b.append(arrow([(185, 300), (238, 262)], "x, x⁺", (214, 297)))

    b.append(frame(610, 160, 210, 240, "Sample"))
    b.append(rect(620, 172, 190, 218, fill="#f4faf5", stroke="#3f8a55", dash=True))
    b.append(text(630, 366, "embedding distribution p_{z}", size=11, weight="bold", color="#3f8a55"))
    b.append(text(630, 382, "limit N, K → ∞", size=11, color="#3f8a55"))
    b.append(rect(635, 186, 160, 150, fill="#ffffff", stroke="#3f8a55"))
    b.append(text(715, 204, "batch Z, N × D", size=12, anchor="middle", weight="bold"))
    b.append(box(645, 214, 140, 52, ["Gram ZZ^{T}, N × N", "between samples"], "Sample", size=11))
    b.append(box(645, 274, 140, 52, ["covariance Z^{T}Z, D × D", "between dimensions"], "Sample", size=11))
    b.append(arrow([(569, 240), (633, 240)], "rows of Z", (601, 233)))
    b.append(arrow([(575, 352), (633, 320)], dash=True))
    b.append(text(497, 394, "z⁺ with no gradient", size=11, anchor="middle", italic=True, color="#333"))

    b.append(frame(845, 160, 215, 240, "Objective"))
    b.append(text(860, 186, "loss L(Z), L[p_{z}]", size=12, weight="bold"))
    b.append(box(860, 198, 185, 54, ["order 0", "value, minimizers"], "Objective", size=11))
    b.append(box(860, 262, 185, 54, ["order 1", "drift −∇_{z}L, V = μ⁺ − μ⁻"], "Objective", size=11))
    b.append(box(860, 326, 185, 54, ["order 2", "curvature ∇²_{z}L"], "Objective", size=11))
    b.append(arrow([(795, 230), (858, 230)]))

    b.append(arrow([(715, 186), (715, 119)], "Z sets τ", (720, 150), dash=True, anchor="start"))
    b.append(arrow([(1020, 117), (1020, 196)], "τ enters L", (1015, 150), dash=True, anchor="end"))

    b.append(frame(610, 430, 450, 140, "Time"))
    b.append(box(625, 455, 200, 90, ["embedding flow", "Ż = −Θ ∇_{Z}L  (ODE)", "∂_{t}p_{z}  (PDE)",
                                     "free particles: Θ = I"], "Time", size=11))
    b.append(box(845, 455, 200, 90, ["parameter flow", "θ̇ = −J^{T}∇_{Z}L  (SGD)", "Θ = JJ^{T}, ND × ND"],
                 "Time", size=11))
    b.append(arrow([(1045, 289), (1075, 289), (1075, 500), (1047, 500)], "∇_{Z}L", (1080, 395), anchor="start"))
    b.append(arrow([(845, 500), (827, 500)], "J", (836, 492)))
    b.append(arrow([(790, 455), (790, 338)], "moves Z", (785, 420), anchor="end"))
    b.append(arrow([(945, 545), (945, 610), (10, 610), (10, 140), (420, 140), (420, 204)],
                   "update θ", (480, 604)))

    b.append(frame(225, 430, 360, 140, "Use"))
    b.append(box(240, 460, 140, 60, ["linear probe", "reads h"], "Use"))
    b.append(box(405, 460, 165, 60, ["task family ℱ", "posterior η = E[Y | X]"], "Use", size=11))
    b.append(arrow([(365, 275), (365, 458)]))
    b.append(arrow([(380, 490), (403, 490)]))

    (OUT / "system.svg").write_text(svg(1110, 630, "\n".join(b)))


ROWS = [
    ("Data", "data", "data distribution P_{X}"),
    ("Data", "rel", "relation G (Ā, 𝒯)"),
    ("Network", "theta", "parameters θ"),
    ("Network", "h", "representation h"),
    ("Network", "z", "embedding z"),
    ("Sample", "batch", "batch Z (Gram, covariance)"),
    ("Sample", "law", "embedding distribution p_{z}"),
    ("Objective", "o0", "order 0: value, minimizers"),
    ("Objective", "o1", "order 1: drift −∇_{z}L"),
    ("Objective", "o2", "order 2: curvature"),
    ("Time", "eflow", "embedding flow (ODE, PDE)"),
    ("Time", "pflow", "parameter flow (SGD, Θ)"),
    ("Control", "ctrl", "control: τ, stop-gradient, EMA"),
    ("Use", "task", "task family ℱ (probe, η)"),
]

# c = conclusion, m = moves (the free variable), a = assumption.
# gives: what the result hands to practice; kind decides its colour.
COLS = [
    ("loss and its limit", "Wang-Isola (4.2-4.4)", dict(law="cm", o0="c", batch="a", theta="a", rel="a"),
     "explanation, metric", "explain"),
    ("loss and its limit", "first variation (4.16)", dict(o1="c", law="m", batch="a"), "correction", "explain"),
    ("loss and its limit", "Wang-Liu (3.2)", dict(o1="c", batch="c", ctrl="m"), "explanation", "explain"),
    ("loss and its limit", "collapse saddle (3.1)", dict(o2="c", batch="m", theta="a"), "explanation", "explain"),
    ("loss and its limit", "Zimmermann (4.9)", dict(z="cm", o0="c", data="a", rel="a", law="a"),
     "identifiability", "bound"),
    ("training", "network flow (4.1)", dict(theta="m", batch="m", eflow="c", pflow="c"), "exact bridge", "explain"),
    ("training", "linear dynamics", dict(pflow="cm", batch="c", theta="a", data="a", rel="a", ctrl="c"),
     "DirectPred", "rule"),
    ("training", "PPS (4.11-4.13)", dict(batch="cm", eflow="c", ctrl="m", o1="a", theta="a"), "τ rule", "rule"),
    ("training", "Gretton et al. (4.17)", dict(o1="c", eflow="c", theta="m", law="m", ctrl="a", data="a"),
     "correction", "explain"),
    ("training", "score matching (4.14-4.15)", dict(o1="c", theta="m", data="a"), "loss without density",
     "rule"),
    ("tasks", "Saunshi (4.5)", dict(task="c", o0="c", z="m", data="a", rel="a"), "bound", "bound"),
    ("tasks", "HaoChen (4.6)", dict(task="c", o0="c", z="m", rel="a", law="a", theta="a"), "bound", "bound"),
    ("tasks", "two-view operator (4.7)", dict(task="c", o0="c", z="m", rel="a", law="a", theta="a"),
     "bound, B(F) estimate", "measure"),
    ("tasks", "NSCL (4.8)", dict(o0="c", z="m", task="ca", theta="a"), "bound", "bound"),
    ("tasks", "directional CDNV", dict(task="c", h="m", data="a"), "bound, Ṽ estimate", "measure"),
    ("tasks", "LeJEPA, SPHERE-JEPA", dict(task="ca", law="cm"), "target distribution", "rule"),
    ("tasks", "InfoMin (6.3)", dict(rel="cm", task="a"), "principle, needs y", "explain"),
    ("instruments", "RankMe", dict(batch="c"), "measure", "measure"),
    ("instruments", "Fang uniformity", dict(batch="c"), "measure", "measure"),
    ("instruments", "drift monitor (2.5)", dict(o1="c", batch="c"), "measure, untested", "measure"),
]

KIND_COLOR = {"rule": "#2f7d3a", "measure": "#2f6fa8", "bound": "#b8651c", "explain": "#777777"}


def mark(x, y, code):
    """Draw up to three marks side by side in one cell."""
    parts, n = [], len(code)
    for i, c in enumerate(code):
        cx = x + (i - (n - 1) / 2) * 11
        if c == "c":
            parts.append(f'<circle cx="{cx}" cy="{y}" r="5.5" fill="#222"/>')
        elif c == "m":
            parts.append(f'<path d="M{cx},{y - 6.5} L{cx + 6.5},{y} L{cx},{y + 6.5} L{cx - 6.5},{y} z" fill="#c0662b"/>')
        elif c == "a":
            parts.append(f'<circle cx="{cx}" cy="{y}" r="5" fill="#fff" stroke="#3d6ea8" stroke-width="1.8"/>')
    return "\n".join(parts)


def atlas():
    left, top, rh, cw = 268, 190, 27, 36
    ncols = len(COLS)
    width = left + ncols * cw + 90
    height = top + len(ROWS) * rh + 185
    b = []

    groups = []
    for g, *_ in ROWS:
        if not groups or groups[-1][0] != g:
            groups.append([g, 0])
        groups[-1][1] += 1
    r0 = 0
    for g, n in groups:
        fill, stroke = GROUP_COLOR[g]
        y = top + r0 * rh
        b.append(f'<rect x="20" y="{y}" width="{left - 26 + ncols * cw + 6}" height="{n * rh}" fill="{fill}" '
                 f'stroke="#ffffff" stroke-width="2"/>')
        b.append(text(28, y + n * rh / 2 + 4, g, size=11, weight="bold", color=stroke))
        r0 += n

    for i, (_, key, label) in enumerate(ROWS):
        y = top + i * rh + rh / 2 + 4
        b.append(text(100, y, label, size=12))

    prev = None
    for j, (cg, name, marks, gives, kind) in enumerate(COLS):
        x = left + j * cw + cw / 2
        if cg != prev:
            if prev is not None:
                xs = left + j * cw
                b.append(f'<line x1="{xs}" y1="{top - 150}" x2="{xs}" y2="{top + len(ROWS) * rh + 150}" '
                         f'stroke="#999" stroke-width="1" stroke-dasharray="3 3"/>')
            b.append(text(left + j * cw + 4, 22, cg, size=12, weight="bold", color="#444"))
            prev = cg
        b.append(text(x + 4, top - 8, name, size=11.5, rotate=-60))
        for i, (_, key, _) in enumerate(ROWS):
            if key in marks:
                b.append(mark(x, top + i * rh + rh / 2, marks[key]))
        b.append(text(x + 4, top + len(ROWS) * rh + 10, gives, size=11, anchor="end", rotate=-60,
                      color=KIND_COLOR[kind], weight="bold"))

    for i in range(len(ROWS) + 1):
        y = top + i * rh
        b.append(f'<line x1="96" y1="{y}" x2="{left + ncols * cw}" y2="{y}" stroke="#ffffff" stroke-width="1"/>')

    ly = height - 34
    b.append(mark(30, ly - 4, "c"))
    b.append(text(42, ly, "concludes about this level", size=12))
    b.append(mark(230, ly - 4, "m"))
    b.append(text(242, ly, "moves it (free variable)", size=12))
    b.append(mark(412, ly - 4, "a"))
    b.append(text(424, ly, "assumes or idealizes it", size=12))
    b.append(text(600, ly, "blank: says nothing", size=12, color="#666"))
    ly2 = height - 12
    x0 = 30
    for kind, label in [("rule", "changes training"), ("measure", "computable on a trained network"),
                        ("bound", "holds under its assumptions"), ("explain", "explains or corrects")]:
        b.append(f'<rect x="{x0}" y="{ly2 - 10}" width="12" height="12" fill="{KIND_COLOR[kind]}"/>')
        b.append(text(x0 + 17, ly2, label, size=12))
        x0 += 17 + len(label) * 6.6 + 26

    (OUT / "atlas.svg").write_text(svg(width, height, "\n".join(b)))


if __name__ == "__main__":
    system()
    atlas()
