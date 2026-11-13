"""Render the user's revised six-stage fraud-model workflow.

Contract: show construction of a sub-1% positive dataset, shared repeated
stratified holdouts, training-only imbalance treatments, paired RF/MLP models,
validation selection and final test evaluation. This is a protocol schematic,
not a new experiment. Only configuration and diagram artifacts are written.

The previous reference supplies the pastel bands and dashed-border style.
The new six-part user specification supplies content. PR-AUC is used for model
selection; F2 sets a classification threshold, matching the existing code.
"""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image
import pymupdf
from figure_alignment_helper import require_matplotlib_panel_alignment


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "workflow_six_stage_config.json"
C = json.loads(CONFIG.read_text(encoding="utf-8"))
OUT = ROOT / C["output_stem"]


def compact(s):
    return re.sub(r"\s+", "", s)


def main():
    # Read-only evidence check: the figure must match the selection protocol.
    experiment_config = json.loads((ROOT / "nn_comparison_config.json").read_text(encoding="utf-8"))
    assert experiment_config["selection_metric"] == "validation_pr_auc"
    assert experiment_config["threshold_metric"] == "validation_f2"
    assert experiment_config["threshold_tie_break"] == "larger_threshold"
    assert len(experiment_config["variants"]) == 7

    OUT.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [C["font"], "DejaVu Sans"],
        "font.size": C["font_size"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.color": C["text_color"], "axes.unicode_minus": False,
        "savefig.facecolor": "white",
    })
    width, height = C["width_pt"], C["height_pt"]
    fig = plt.figure(figsize=(width / 72, height / 72), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis("off")
    groups = {g["id"]: g for g in C["groups"]}
    nodes = {n["id"]: n for n in C["nodes"]}
    tracked_text = []

    def text_item(identifier, x, y, text, rect, size, bold=False):
        artist = ax.text(x, y, text, ha="center", va="center", fontsize=size,
                         fontweight="bold" if bold else "normal", zorder=4)
        tracked_text.append((identifier, artist, rect))

    for g in C["groups"]:
        x, y, w, h = g["rect"]
        ax.add_patch(Rectangle((x, y), w, h, facecolor=g["fill"],
                               edgecolor=g["border"], linewidth=.75,
                               linestyle=(0, (3.5, 3)), zorder=1))
        ax.plot([116, 116], [y + 12, y + h - 12], color=g["border"],
                alpha=.65, linewidth=.55, zorder=2)
        lines = [g["number"]] + g["label"].split("\n")
        for i, line in enumerate(lines):
            cy = y + h / 2 + (i - (len(lines) - 1) / 2) * 15
            text_item(g["id"] + "_label", 64, cy, line, [x, y, 104, h],
                      C["section_size"] if i else 9.8, bold=True)

    for node in C["nodes"]:
        x, y, w, h = node["rect"]
        g = groups[node["group"]]
        ax.add_patch(Rectangle((x, y), w, h, facecolor=g["node_fill"],
                               edgecolor=g["node_border"], linewidth=.8, zorder=3))
        for i, line in enumerate(node["lines"]):
            cy = y + h / 2 + (i - (len(node["lines"]) - 1) / 2) * node.get("line_gap", C["line_spacing_pt"])
            size = node.get("heading_size", C["heading_size"]) if i == 0 else C["font_size"]
            text_item(node["id"], x + w / 2, cy, line, node["rect"], size,
                      bold=(i == 0 and node.get("bold", True)))

    for note in C["notes"]:
        x, y = note["xy"]
        g = groups[note["group"]]
        rect = [130, g["rect"][1], 526, g["rect"][3]]
        text_item(note["id"], x, y, note["text"], rect, C["note_size"])

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8,
                                     linewidth=.9, edgecolor=C["arrow_color"],
                                     facecolor=C["arrow_color"], shrinkA=1.5,
                                     shrinkB=1.5, zorder=5))

    # These arrows order workflow stages. Stage labels explicitly define which
    # dataset is permitted at each stage; no validation/test fitting is implied.
    for a, b in zip(C["groups"][:-1], C["groups"][1:]):
        arrow((393, a["rect"][1] + a["rect"][3]), (393, b["rect"][1]))
    arrow((326, 54.5), (346, 54.5))
    arrow((383, 467.5), (403, 467.5))

    x, y, w, h = C["footer_rect"]
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#F2EFF8",
                           edgecolor="#A398B3", linewidth=.7, zorder=1))
    for i, line in enumerate(C["footer_lines"]):
        text_item("footer", x + w / 2, y + h / 2 + (i - .5) * 13.2,
                  line, C["footer_rect"], C["footer_size"])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    padding_checks = []
    for identifier, text, rect in tracked_text:
        box = text.get_window_extent(renderer).transformed(ax.transData.inverted())
        x, y, w, h = rect
        xmin, xmax = sorted([box.x0, box.x1])
        ymin, ymax = sorted([box.y0, box.y1])
        pad = min(xmin - x, x + w - xmax, ymin - y, y + h - ymax)
        assert pad >= 1.8, (identifier, text.get_text(), pad)
        padding_checks.append({"id": identifier, "text": text.get_text(),
                               "padding_pt": round(float(pad), 3)})
    for row in C["equal_rows"]:
        rects = [nodes[n]["rect"] for n in row]
        assert len({tuple(r[1:]) for r in rects}) == 1, row
        gaps = [b[0] - a[0] - a[2] for a, b in zip(rects[:-1], rects[1:])]
        assert max(gaps) - min(gaps) < .01, row
    require_matplotlib_panel_alignment(fig, json_out=OUT.with_suffix(".alignment.json"),
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    assert C["dpi"] >= 300
    fig.savefig(OUT.with_suffix(".png"), dpi=C["dpi"], facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
    plt.close(fig)

    svg_text = compact("".join(ET.parse(OUT.with_suffix(".svg")).getroot().itertext()))
    with pymupdf.open(OUT.with_suffix(".pdf")) as pdf:
        pdf_text = compact("".join(page.get_text() for page in pdf))
    expected = [item[1].get_text() for item in tracked_text]
    for text in expected + C["mandatory_terms"]:
        assert compact(text) in svg_text, ("SVG missing", text)
        assert compact(text) in pdf_text, ("PDF missing", text)
    with Image.open(OUT.with_suffix(".png")) as png:
        pixel_size = png.size
        dpi = png.info.get("dpi")
        png.thumbnail((1450, 1600), Image.Resampling.LANCZOS)
        png.save(OUT.with_suffix(".preview.png"))
    qa = {
        "content_authority": "Latest user six-stage specification; model/threshold metrics corrected against existing experiment config",
        "configuration_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "stages": [g["label"].replace("\n", "") for g in C["groups"]],
        "model_selection": experiment_config["selection_metric"],
        "threshold_selection": experiment_config["threshold_metric"],
        "threshold_tie_break": experiment_config["threshold_tie_break"],
        "mandatory_terms_verified": C["mandatory_terms"],
        "rendered_text_items_verified": len(expected),
        "repeated_boxes": "Equal width, height, baseline and gutters verified",
        "padding_checks": padding_checks,
        "pixels": pixel_size, "dpi": dpi,
        "notes": ["No experiment was rerun; no source observation or training file was altered.",
                  "Existing figures remain available under their original filenames.",
                  "The original 225-fit count concerns only the old RF nine-method learning-curve analysis and is omitted from the broader RF/MLP workflow.",
                  "Learning curves use training/validation data; test evaluation is kept independent."]
    }
    OUT.with_suffix(".content-check.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "stages": len(C["groups"]),
                      "text_items_verified_in_pdf_and_svg": len(expected),
                      "minimum_padding_pt": min(p["padding_pt"] for p in padding_checks),
                      "png_pixels": pixel_size}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
