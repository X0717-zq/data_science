"""Restyle Figure 6-1 without changing its text, data, or workflow.

Contract: show the frozen five-round fraud modelling workflow and keep test
evaluation separate from all fitting and selection. The first reference is
the content authority; the second contributes only pastel horizontal groups,
dashed group borders, rectangular nodes and a clear typographic hierarchy.
Archetype: single-axis workflow schematic; no new experiment is performed.
"""
from pathlib import Path
import ast
import hashlib
import json
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from figure_alignment_helper import require_matplotlib_panel_alignment


ROOT = Path(__file__).resolve().parent
C = json.loads((ROOT / "workflow_style_config.json").read_text(encoding="utf-8"))
OUT = ROOT / C["output_stem"]


def compact(text):
    return re.sub(r"\s+", "", text)


def original_texts():
    tree = ast.parse((ROOT / C["original_script"]).read_text(encoding="utf-8-sig"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "flowchart")
    strings = []
    for statement in function.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        if isinstance(call.func, ast.Name) and call.func.id == "box":
            strings.append(ast.literal_eval(call.args[4]))
        elif isinstance(call.func, ast.Attribute) and call.func.attr == "text":
            strings.append(ast.literal_eval(call.args[2]))
    assert len(strings) == 11, "Expected ten source nodes and the protocol footer."
    return strings


def draw():
    source_text = original_texts()
    target_text = ["\n".join(n["lines"]) for n in C["nodes"]] + [C["footer"]]
    assert [compact(s) for s in source_text] == [compact(s) for s in target_text]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [C["font"], "DejaVu Sans"],
        "font.size": C["font_size"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "text.color": C["text_color"],
        "savefig.facecolor": "white",
    })
    width, height = C["width_pt"], C["height_pt"]
    fig = plt.figure(figsize=(width / 72, height / 72), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis("off")
    groups = {g["id"]: g for g in C["groups"]}
    node_map = {n["id"]: n for n in C["nodes"]}
    artists = []
    geometry = []

    for group in C["groups"]:
        x, y, w, h = group["rect"]
        ax.add_patch(Rectangle((x, y), w, h, facecolor=group["fill"],
                               edgecolor=group["border"], linewidth=.75,
                               linestyle=(0, (3.5, 3)), zorder=1))
        ax.text(54, y + h / 2, group["label"], ha="center", va="center",
                fontsize=C["section_size"], fontweight="bold", zorder=3)
        ax.plot([88, 88], [y + 13, y + h - 13], color=group["border"],
                linewidth=.55, alpha=.65, zorder=2)

    for node in C["nodes"]:
        x, y, w, h = node["rect"]
        group = groups[node["group"]]
        ax.add_patch(Rectangle((x, y), w, h, facecolor=group["node_fill"],
                               edgecolor=group["node_border"], linewidth=.8,
                               zorder=3))
        lines = node["lines"]
        line_gap = C["line_spacing_pt"]
        for index, line in enumerate(lines):
            y_text = y + h / 2 + (index - (len(lines) - 1) / 2) * line_gap
            text = ax.text(x + w / 2, y_text, line, ha="center", va="center",
                           fontsize=C["heading_size"] if index == 0 else C["font_size"],
                           fontweight="bold" if index == 0 else "normal", zorder=4)
            artists.append((node["id"], text, node["rect"]))
        geometry.append({"id": node["id"], "rect_pt": node["rect"]})

    def anchor(name, side):
        x, y, w, h = node_map[name]["rect"]
        return {"top": (x + w / 2, y), "bottom": (x + w / 2, y + h),
                "left": (x, y + h / 2), "right": (x + w, y + h / 2)}[side]

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7.5,
                                     linewidth=.85, edgecolor=C["arrow_color"],
                                     facecolor=C["arrow_color"], shrinkA=1.8,
                                     shrinkB=1.8, zorder=5, capstyle="butt"))

    arrow(anchor("data", "bottom"), anchor("holdout", "top"))
    # A shared, forward-only split feeds the three aligned data partitions.
    sx, sy = anchor("holdout", "bottom")
    branch_y = 122
    ax.plot([sx, sx], [sy + 1.8, branch_y], color=C["arrow_color"], lw=.85, zorder=5)
    left_x = anchor("train", "top")[0]
    right_x = anchor("test", "top")[0]
    ax.plot([left_x, right_x], [branch_y, branch_y], color=C["arrow_color"], lw=.85, zorder=5)
    for name in ["train", "validation", "test"]:
        tx, ty = anchor(name, "top")
        arrow((tx, branch_y), (tx, ty))
    for src, dst in [("train", "fit"), ("validation", "threshold"), ("test", "metrics")]:
        arrow(anchor(src, "bottom"), anchor(dst, "top"))
    arrow(anchor("fit", "right"), anchor("threshold", "left"))
    arrow(anchor("threshold", "right"), anchor("metrics", "left"))
    for src, dst in [("fit", "learning"), ("metrics", "summary")]:
        start = anchor(src, "bottom")
        arrow(start, (start[0], node_map[dst]["rect"][1]))

    # This footer is a protocol constraint, not a feedback connector.
    ax.add_patch(Rectangle((12, 427), 588, 21, facecolor="#F2EFF8",
                           edgecolor="#A398B3", linewidth=.7, zorder=1))
    footer = ax.text(306, 437.5, C["footer"], ha="center", va="center",
                     fontsize=C["footer_size"], zorder=3)
    artists.append(("footer", footer, [12, 427, 588, 21]))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clearance = []
    for name, text, rectangle in artists:
        bbox = text.get_window_extent(renderer).transformed(ax.transData.inverted())
        x, y, w, h = rectangle
        xmin, xmax = sorted([bbox.x0, bbox.x1])
        ymin, ymax = sorted([bbox.y0, bbox.y1])
        padding = min(xmin - x, x + w - xmax, ymin - y, y + h - ymax)
        assert padding >= 2.0, (name, text.get_text(), padding)
        clearance.append({"node": name, "text": text.get_text(),
                          "minimum_padding_pt": round(float(padding), 3)})
    # Single plotting axis; test the actual repeated node geometry separately.
    require_matplotlib_panel_alignment(fig, json_out=OUT.with_suffix(".alignment.json"),
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    for names in [("train", "validation", "test"), ("fit", "threshold", "metrics")]:
        rects = [node_map[name]["rect"] for name in names]
        assert len({(r[1], r[2], r[3]) for r in rects}) == 1
        gaps = [rects[i + 1][0] - rects[i][0] - rects[i][2] for i in (0, 1)]
        assert abs(gaps[0] - gaps[1]) < .01
    assert C["dpi"] >= 300
    fig.savefig(OUT.with_suffix(".png"), dpi=C["dpi"], facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
    plt.close(fig)

    # Verify that the exported vector artifact retains every original word.
    svg = ET.parse(OUT.with_suffix(".svg"))
    rendered_svg = compact("".join(svg.getroot().itertext()))
    coverage = []
    for source in source_text:
        assert compact(source) in rendered_svg, source
        coverage.append({"text": source, "retained": True})
    report = {
        "content_match": "PASS: all 11 original text blocks preserved",
        "content": coverage,
        "workflow_edges": [["data", "holdout"], ["holdout", "train"],
                           ["holdout", "validation"], ["holdout", "test"],
                           ["train", "fit"], ["validation", "threshold"],
                           ["test", "metrics"], ["fit", "threshold"],
                           ["threshold", "metrics"], ["fit", "learning"],
                           ["metrics", "summary"]],
        "source_script_sha256": hashlib.sha256((ROOT / C["original_script"]).read_bytes()).hexdigest(),
        "export_dimensions_pt": [width, height], "png_dpi": C["dpi"],
        "minimum_font_pt": C["footer_size"],
        "node_geometry": geometry, "text_clearance": clearance,
        "grid_check": "PASS: equal three-column widths, heights, and gutters",
        "content_authority": "User reference image 1 and the existing flowchart() source",
        "style_reference": "User reference image 2; pastel bands and dashed group boundaries only",
        "excluded_data": "None; this redraw does not fit models or modify observations"
    }
    OUT.with_suffix(".content-check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"outputs": [str(OUT.with_suffix('.' + e)) for e in ['png', 'pdf', 'svg']],
                      "content_blocks_preserved": len(coverage),
                      "minimum_text_padding_pt": min(c['minimum_padding_pt'] for c in clearance)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    draw()
