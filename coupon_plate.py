"""Infill test coupons: one Bambu Studio plate of small TPU slabs, each with its own sparse infill, for choosing
densities by feel before they go into an insole.

    python coupon_plate.py [out_dir] [--gyroid 10,12,15,20,28,38] [--crosszag 15,28] [--thickness 4] [--size 30]

Each coupon is a size x size slab with the same top/bottom shell layers and wall loops as the insoles, so its sparse
core is as thick as the insole's (thickness - (top + bottom layers) * 0.2 mm: 3 mm at 4 mm, 4 mm at 5 mm, i.e. the
forefoot and heel floors). A label tab with the density raised on it sits in front of each slab, so the coupons can be
told apart after printing; `<name>.json` lists them (with their position on the bed).

insole_pipeline.py writes this plate into every version folder (test_coupons.3mf), using that version's shell layers
and floor thicknesses. Press each coupon with a thumb or stand on it; sort them soft -> firm and compare with the
densities the insole zones use.
"""
import argparse
import json
import os
from functools import reduce

import shapely.geometry as sg
import trimesh
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from shapely import affinity

from generate_clean_bambu_project import BED, write_plate_3mf

GYROID = (10, 12, 15, 20, 28, 38)   # %, covers the insole base (12), cushion (10), arch gradient and core (17-38)
CROSSZAG = (15, 28)                 # %, Cross Zag is Bambu's own TPU-shoe pattern; compare it against gyroid at equal density
SIZE = 30.0                         # mm, coupon edge; 40 mm coupons take ~1.8x as long to print
TAB_H, TAB_T, TEXT_T = 9.0, 2.0, 0.8   # label tab depth (Y), thickness, raised text height
MARGIN, GAP = 10.0, 6.0
LAYER = 0.2


def label_mesh(text, width, height, thickness):
    """Raised text (bold sans) fitted into width x height, centred, from z=0 to `thickness`."""
    fp = FontProperties(family="DejaVu Sans", weight="bold")
    polys = [sg.Polygon(p).buffer(0) for p in TextPath((0, 0), text, size=10, prop=fp).to_polygons(closed_only=True) if len(p) >= 3]
    shape = reduce(lambda a, b: a.symmetric_difference(b), polys)       # even-odd: counters (holes in 0, 8, A) stay holes
    x0, y0, x1, y1 = shape.bounds
    s = min(width / (x1 - x0), height / (y1 - y0))
    shape = affinity.scale(shape, s, s, origin=(0, 0))
    sx0, sy0, sx1, sy1 = shape.bounds
    mesh = trimesh.util.concatenate([trimesh.creation.extrude_polygon(g, thickness) for g in getattr(shape, "geoms", [shape]) if g.area > 0.05])
    mesh.apply_translation([(width - (sx1 - sx0)) / 2 - sx0, (height - (sy1 - sy0)) / 2 - sy0, 0])
    return mesh


def box(x0, y0, z0, dx, dy, dz):
    return trimesh.creation.box(extents=[dx, dy, dz], transform=trimesh.transformations.translation_matrix([x0 + dx / 2, y0 + dy / 2, z0 + dz / 2]))


def coupon_item(label, pattern, density, thickness, size):
    """Item frame: x 0..size; y 0..TAB_H is the label tab, y TAB_H..TAB_H+size the coupon (the infill modifier covers it)."""
    text = label_mesh(label, size - 4.0, TAB_H - 3.0, TEXT_T)
    text.apply_translation([2.0, 1.5, TAB_T])
    return {
        "name": f"{label} {thickness:g}mm",
        "parts": [
            {"name": "Coupon", "mesh": box(0, TAB_H, 0, size, size, thickness), "subtype": "normal_part"},
            {"name": "Label tab", "mesh": box(0, 0, 0, size, TAB_H, TAB_T), "subtype": "normal_part"},
            {"name": "Label text", "mesh": text, "subtype": "normal_part"},
            {"name": f"{pattern} {density}%", "mesh": box(0, TAB_H, 0, size, size, thickness + 2.0), "subtype": "modifier_part",
             "settings": {"sparse_infill_density": f"{density}%", "sparse_infill_pattern": pattern}},
        ],
    }


def write_coupon_plate(out_dir, gyroid=GYROID, crosszag=CROSSZAG, thicknesses=(4.0,), size=SIZE, top_layers=3, bottom_layers=2,
                       base_density=12, name="test_coupons"):
    """Writes <out_dir>/<name>.3mf, <name>.json (layout) and <name>.png (layout picture). Returns the layout list."""
    os.makedirs(out_dir, exist_ok=True)
    specs = [(t, "gyroid", d) for t in thicknesses for d in gyroid] + [(t, "crosszag", d) for t in thicknesses for d in crosszag]
    cols = int((BED - 2 * MARGIN + GAP) // (size + GAP))
    rows = int((BED - 2 * MARGIN + GAP) // (TAB_H + size + GAP))
    if len(specs) > cols * rows:
        raise SystemExit(f"{len(specs)} coupons do not fit one {BED:.0f} mm plate ({cols} x {rows} max at {size:g} mm)")
    items, layout = [], []
    for k, (t, pattern, d) in enumerate(specs):
        core = t - (top_layers + bottom_layers) * LAYER
        if core < 1.0:
            raise SystemExit(f"thickness {t:g} mm leaves {core:.1f} mm of sparse core with {top_layers}+{bottom_layers} shell layers (< 1 mm)")
        label = f"{'G' if pattern == 'gyroid' else 'Z'}{d}" + (f"/{t:g}" if len(thicknesses) > 1 else "")
        it = coupon_item(label, pattern, d, t, size)
        it["x"], it["y"] = MARGIN + (k % cols) * (size + GAP), MARGIN + (k // cols) * (TAB_H + size + GAP)
        items.append(it)
        layout.append({"label": label, "pattern": pattern, "density_pct": d, "thickness_mm": t, "size_mm": size,
                       "sparse_core_mm": round(core, 2), "x_mm": it["x"], "y_mm": it["y"]})
    process = {"sparse_infill_density": f"{base_density}%", "top_shell_layers": str(top_layers), "bottom_shell_layers": str(bottom_layers)}
    write_plate_3mf(items, os.path.join(out_dir, name + ".3mf"), process, plate_name="Infill coupons", title="Infill test coupons")
    json.dump({"process": process, "coupons": layout}, open(os.path.join(out_dir, name + ".json"), "w"), indent=2)
    draw_layout(layout, size, os.path.join(out_dir, name + ".png"))
    return layout


def draw_layout(layout, size, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(Rectangle((0, 0), BED, BED, fill=False, lw=1.5))
    for c in layout:
        ax.add_patch(Rectangle((c["x_mm"], c["y_mm"] + TAB_H), size, size, fc="#9ecae1" if c["pattern"] == "gyroid" else "#fdae6b", ec="k"))
        ax.add_patch(Rectangle((c["x_mm"], c["y_mm"]), size, TAB_H, fc="#eee", ec="k"))
        ax.text(c["x_mm"] + size / 2, c["y_mm"] + TAB_H / 2, c["label"], ha="center", va="center", fontsize=8, weight="bold")
        ax.text(c["x_mm"] + size / 2, c["y_mm"] + TAB_H + size / 2, f"{c['pattern']}\n{c['density_pct']}%\n{c['thickness_mm']:g} mm",
                ha="center", va="center", fontsize=7)
    ax.set_xlim(-5, BED + 5)
    ax.set_ylim(-5, BED + 5)
    ax.set_aspect("equal")
    ax.set_title("Bambu A1 bed, front at bottom: blue = gyroid, orange = cross zag", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    import insole_pipeline as ip
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", nargs="?", default="coupon_plates")
    ap.add_argument("--gyroid", default=",".join(map(str, GYROID)))
    ap.add_argument("--crosszag", default=",".join(map(str, CROSSZAG)))
    ap.add_argument("--thickness", default=f"{ip.FOREFOOT_BASE:g}", help=f"mm, comma list; {ip.FOREFOOT_BASE:g} = forefoot floor, {ip.HEEL_BASE:g} = heel floor")
    ap.add_argument("--size", type=float, default=SIZE)
    a = ap.parse_args()
    ints = lambda s: tuple(int(x) for x in s.split(",") if x)
    lay = write_coupon_plate(a.out_dir, ints(a.gyroid), ints(a.crosszag), tuple(float(x) for x in a.thickness.split(",") if x), a.size,
                             ip.TOP_LAYERS, ip.BOTTOM_LAYERS, ip.BASE_DENSITY)
    print(f"{len(lay)} coupons -> {a.out_dir}")
