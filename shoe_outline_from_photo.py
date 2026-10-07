"""Factory-insole outline (mm, insole frame) from a top-down photo.

    python shoe_outline_from_photo.py <photo.jpg> <printed_insole.stl> <out.json>
    python shoe_outline_from_photo.py mirror <src_outline.json> <dst_outline.json>

Photo: factory insole on a light-teal cutting mat with the printed insole (same STL) lying on it, top view.
The printed insole is the scale/pose reference: its known outline is fitted (rotation, scale, shift) to its
white silhouette, then the union of white + dark pixels (the factory insole) is mapped back to mm.
If <out.json> already exists the photo only ADDS to it: a print that overhangs hides the factory edge, so the photo can
only show where the factory insole sticks out past the print (protrusions >= 2 mm wide); everything else is kept.
Each add-on is recorded in the json ("addons"), so it can be copied to the other foot: `mirror` mirrors the source
outline's add-ons onto the destination outline (the two insoles are mirror images), registering the source's base
shape to the destination first (needs IoU >= 0.90) and closing the seam. Writes <out.json> and (photo mode) <out>.png.
"""
import json
import os
import sys

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageOps
from scipy.optimize import minimize
import shapely
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

from insole_pipeline import smooth_polygon

SCALE = 0.5            # photo downscale for processing
MIN_IOU = 0.95         # registration must explain the white silhouette at least this well
CLEAN_R = 5.0          # mm, opening/closing radius to remove fabric-edge and label artifacts
MERGE_CLOSE = 5.0      # mm, closing radius where an add-on meets the outline it is merged into
MIN_MIRROR_IOU = 0.90  # `mirror`: the mirrored source outline must match the destination outline at least this well


def load_photo(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.array(im.resize((int(im.width * SCALE), int(im.height * SCALE)), Image.LANCZOS))


def segment(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H, S, V = (hsv[..., i].astype(int) for i in range(3))
    teal = (H >= 80) & (H <= 105) & (S >= 15) & (V >= 80)
    mat = cv2.morphologyEx(teal.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mat)
    areas = st[1:, cv2.CC_STAT_AREA]
    # the insole can split the mat into pieces; take every substantial piece, not just the largest
    big = np.isin(lab, 1 + np.nonzero(areas >= 0.05 * areas.max())[0]).astype(np.uint8)
    hull = cv2.convexHull(np.vstack(cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]))
    region = np.zeros(mat.shape, np.uint8)
    cv2.fillConvexPoly(region, hull, 1)
    white = ((S <= 45) & (V >= 140) & ~teal & (region > 0)).astype(np.uint8)  # mat can be nearly as unsaturated as the insole: separate by hue
    dark = ((V <= 75) & (region > 0)).astype(np.uint8)

    Hh = white.shape[0]
    w = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(w)
    cands = [i for i in range(1, n) if cen[i][1] > 0.35 * Hh and st[i, cv2.CC_STAT_AREA] > 50000]  # skip the cut-off insole at the top
    wmain = (lab == max(cands, key=lambda i: st[i, cv2.CC_STAT_AREA])).astype(np.uint8)
    d = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    n2, lab2, _, _ = cv2.connectedComponentsWithStats(d)
    touch = [t for t in np.unique(lab2[cv2.dilate(wmain, np.ones((9, 9), np.uint8)) > 0]) if t != 0]
    dmain = np.isin(lab2, touch).astype(np.uint8)
    # drop dark blobs far from the printed insole (e.g. another insole in frame); the factory insole protrudes < ~25 mm (~110 px)
    near = cv2.distanceTransform(1 - wmain, cv2.DIST_L2, 5) <= 120
    dmain = (dmain > 0) & near
    dmain = dmain.astype(np.uint8)
    return wmain, dmain


def printed_outline(stl):
    """Top-down silhouette (what the photo sees); a Z section would miss the underside bevel/pocket."""
    sil = trimesh.path.polygons.projected(trimesh.load(stl), normal=[0, 0, 1])
    return max(getattr(sil, "geoms", [sil]), key=lambda p: p.area)


def to_px(xy, p):
    th, s, tx, ty = p
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return s * np.column_stack([xy[:, 0], -xy[:, 1]]) @ R.T + [tx, ty]


def to_mm(px, p):
    th, s, tx, ty = p
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    v = (px - [tx, ty]) / s @ R
    return np.column_stack([v[:, 0], -v[:, 1]])


def register(poly, target):
    xy = np.array(poly.exterior.coords)

    def cost(p):
        m = np.zeros(target.shape, np.uint8)
        cv2.fillPoly(m, [np.round(to_px(xy, p)).astype(np.int32)], 1)
        return 1 - np.logical_and(m, target).sum() / np.logical_or(m, target).sum()

    ys, xs = np.nonzero(target)
    c = np.array([xs.mean(), ys.mean()])
    ax = np.linalg.eigh(np.cov((np.column_stack([xs, ys]) - c).T))[1][:, 1]
    pc = np.array(poly.centroid.coords[0])
    best = None
    for flip in (0.0, np.pi):
        th0 = np.arctan2(ax[1], ax[0]) + np.pi / 2 + flip
        for s0 in (4.5, 5.0, 5.5):
            R = np.array([[np.cos(th0), -np.sin(th0)], [np.sin(th0), np.cos(th0)]])
            p0 = np.array([th0, s0, *(c - s0 * (np.array([pc[0], -pc[1]]) @ R.T))])
            r = minimize(cost, p0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 1500})
            if best is None or r.fun < best.fun:
                best = r
    return best.x, 1 - best.fun


def merge_outline(base, extra):
    """base + extra as one smoothed polygon. The add-on only touches the base outside the print, which can leave a
    slit/hole between them: close it (MERGE_CLOSE)."""
    closed = unary_union([base, extra]).buffer(MERGE_CLOSE).buffer(-MERGE_CLOSE)
    return smooth_polygon(Polygon(max(getattr(closed, "geoms", [closed]), key=lambda g: g.area).exterior))


def addon_record(source, region):
    parts = [g for g in getattr(region, "geoms", [region]) if g.geom_type == "Polygon" and g.area > 4.0]
    return {"source": source, "rings_mm": [np.array(g.exterior.coords).round(3).tolist() for g in parts]}


def addon_region(js):
    return unary_union([Polygon(r) for a in js.get("addons", []) for r in a["rings_mm"]])


def main(photo, stl, out_json):
    img = load_photo(photo)
    wmain, dmain = segment(img)
    printed = printed_outline(stl)
    params, iou = register(printed, wmain)
    if iou < MIN_IOU:
        raise RuntimeError(f"registration IoU {iou:.3f} < {MIN_IOU}: check photo/segmentation")
    F = cv2.morphologyEx(np.maximum(wmain, dmain), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnt = max(cv2.findContours(F, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)[:, 0, :].astype(float)
    poly = Polygon(to_mm(cnt, params)).buffer(0)
    poly = poly.buffer(-CLEAN_R, resolution=16).buffer(2 * CLEAN_R, resolution=16).buffer(-CLEAN_R, resolution=16)  # open + close
    poly = smooth_polygon(poly)
    prev_js = json.load(open(out_json)) if os.path.exists(out_json) else {}
    addons = prev_js.get("addons", [])
    if prev_js:
        extra = poly.difference(printed).buffer(-1.0).buffer(1.0)
        poly = merge_outline(Polygon(prev_js["points_mm"]).buffer(0), extra)
        addons = addons + [addon_record(photo, extra)]
    json.dump({"points_mm": np.array(poly.exterior.coords).round(3).tolist(), "px_per_mm": round(params[1] / SCALE, 3), "iou": round(iou, 4),
               "source_photo": photo, "addons": addons}, open(out_json, "w"))

    ov = img.copy()
    cv2.polylines(ov, [np.round(to_px(np.array(printed.exterior.coords), params)).astype(np.int32)], True, (0, 90, 255), 2)
    cv2.polylines(ov, [np.round(to_px(np.array(poly.exterior.coords), params)).astype(np.int32)], True, (255, 0, 0), 2)
    Image.fromarray(ov).save(out_json.replace(".json", ".png"))
    fb, pb = poly.bounds, printed.bounds
    print(f"IoU {iou:.3f}  {params[1] / SCALE:.2f} px/mm  factory {fb[2]-fb[0]:.1f} x {fb[3]-fb[1]:.1f} mm  printed {pb[2]-pb[0]:.1f} x {pb[3]-pb[1]:.1f} mm")


def mirror(src_json, dst_json):
    """Copy the photo add-ons of one foot's outline onto the other foot (factory insoles are mirrored left/right).
    The source outline minus its add-ons is mirrored and registered (shift + rotation) onto the destination outline;
    the add-ons then go through the same transform and are merged into the destination."""
    src, dst = (json.load(open(p)) for p in (src_json, dst_json))
    add = addon_region(src)
    if add.is_empty:
        raise SystemExit(f"{src_json} has no recorded add-ons (run shoe_outline_from_photo.py with a photo on it first)")
    S, D = Polygon(src["points_mm"]).buffer(0), Polygon(dst["points_mm"]).buffer(0)
    base = S.difference(add)
    # snap to 1 um: reflected/rotated copies of near-identical outlines otherwise trip GEOS "side location conflict"
    snap = lambda g: shapely.set_precision(shapely.make_valid(g), 1e-3)
    mir = lambda g: affinity.scale(g, -1, 1, origin=(0, 0))
    place = lambda g, q: snap(affinity.translate(affinity.rotate(mir(g), q[2], origin=(0, 140)), q[0], q[1]))
    iou = lambda a, b: a.intersection(b).area / a.union(b).area
    runs = [minimize(lambda q: 1 - iou(D, place(base, q)), q0, method="Nelder-Mead", options={"xatol": 0.02, "fatol": 1e-6, "maxiter": 600})
            for q0 in ((0, 0, 0), (3, -2, 0), (3, 0, 2), (3, 0, -2))]
    best = min(runs, key=lambda r: r.fun)
    if 1 - best.fun < MIN_MIRROR_IOU:
        raise SystemExit(f"mirrored {src_json} matches {dst_json} only to IoU {1 - best.fun:.3f} (< {MIN_MIRROR_IOU}): not the same insole model?")
    q = best.x
    extra = place(add, q).difference(D).buffer(-1.0).buffer(1.0)
    poly = merge_outline(D, extra)
    out = dict(dst)
    out.update({"points_mm": np.array(poly.exterior.coords).round(3).tolist(),
                "addons": dst.get("addons", []) + [addon_record(f"mirrored from {os.path.basename(src_json)}", extra)],
                "mirror_registration": {"source": os.path.basename(src_json), "dx_mm": round(q[0], 2), "dy_mm": round(q[1], 2),
                                        "rot_deg": round(q[2], 2), "iou": round(1 - best.fun, 3)}})
    json.dump(out, open(dst_json, "w"))
    print(f"mirrored {os.path.basename(src_json)} -> {os.path.basename(dst_json)}: registration IoU {1 - best.fun:.3f} "
          f"(shift {q[0]:.1f}, {q[1]:.1f} mm, rotation {q[2]:.1f} deg); added {poly.difference(D).area:.0f} mm2, outline {D.area:.0f} -> {poly.area:.0f} mm2")
    return poly


if __name__ == "__main__":
    if sys.argv[1:2] == ["mirror"]:
        mirror(*sys.argv[2:4])
    else:
        main(*sys.argv[1:4])
