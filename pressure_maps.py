# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 JoshKirk800
"""Walking pressure maps from a Volumental scan, placed in the insole frame.

    python pressure_maps.py <scan_dir> [--insole-dir generated_insoles/vNN_label] [--out DIR]

<scan_dir> needs the per-step pressure files (from my.volumental.com):
    <i>.json (type left/right, nbFrames, height, width)  <i>.bin (float32 frames [nbFrames, height, width])
    footaxis_<i>.json (h4 = heel point, m2 = 2nd-metatarsal point, mm in the step's own crop frame)
    left.stl, right.stl
Steps missing any of these are skipped (and reported). Without any usable step the tool writes a summary of the
scan-level metrics only (pressure_measurement / running_pressure_measurement / kinetic_profile / fitstation).

Each step is normalised to its own peak, rotated so the heel->2nd-metatarsal axis is +Y with the heel at the
insole's heel, then averaged per foot. Alignment is only as good as one sensor row (~8 mm); a side with < 3 steps
is flagged low confidence.

Writes to --out (default <scan_dir>/pressure_maps): pressure_maps.png, pressure_summary.json, pressure_maps.npz.
With --insole-dir the insole silhouette and infill zones (<side>_zone_NN_<pct>pct.stl) are overlaid and the
pressure under each zone is reported; without it the foot outline from the scan is used.
"""
import argparse
import glob
import json
import os
import re
import sys
from functools import reduce

import numpy as np
import shapely
import shapely.geometry as sg
import trimesh
from scipy.interpolate import griddata
from shapely.ops import unary_union

import insole_pipeline as ip

SENSOR_W, SENSOR_H = 5.08, 7.62   # mm per sensor; overridden by fitstation.json metadata when present
GRID = 2.0                        # mm, output grid pitch
MIN_STEPS = 3                     # fewer steps per foot -> low-confidence flag
MIN_FOOTPRINT_MM = 150.0          # steps whose pressure crop is shorter than this are partial footprints (skipped)
LENGTH_BANDS = {"heel": (0.0, 0.28), "midfoot": (0.28, 0.58), "forefoot": (0.58, 0.80), "toes": (0.80, 1.0)}
HOTSPOT_FRACTION = 0.9            # hotspot = centroid of the cells above this fraction of the band maximum
HEEL_HOTSPOT_MAX, FOREFOOT_HOTSPOT = 0.30, (0.58, 0.85)   # fractions of insole length


def load_json(path, default=None):
    return json.load(open(path)) if os.path.exists(path) else default


# ── Steps ──────────────────────────────────────────────────────────────────────
def load_steps(scan_dir, exclude=frozenset()):
    """Per-step peak-pressure maps + axis. Returns (usable steps, [(index, reason skipped)])."""
    fs = load_json(os.path.join(scan_dir, "fitstation.json"), {})
    meta = fs.get("metadata", {}) if isinstance(fs, dict) else {}
    sw, sh = meta.get("sensor_width", SENSOR_W), meta.get("sensor_height", SENSOR_H)
    steps, skipped = [], []
    idx = sorted(int(m.group(1)) for f in glob.glob(os.path.join(scan_dir, "*.bin")) if (m := re.fullmatch(r"(\d+)\.bin", os.path.basename(f))))
    for i in idx:
        m = load_json(os.path.join(scan_dir, f"{i}.json"))
        axis = load_json(os.path.join(scan_dir, f"footaxis_{i}.json"), {}).get("result")
        if m is None or axis is None:
            skipped.append((i, "missing " + ("<i>.json" if m is None else f"footaxis_{i}.json")))
            continue
        n, h, w = m["nbFrames"], m["height"], m["width"]
        raw = np.fromfile(os.path.join(scan_dir, f"{i}.bin"), dtype="<f4")
        if raw.size != n * h * w:
            skipped.append((i, f"{i}.bin size {raw.size} != {n}x{h}x{w}"))
            continue
        if h * sh < MIN_FOOTPRINT_MM:
            skipped.append((i, f"partial footprint: {h * sh:.0f} mm long crop (< {MIN_FOOTPRINT_MM:.0f} mm)"))
            continue
        if i in exclude:
            skipped.append((i, "excluded by --exclude"))
            continue
        peak = raw.reshape(n, h, w).max(axis=0)
        steps.append({"index": i, "side": m["type"], "peak_map": peak, "peak_value": float(peak.max()), "frames": n,
                      "axis": axis, "sensor": (sw, sh)})
    # a stance much longer than the others (standing, shuffling, double contact) smears the peak map: warn, don't drop
    med = float(np.median([s["frames"] for s in steps])) if steps else 0.0
    for s in steps:
        s["warning"] = f"long contact ({s['frames']} frames vs median {med:.0f})" if s["frames"] > 2 * med else None
        if s["warning"]:
            print(f"step {s['index']}: WARNING {s['warning']}; consider --exclude {s['index']}")
    return steps, skipped


def step_points(step):
    """Sensor centres (x, y) in the foot frame: origin at the heel point, +Y toward the 2nd metatarsal, +X to the
    right of +Y. Returns points [N, 2], normalised pressure [N], axis length (mm)."""
    pk = step["peak_map"]
    h, w = pk.shape
    sw, sh = step["sensor"]
    rr, cc = np.mgrid[0:h, 0:w]
    pts = np.column_stack([((cc + 0.5) * sw).ravel(), ((rr + 0.5) * sh).ravel()])
    a = step["axis"]
    h4, m2 = np.array([a["h4x"], a["h4y"]]), np.array([a["m2x"], a["m2y"]])
    v = (m2 - h4) / np.linalg.norm(m2 - h4)
    u = np.array([v[1], -v[0]])
    return np.column_stack([(pts - h4) @ u, (pts - h4) @ v]), (pk / pk.max()).ravel(), float(np.linalg.norm(m2 - h4))


# ── Geometry ───────────────────────────────────────────────────────────────────
def scan_foot(scan_dir, side):
    """Foot outline (insole frame, Y=0 heel) and medial direction from the scan mesh."""
    scan = trimesh.load(os.path.join(scan_dir, f"{side}.stl"))
    xs, ys, z = ip.plantar_surface(scan)
    zc = np.where(z <= ip.PLANTAR_MAX, z, np.nan)
    return ip.smooth_polygon(ip.alpha_outline(xs, ys, zc)), bool(ip.medial_is_positive_x(xs, ys, zc, scan.bounds[1, 1]))


def insole_outline(insole_dir, side):
    sil = trimesh.path.polygons.projected(trimesh.load(os.path.join(insole_dir, f"{side}_insole.stl")), normal=[0, 0, 1])
    return max(getattr(sil, "geoms", [sil]), key=lambda p: p.area)


def zone_footprints(insole_dir, side):
    """[(name, density %, polygon)] from the zone STLs (even-odd, so ring holes stay holes)."""
    out = []
    for f in sorted(glob.glob(os.path.join(insole_dir, f"{side}_zone_*.stl"))):
        sec = trimesh.load(f).section(plane_origin=[0, 0, 0.5], plane_normal=[0, 0, 1])
        loops = [sg.Polygon(p[:, :2]).buffer(0) for p in sec.discrete]
        out.append((os.path.basename(f), int(re.search(r"_(\d+)pct", f).group(1)), reduce(lambda a, b: a.symmetric_difference(b), loops)))
    return out


# ── Maps and statistics ────────────────────────────────────────────────────────
def side_map(steps, outline):
    b = outline.bounds
    gx, gy = np.meshgrid(np.arange(b[0], b[2], GRID), np.arange(b[1], b[3], GRID))
    heel_x = float(np.mean(np.array(outline.intersection(sg.LineString([(b[0] - 5, b[1] + 20), (b[2] + 5, b[1] + 20)])).coords)[:, 0]))
    acc, info = [], []
    for s in steps:
        P, p, axis_len = step_points(s)
        acc.append(griddata(P + [heel_x, b[1]], p, (gx, gy), method="linear", fill_value=0.0))
        info.append({"step": s["index"], "peak": round(s["peak_value"], 2), "axis_mm": round(axis_len, 0), "frames": s["frames"], "warning": s["warning"]})
    M = np.mean(acc, axis=0)
    M[~shapely.contains_xy(outline, gx, gy)] = np.nan
    return gx, gy, M, info


def region_stats(gx, gy, M, outline, medial_pos):
    b = outline.bounds
    f = (gy - b[1]) / (b[3] - b[1])
    ys, lo, hi = ip.row_extents(outline)
    ml = np.clip((gx - np.interp(gy, ys, lo)) / np.maximum(np.interp(gy, ys, hi) - np.interp(gy, ys, lo), 1e-6), 0, 1)
    if not medial_pos:
        ml = 1 - ml
    ok = ~np.isnan(M)
    out = {}
    for name, (f0, f1) in LENGTH_BANDS.items():
        for half, (m0, m1) in {"lateral": (0, 0.5), "medial": (0.5, 1.01)}.items():
            sel = ok & (f >= f0) & (f < f1) & (ml >= m0) & (ml < m1)
            out[f"{name} {half}"] = {"max": round(float(M[sel].max()), 2), "loaded_fraction": round(float((M[sel] > 0.05).mean()), 2)} if sel.any() else None
    def hot(lo_f, hi_f):
        """Pressure-weighted centroid of the cells within 90% of the band's maximum. (A bare argmax jumps across a flat
        plateau on float noise; the centroid does not.)"""
        sel = ok & (f >= lo_f) & (f < hi_f)
        top = sel & (M >= HOTSPOT_FRACTION * np.nanmax(np.where(sel, M, np.nan)))
        w = M[top]
        x, y = float((gx[top] * w).sum() / w.sum()), float((gy[top] * w).sum() / w.sum())
        mlc = (x - np.interp(y, ys, lo)) / max(np.interp(y, ys, hi) - np.interp(y, ys, lo), 1e-6)
        return {"x_mm": round(x, 1), "y_mm": round(y, 1), "medial_fraction": round(float(np.clip(mlc if medial_pos else 1 - mlc, 0, 1)), 2),
                "value": round(float(np.nanmax(np.where(sel, M, np.nan))), 2), "cells_in_hotspot": int(top.sum())}
    return out, {"heel": hot(0, HEEL_HOTSPOT_MAX), "forefoot": hot(*FOREFOOT_HOTSPOT)}


def zone_stats(gx, gy, M, zones):
    total = float(np.nansum(M))
    rows = []
    for name, dens, poly in zones:
        sel = shapely.contains_xy(poly, gx, gy) & ~np.isnan(M)
        if sel.any():
            rows.append({"zone": name, "density_pct": dens, "mean_pressure": round(float(M[sel].mean()), 3),
                         "share_of_total_load": round(float(M[sel].sum() / total), 3), "area_mm2": round(float(poly.area), 0)})
    return rows


# ── Scan-level summary (always available) ──────────────────────────────────────
def scan_summary(scan_dir):
    out = {}
    for name in ("pressure_measurement", "running_pressure_measurement", "kinetic_profile"):
        d = load_json(os.path.join(scan_dir, name + ".json"))
        if d is not None:
            out[name] = d
    # acquire_scan.py saves the summary as fitstation_summary.json; older downloads have it as fitstation.json
    for fname in ("fitstation_summary.json", "fitstation.json"):
        fs = load_json(os.path.join(scan_dir, fname))
        if isinstance(fs, dict) and "data" not in fs:
            out["fitstation"] = fs
            break
    return out


# ── Plot ───────────────────────────────────────────────────────────────────────
def plot(results, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 11), squeeze=False)
    for ax, (side, r) in zip(axes[0], results.items()):
        im = ax.pcolormesh(r["gx"], r["gy"], r["M"], cmap="inferno", shading="auto", vmin=0, vmax=np.nanmax(r["M"]))
        ax.plot(*r["outline"].exterior.xy, color="c", lw=1.5)
        for name, dens, poly in r["zones"]:
            for g in getattr(poly, "geoms", [poly]):
                ax.plot(*g.exterior.xy, color="lime" if dens > 15 else "w", lw=0.7)
        for k, c in (("heel", "w"), ("forefoot", "w")):
            h = r["hotspots"][k]
            ax.plot(h["x_mm"], h["y_mm"], "+", color=c, ms=12)
        ax.set_aspect("equal")
        ax.set_title(f"{side}: mean normalised peak pressure, {len(r['steps'])} steps" + ("  (LOW CONFIDENCE)" if len(r["steps"]) < MIN_STEPS else ""), fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.55)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close(fig)


def main(scan_dir, insole_dir=None, out_dir=None, exclude=frozenset()):
    out_dir = out_dir or os.path.join(scan_dir, "pressure_maps")
    os.makedirs(out_dir, exist_ok=True)
    steps, skipped = load_steps(scan_dir, exclude)
    summary = {"scan_dir": os.path.abspath(scan_dir), "insole_dir": insole_dir, "scan_metrics": scan_summary(scan_dir),
               "steps_used": [], "steps_skipped": [{"index": i, "reason": r} for i, r in skipped], "sides": {}}
    for i, r in skipped:
        print(f"step {i}: skipped ({r})")
    if not steps:
        summary["note"] = "no per-step pressure frames in this scan: scan-level metrics only, no maps"
        json.dump(summary, open(os.path.join(out_dir, "pressure_summary.json"), "w"), indent=2)
        print(f"no per-step pressure frames (<i>.bin + <i>.json + footaxis_<i>.json) in {scan_dir}; wrote scan-level summary only")
        print(json.dumps(summary["scan_metrics"], indent=2))
        return summary
    results, npz = {}, {}
    for side in ("left", "right"):
        mine = [s for s in steps if s["side"] == side]
        if not mine:
            print(f"{side}: no usable steps")
            continue
        foot, medial_pos = scan_foot(scan_dir, side)
        outline = insole_outline(insole_dir, side) if insole_dir else foot
        gx, gy, M, info = side_map(mine, outline)
        regions, hotspots = region_stats(gx, gy, M, outline, medial_pos)
        zones = zone_footprints(insole_dir, side) if insole_dir else []
        results[side] = {"gx": gx, "gy": gy, "M": M, "outline": outline, "zones": zones, "hotspots": hotspots, "steps": info}
        summary["sides"][side] = {"steps": info, "low_confidence": len(mine) < MIN_STEPS, "medial_side_x": "+" if medial_pos else "-",
                                  "regions": regions, "hotspots": hotspots, "zones": zone_stats(gx, gy, M, zones)}
        npz.update({f"{side}_gx": gx, f"{side}_gy": gy, f"{side}_mean_pressure": M})
        print(f"{side}: {len(mine)} steps{' (LOW CONFIDENCE)' if len(mine) < MIN_STEPS else ''}; heel peak at Y={hotspots['heel']['y_mm']:.0f} mm, "
              f"forefoot peak at Y={hotspots['forefoot']['y_mm']:.0f} mm (medial fraction {hotspots['forefoot']['medial_fraction']})")
    summary["steps_used"] = [s["index"] for s in steps]
    plot(results, os.path.join(out_dir, "pressure_maps.png"), f"{os.path.basename(os.path.abspath(scan_dir))}" + (f" vs {os.path.basename(insole_dir)}" if insole_dir else ""))
    np.savez_compressed(os.path.join(out_dir, "pressure_maps.npz"), **npz)
    json.dump(summary, open(os.path.join(out_dir, "pressure_summary.json"), "w"), indent=2)
    print("wrote", out_dir)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir")
    ap.add_argument("--insole-dir", help="version folder with <side>_insole.stl and zone STLs to overlay")
    ap.add_argument("--out", help="output folder (default <scan_dir>/pressure_maps)")
    ap.add_argument("--exclude", default="", help="comma-separated step indices to leave out, e.g. 1,4")
    a = ap.parse_args()
    main(a.scan_dir, a.insole_dir, a.out, {int(x) for x in a.exclude.split(",") if x})
