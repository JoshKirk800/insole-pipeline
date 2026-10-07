# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 JoshKirk800
"""Volumental foot scan -> custom insole STLs + Bambu Studio 3MF.

    python insole_pipeline.py <scan_dir> <out_root> [label]

Each run writes a new version folder <out_root>/vNN[_label]/ (STLs, zone STLs, 3MF, preview, design.json with all
design parameters) plus inputs/ (shoe outlines, overrides and the generator code used), so printed versions are kept.

<scan_dir> needs left.stl, right.stl, measurements.json, kinetic_profile.json,
running_pressure_measurement.json (as downloaded from my.volumental.com).
Optional, per scan_dir:
  shoe_outline_<side>.json   factory-insole outline in the insole frame (see shoe_outline_from_photo.py);
                             the insole is sized to it, minus SHOE_CLEARANCE. The heel cup stays anchored
                             to the foot outline; extra width becomes a thin flange.
  design_overrides.json      {"left": {...}, "right": {...}} keys of the per-foot plan, applied last.
Foot frame: Y=0 heel, +Y toes, Z up, mm.
"""
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict

import numpy as np
import shapely
import shapely.geometry as sg
import triangle
import trimesh
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry.polygon import orient
from shapely.ops import polygonize, unary_union
from skimage.measure import find_contours

from coupon_plate import write_coupon_plate
from generate_clean_bambu_project import write_3mf

# ── Geometry constants (mm) ────────────────────────────────────────────────────
HEEL_BASE = 5.0        # floor thickness under the heel (mm)
FOREFOOT_BASE = 4.0    # floor thickness under the forefoot (mm)
BASE_BLEND = (60.0, 140.0)   # Y range over which the floor thins from HEEL_BASE to FOREFOOT_BASE
PLANTAR_MAX = 15.0      # scan heights above this are foot sides, not sole
INSET = -2.0            # foot-outline inset from foot footprint
SHOE_CLEARANCE = 1.5    # shoe outline inset (photo edge bias + fit)
CUP_H, CUP_LEN, CUP_TAPER, CUP_WALL, CUP_OUT = 14.0, 65.0, 25.0, 10.0, 4.0   # medial heel-cup wall
CUP_H_LAT, CUP_LEN_LAT = 8.0, 40.0   # lateral wall: lower and shorter (v03 fit: 14 mm lateral wall rode too high and stiff)
CUP_SIDE_ML = (0.25, 0.75)           # medial fraction over which the wall blends from lateral to medial (around the heel)
ARCH_Y = (40.0, 70.0, 130.0, 180.0)   # arch envelope: ramp start, full start, full end, ramp end
FILL_LATERAL = 0.45     # fraction of plantar gap filled on the lateral side
EXTRAP_MM = 12.0        # plantar gap fades to 0 this far outside the scanned footprint
OUTLINE_STEP = 0.6      # boundary vertex spacing
TRI_AREA = 0.6          # max triangle area (mm^2)
FIELD_RES = 0.5         # height-field grid spacing
SMOOTH_SIGMA = 1.5
EDGE_R = 3.0            # top-edge rounding radius
MIN_EDGE = 2.5          # minimum thickness where edge rounding meets the arch pocket
SHOE_BED_FRACTION = 0.0 # arch pocket depth / arch height; 0 = flat underside. Shoe-specific: set in <scan_dir>/shoe_profile.json
BED_SIGMA = 4.0         # smoothing of the pocket surface (mm)
BOTTOM_BEVEL = 0.0      # 45 deg underside edge bevel (mm rise = mm inset) at the heel; 0 = none. Shoe-specific (shoe_profile.json)
BEVEL_Y = (60.0, 90.0, 170.0)   # bevel full until Y0, gone by Y1 on the medial side (arch pocket takes over), Y2 laterally
POCKET_ML = (0.30, 0.50)        # medial fraction across which the arch pocket fades in (0 lateral .. 1 medial edge)

# ── Infill zones: nested rings, graded density, infill density is the ONLY override ─
BASE_DENSITY = 12       # gyroid %, cushion zones (heel/forefoot)
TOP_LAYERS, BOTTOM_LAYERS = 3, 2   # solid shell layers; with the floors above this leaves 3-4 mm of sparse core
ZONE_LEVELS = 3        # arch gradient steps; each band must still hold ~2 infill line spacings (see MIN_BAND_SPACINGS)
HEEL_ZONE_LEVELS = 1   # heel cushion: one soft core; finer rings were narrower than one gyroid line spacing
INFILL_LINE_W = 0.45   # mm, Bambu default sparse-infill line width for a 0.4 nozzle
MIN_BAND_SPACINGS = 2.0   # a zone part narrower than this many line spacings (INFILL_LINE_W / density) can't print its density
# Full-density core ~ medial arch (Y 74-147, mid-foot to medial edge); starts past the measured heel-pressure peak
# (Y 45-53 on the walking pressure maps) so the firm zone never sits under the heel. Ramps sized for ~5 mm bands.
ARCH_ZONE = {"y_up": (58.0, 80.0), "y_down": (140.0, 165.0), "ml": (0.22, 0.52)}
HEEL_ZONE = {"y": 46.0, "r_full": 12.0, "r_zero": 22.0}   # soft core on the heel load centroid (scan 1: Y 50-52; v03/v04 used 40)
ARCH_ZONE_PEAK, HEEL_ZONE_PEAK = 38, 10
# Heel-cup rim (the walls + flange under them): firmer than the cushion core so the cup holds the calcaneus.
RIM_Y_END = 75.0                                # rim zone ends where the cup has mostly tapered out
RIM_DENSITY_PRONATION = (30, None)              # (medial, lateral) %; None = base density (lateral wall stays flexible)
RIM_DENSITY_DEFAULT = (20, None)

# ── Analysis rules (scan data -> per-foot design) ──────────────────────────────
PRONATION_FILL, PRONATION_CORR = 0.75, 4.5
FLAT_FILL = 0.65
FLAT_ARCH_DIFF_MM = 2.0
HEEL_ASYMMETRY = 1.2


def analyze(scan_dir):
    load = lambda n: json.load(open(os.path.join(scan_dir, n)))
    m = {k: v * 1000 for k, v in load("measurements.json")["measurements"].items()}
    kin, rp = load("kinetic_profile.json"), load("running_pressure_measurement.json")
    plans = {}
    for side, other in (("left", "right"), ("right", "left")):
        pronating = kin[side]["pressurePattern"] == "pronation"
        arch_diff = m[f"{other}_arch_height"] - m[f"{side}_arch_height"]
        heel_ratio = rp[f"{side}MaximumHeelPressure"] / rp[f"{other}MaximumHeelPressure"]
        plan = {"fill_medial": FILL_LATERAL, "correction_mm": 0.0,
                "arch_zone_peak": None, "heel_zone_peak": HEEL_ZONE_PEAK if heel_ratio >= HEEL_ASYMMETRY else None,
                "rim_density": RIM_DENSITY_PRONATION if pronating else RIM_DENSITY_DEFAULT, "label": "Neutral"}
        if pronating:
            plan.update(fill_medial=PRONATION_FILL, correction_mm=PRONATION_CORR, arch_zone_peak=ARCH_ZONE_PEAK, label="Corrective")
        elif arch_diff >= FLAT_ARCH_DIFF_MM:
            plan.update(fill_medial=FLAT_FILL, correction_mm=round(arch_diff / 2, 2), label="Arch Support")
        plan["inputs"] = {"pressurePattern": kin[side]["pressurePattern"], "arch_height_mm": round(m[f"{side}_arch_height"], 1),
                          "arch_diff_vs_other_mm": round(arch_diff, 1), "heel_pressure_ratio": round(heel_ratio, 2)}
        plans[side] = plan
    ov_path = os.path.join(scan_dir, "design_overrides.json")
    if os.path.exists(ov_path):
        for side, ov in json.load(open(ov_path)).items():
            plans[side].update(ov)
            plans[side]["overrides"] = ov
    return plans


# ── Closed-curve helpers ───────────────────────────────────────────────────────
def resample_closed(coords, step):
    ring = sg.LinearRing(coords)
    n = max(int(round(ring.length / step)), 16)
    return shapely.get_coordinates(shapely.line_interpolate_point(ring, np.linspace(0, ring.length, n, endpoint=False)))


OUTLINE_SMOOTH_ITERS = 600   # low-pass length ~ sqrt(600 * 0.5) * OUTLINE_STEP; moves the outline < 0.3 mm


def taubin(P, iters, lam=0.5, mu=-0.5):
    """Low-pass smoothing of a closed polyline. mu = -lam: transfer 1 - (lam*k)^2 <= 1, so no shrink and no
    passband gain (a Taubin mu slightly larger than lam amplifies ~8 mm wobble instead of removing it)."""
    for _ in range(iters):
        for f in (lam, mu):
            P = P + f * (0.5 * (np.roll(P, 1, 0) + np.roll(P, -1, 0)) - P)
    return P


def smooth_polygon(poly, iters=OUTLINE_SMOOTH_ITERS):
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda p: p.area)
    ring = taubin(resample_closed(poly.exterior.coords, OUTLINE_STEP), iters)
    return orient(sg.Polygon(ring), 1.0)


# ── Scan -> plantar surface -> foot outline ────────────────────────────────────
def plantar_surface(mesh, res=2.0):
    """Lowest scan Z above each (x, y) grid cell (NaN where no foot). Unclipped."""
    b = mesh.bounds
    xs, ys = np.arange(b[0, 0] - 1, b[1, 0] + 1, res), np.arange(b[0, 1] - 1, b[1, 1] + 1, res)
    xx, yy = np.meshgrid(xs, ys)
    origins = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, -10.0)])
    loc, ray, _ = mesh.ray.intersects_location(origins, np.tile([0, 0, 1.0], (xx.size, 1)))
    z = np.full(xx.size, np.inf)
    np.minimum.at(z, ray, loc[:, 2])
    z[np.isinf(z)] = np.nan
    return xs, ys, z.reshape(xx.shape)


def alpha_outline(xs, ys, z, alpha=0.02):
    iy, ix = np.where(~np.isnan(z))
    pts = np.column_stack([xs[ix], ys[iy]])
    tri = Delaunay(pts)
    edges = set()
    for s in tri.simplices:
        p = pts[s]
        a, b, c = (np.linalg.norm(p[i] - p[(i + 1) % 3]) for i in range(3))
        sp = (a + b + c) / 2
        area = max(sp * (sp - a) * (sp - b) * (sp - c), 1e-10) ** 0.5
        if a * b * c / (4 * area) < 1 / alpha:
            for i in range(3):
                edges.symmetric_difference_update({tuple(sorted((s[i], s[(i + 1) % 3])))})
    polys = list(polygonize(sg.MultiLineString([(tuple(pts[a]), tuple(pts[b])) for a, b in edges])))
    outline = max(polys, key=lambda p: p.area)
    return outline.buffer(4.0, join_style=1).buffer(-4.0, join_style=1).buffer(INSET, join_style=1)


def medial_is_positive_x(xs, ys, z, length):
    """Medial side = the side with the larger plantar gap (arch) in the midfoot band."""
    zb = z[(ys > 0.2 * length) & (ys < 0.5 * length)]
    xc = np.nanmean(np.where(np.isnan(zb), np.nan, xs[None, :]))
    return np.nanmean(zb[:, xs > xc]) > np.nanmean(zb[:, xs <= xc])


def plantar_gap_field(xs, ys, z_raw):
    """Gap (mm) between sole and ground as a function (x, y). Sides above PLANTAR_MAX count as PLANTAR_MAX;
    outside the scanned footprint the gap fades to 0 over EXTRAP_MM."""
    hit = ~np.isnan(z_raw)
    z = np.where(hit, np.minimum(z_raw, PLANTAR_MAX), 0.0)
    dist, (iy, ix) = distance_transform_edt(~hit, return_indices=True)
    z = np.where(hit, z, z[iy, ix] * np.clip(1 - dist * (xs[1] - xs[0]) / EXTRAP_MM, 0, 1))
    rgi = RegularGridInterpolator((ys, xs), z, bounds_error=False, fill_value=0.0)
    return lambda x, y: rgi(np.column_stack([y.ravel(), x.ravel()])).reshape(x.shape)


def load_shoe_outline(scan_dir, side):
    path = os.path.join(scan_dir, f"shoe_outline_{side}.json")
    if not os.path.exists(path):
        return None
    poly = sg.Polygon(json.load(open(path))["points_mm"]).buffer(0).buffer(-SHOE_CLEARANCE, resolution=32)
    return smooth_polygon(poly)


# ── Height field ───────────────────────────────────────────────────────────────
def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def floor_z(y):
    """Floor thickness at Y: HEEL_BASE at the heel, FOREFOOT_BASE past BASE_BLEND, smooth between."""
    return FOREFOOT_BASE + (HEEL_BASE - FOREFOOT_BASE) * (1 - smoothstep((y - BASE_BLEND[0]) / (BASE_BLEND[1] - BASE_BLEND[0])))


def row_extents(outline, step=1.0):
    b = outline.bounds
    ys = np.arange(b[1], b[3] + step, step)
    lo, hi = np.full(len(ys), np.nan), np.full(len(ys), np.nan)
    for i, y in enumerate(ys):
        inter = outline.intersection(sg.LineString([(b[0] - 5, y), (b[2] + 5, y)]))
        if inter.is_empty:
            continue
        xs = np.array([c for g in getattr(inter, "geoms", [inter]) for c in g.coords])[:, 0]
        lo[i], hi[i] = xs.min(), xs.max()
    ok = ~np.isnan(lo)
    return ys[ok], lo[ok], hi[ok]


def masked_smooth(z, mask, sigma_cells):
    num, den = gaussian_filter(np.where(mask, z, 0), sigma_cells), gaussian_filter(mask.astype(float), sigma_cells)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), FOREFOOT_BASE)


class Field:
    """Regular-grid quantities over the insole outline (plus padding)."""

    def __init__(self, outline, foot_outline, gap_fn, plan, medial_pos, pad=6.0):
        b = outline.bounds
        self.gx = np.arange(b[0] - pad, b[2] + pad, FIELD_RES)
        self.gy = np.arange(b[1] - pad, b[3] + pad, FIELD_RES)
        X, Y = np.meshgrid(self.gx, self.gy)
        P = np.column_stack([X.ravel(), Y.ravel()])
        inside = shapely.contains_xy(outline, P[:, 0], P[:, 1]).reshape(X.shape)

        # medial fraction across the insole at each row (0 lateral .. 1 medial)
        ys, lo_, hi_ = row_extents(outline)
        lo, hi = np.interp(Y, ys, lo_), np.interp(Y, ys, hi_)
        ml = np.clip((X - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
        self.ml = ml if medial_pos else 1 - ml
        self.X, self.Y, self.inside, self.foot_outline = X, Y, inside, foot_outline

        # arch: fraction of plantar gap filled, medial-biased, plus medial correction
        y0, y1, y2, y3 = ARCH_Y
        yf = smoothstep((Y - y0) / (y1 - y0)) * (1 - smoothstep((Y - y2) / (y3 - y2)))
        gap = gap_fn(X, Y)
        gap = np.where(gap > 0.3, gap, 0.0)
        fill = FILL_LATERAL + (plan["fill_medial"] - FILL_LATERAL) * self.ml
        arch = (gap * fill + plan["correction_mm"] * self.ml * np.clip(gap / 5, 0, 1)) * yf

        # heel cup: wall anchored to the FOOT outline; outside it a rounded outer face down to the flange
        Pin = P[inside.ravel()]
        dd = shapely.distance(foot_outline.exterior, shapely.points(Pin))
        in_foot = shapely.contains_xy(foot_outline, Pin[:, 0], Pin[:, 1])
        d_signed = np.zeros(X.shape)
        d_signed[inside] = np.where(in_foot, dd, -dd)
        side = smoothstep((self.ml - CUP_SIDE_ML[0]) / (CUP_SIDE_ML[1] - CUP_SIDE_ML[0]))   # 0 lateral .. 1 medial
        cup_h = CUP_H_LAT + (CUP_H - CUP_H_LAT) * side
        cup_len = CUP_LEN_LAT + (CUP_LEN - CUP_LEN_LAT) * side
        yc = 1 - smoothstep((Y - cup_len) / CUP_TAPER)
        cup = cup_h * yc * np.where(d_signed >= 0, smoothstep(1 - d_signed / CUP_WALL), 1 - smoothstep(-d_signed / CUP_OUT))

        top = floor_z(Y) + np.maximum(arch, cup)
        self.top = masked_smooth(top, inside, SMOOTH_SIGMA / FIELD_RES)
        self._rgi = RegularGridInterpolator((self.gy, self.gx), self.top, bounds_error=False, fill_value=FOREFOOT_BASE)
        # underside: the shoe's footbed rises under the medial arch, so the bottom follows it (pocket) instead of
        # bridging it flat; the top stays put, leaving a floor_z-thick shell over the bed.
        med = smoothstep((self.ml - POCKET_ML[0]) / (POCKET_ML[1] - POCKET_ML[0]))
        bed = masked_smooth(SHOE_BED_FRACTION * arch * med, inside, BED_SIGMA / FIELD_RES)
        self.bottom = np.clip(np.minimum(bed, self.top - floor_z(Y)), 0, None)
        self._rgi_bot = RegularGridInterpolator((self.gy, self.gx), self.bottom, bounds_error=False, fill_value=0.0)

    def sample(self, pts):
        return self._rgi(np.column_stack([pts[:, 1], pts[:, 0]]))

    def sample_bottom(self, pts):
        return self._rgi_bot(np.column_stack([pts[:, 1], pts[:, 0]]))

    def sample_ml(self, pts):
        return RegularGridInterpolator((self.gy, self.gx), self.ml, bounds_error=False, fill_value=0.5)(np.column_stack([pts[:, 1], pts[:, 0]]))

    # -- infill zone weights (1 at zone core, 0 outside) --
    def arch_weight(self):
        a = ARCH_ZONE
        return (smoothstep((self.Y - a["y_up"][0]) / (a["y_up"][1] - a["y_up"][0]))
                * (1 - smoothstep((self.Y - a["y_down"][0]) / (a["y_down"][1] - a["y_down"][0])))
                * smoothstep((self.ml - a["ml"][0]) / (a["ml"][1] - a["ml"][0])))

    def heel_weight(self, xc):
        h = HEEL_ZONE
        r = np.hypot(self.X - xc, self.Y - h["y"])
        return 1 - smoothstep((r - h["r_full"]) / (h["r_zero"] - h["r_full"]))


# ── Constrained triangulation + solid ──────────────────────────────────────────
def triangulate(outline):
    ring = np.array(outline.exterior.coords)[:-1]
    n = len(ring)
    seg = np.column_stack([np.arange(n), (np.arange(n) + 1) % n])
    t = triangle.triangulate({"vertices": ring, "segments": seg}, f"pq25Ya{TRI_AREA}")
    pts, tris = t["vertices"], t["triangles"]
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    flip = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]) < 0
    tris[flip] = tris[flip][:, ::-1]
    return pts, tris, n


def solid_from_heightfield(pts, tris, top_z, bottom_z, n_boundary):
    """Top + bottom surfaces sharing the constrained boundary ring (vertices 0..n_boundary-1)."""
    n = len(pts)
    verts = np.vstack([np.column_stack([pts, top_z]), np.column_stack([pts, bottom_z])])
    side = []
    for i in range(n_boundary):
        a, b = i, (i + 1) % n_boundary
        side += [[a, b, b + n], [a, b + n, a + n]]
    mesh = trimesh.Trimesh(verts, np.vstack([tris, tris[:, ::-1] + n, side]))
    mesh.fix_normals()
    return mesh


def build_insole(scan_path, plan, shoe_outline):
    scan = trimesh.load(scan_path)
    xs, ys, z_raw = plantar_surface(scan)
    z_clipped = np.where(z_raw <= PLANTAR_MAX, z_raw, np.nan)
    foot_outline = smooth_polygon(alpha_outline(xs, ys, z_clipped))
    # Never smaller than the foot-fitted outline that already fit; grows to the shoe where the shoe is bigger.
    outline = smooth_polygon(foot_outline.union(shoe_outline)) if shoe_outline is not None else foot_outline
    medial_pos = bool(medial_is_positive_x(xs, ys, z_clipped, scan.bounds[1, 1]))
    field = Field(outline, foot_outline, plantar_gap_field(xs, ys, z_raw), plan, medial_pos)

    pts, tris, nb = triangulate(outline)
    top = field.sample(pts)
    pocket = field.sample_bottom(pts)
    d = shapely.distance(outline.exterior, shapely.points(pts))
    drop = np.where(d < EDGE_R, EDGE_R - np.sqrt(np.maximum(EDGE_R ** 2 - (EDGE_R - d) ** 2, 0)), 0.0)
    top = np.maximum.reduce([floor_z(pts[:, 1]), top - drop, pocket + MIN_EDGE])
    # underside edge bevel around the heel and along the lateral side, where the footbed curls up to the sidewall
    # (factory insole's underside is rolled there); 45 deg so it prints without support. Fades out toward the ball.
    y, ml = pts[:, 1], field.sample_ml(pts)
    y_end = BEVEL_Y[1] + (BEVEL_Y[2] - BEVEL_Y[1]) * (1 - ml)
    bevel = BOTTOM_BEVEL * (1 - smoothstep((y - BEVEL_Y[0]) / (y_end - BEVEL_Y[0])))
    # cap the bevel by the edge thickness at the nearest boundary vertex: the 45 deg face then ends on the side wall
    # instead of flattening into a horizontal (unsupported) ledge where the edge is thin
    near_b = cKDTree(pts[:nb]).query(pts)[1]
    bevel = np.minimum(bevel, np.clip(top[near_b] - MIN_EDGE, 0, None))
    rise = np.minimum(np.clip(bevel - d, 0, None), top - MIN_EDGE)
    bottom = np.maximum(pocket, rise)
    mesh = solid_from_heightfield(pts, tris, top, bottom, nb)
    if not (mesh.is_watertight and mesh.volume > 0):
        raise RuntimeError(f"{scan_path}: insole mesh not watertight/positive volume")
    return mesh, medial_pos, field, outline


# ── Infill zones (graded, nested rings) ────────────────────────────────────────
def nested_zones(field, weight, outline, densities, name, zlo, zhi, exclude=None):
    """Ring k = {w >= t_k} minus {w >= t_k+1}; densities[k] ascending toward the core.
    Rings don't overlap each other or `exclude`, so the result doesn't depend on modifier ordering."""
    K = len(densities)
    w = weight.copy()
    w[0, :] = w[-1, :] = w[:, 0] = w[:, -1] = 0.0
    regions = []
    for k in range(K):
        polys = [sg.Polygon(np.column_stack([field.gx[0] + c[:, 1] * FIELD_RES, field.gy[0] + c[:, 0] * FIELD_RES])).buffer(0)
                 for c in find_contours(w, (k + 0.5) / K) if len(c) >= 4]
        regions.append(unary_union(polys).simplify(0.15) if polys else sg.Polygon())
    limit = outline.buffer(3.0)
    mods = []
    for k in range(K):
        ring = regions[k] if k == K - 1 else regions[k].difference(regions[k + 1])
        ring = ring.intersection(limit)
        if exclude is not None:
            ring = ring.difference(exclude)
        parts = [g for g in getattr(ring, "geoms", [ring]) if g.geom_type == "Polygon" and g.area > 4.0]
        min_w = MIN_BAND_SPACINGS * INFILL_LINE_W / (densities[k] / 100)
        keep = [g for g in parts if 2 * g.area / g.length >= min_w]          # 2A/P ~ band width
        lost = sum(g.area for g in parts) - sum(g.area for g in keep)
        if parts and lost > 0.25 * sum(g.area for g in parts):
            raise RuntimeError(f"{name} {k + 1} ({densities[k]}%): {lost:.0f} mm² narrower than {min_w:.1f} mm; widen the zone ramps")
        if not keep:
            continue
        mods.append(extruded_mod(keep, zlo, zhi, f"{name} {k + 1}", densities[k]))
    return mods


def extruded_mod(polys, zlo, zhi, name, density):
    mesh = trimesh.util.concatenate([trimesh.creation.extrude_polygon(g, zhi - zlo) for g in polys])
    mesh.apply_translation([0, 0, zlo])
    return {"name": name, "mesh": mesh, "density": f"{density}%", "footprint": unary_union(polys)}


def rim_zones(field, outline, plan, medial_pos, zlo, zhi):
    """Heel-cup walls (inside CUP_WALL of the foot outline, plus the flange outside it) up to RIM_Y_END, split at the
    row midline into a medial and a lateral part."""
    b = outline.bounds
    band = outline.difference(field.foot_outline.buffer(-CUP_WALL)).intersection(sg.box(b[0] - 1, b[1] - 1, b[2] + 1, RIM_Y_END))
    ys, lo, hi = row_extents(outline)
    ys, mid = ys[ys <= RIM_Y_END + 2], ((lo + hi) / 2)[ys <= RIM_Y_END + 2]
    far = b[2] + 10 if medial_pos else b[0] - 10
    medial_half = sg.Polygon([(far, b[1] - 10), *zip(mid, ys), (mid[-1], RIM_Y_END + 10), (far, RIM_Y_END + 10)]).buffer(0)
    mods = []
    for name, region, dens in (("Heel Rim Medial", band.intersection(medial_half), plan["rim_density"][0]),
                               ("Heel Rim Lateral", band.difference(medial_half), plan["rim_density"][1])):
        parts = [g for g in getattr(region, "geoms", [region]) if g.geom_type == "Polygon" and g.area > 4.0]
        if parts and dens is not None:
            mods.append(extruded_mod(parts, zlo, zhi, name, dens))
    return mods


def modifiers_for(mesh, field, outline, plan, medial_pos):
    """Rim first, then heel cushion, then arch; each later zone excludes the earlier ones so no modifiers overlap."""
    zlo, zhi = -1.0, mesh.bounds[1, 2] + 2
    mods = rim_zones(field, outline, plan, medial_pos, zlo, zhi) if plan["rim_density"] else []
    taken = lambda: unary_union([m["footprint"] for m in mods]) if mods else None
    if plan["heel_zone_peak"]:
        peak = plan["heel_zone_peak"]
        dens = [round(BASE_DENSITY + (peak - BASE_DENSITY) * (k + 1) / HEEL_ZONE_LEVELS) for k in range(HEEL_ZONE_LEVELS)]
        ys, lo, hi = row_extents(outline)
        xc = float(np.interp(HEEL_ZONE["y"], ys, (lo + hi) / 2))   # heel centre, not the bbox centre (forefoot skews it)
        mods += nested_zones(field, field.heel_weight(xc), outline, dens, "Heel Cushion Zone", zlo, zhi, taken())
    if plan["arch_zone_peak"]:
        peak = plan["arch_zone_peak"]
        dens = [round(BASE_DENSITY + (peak - BASE_DENSITY) * (k + 1) / ZONE_LEVELS) for k in range(ZONE_LEVELS)]
        mods += nested_zones(field, field.arch_weight(), outline, dens, "Medial Arch Zone", zlo, zhi, taken())
    return mods


def save_preview(meshes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    for ax, (name, m) in zip(axes, meshes.items()):
        tv = m.vertices[m.vertices[:, 2] > 0.01]
        b = m.bounds
        gx, gy = np.meshgrid(np.arange(b[0, 0], b[1, 0], 0.5), np.arange(b[0, 1], b[1, 1], 0.5))
        g = griddata(tv[:, :2], tv[:, 2], (gx, gy), method="linear")
        rgb = LightSource(315, 35).shade(np.nan_to_num(g), cmap=plt.cm.gray, vert_exag=3, blend_mode="overlay")
        ax.imshow(rgb, origin="lower", extent=[b[0, 0], b[1, 0], b[0, 1], b[1, 1]])
        ax.set_title(f"{name}  (max {m.bounds[1, 2]:.1f} mm)")
    plt.tight_layout()
    plt.savefig(path, dpi=80)


def next_version_dir(out_root, label):
    os.makedirs(out_root, exist_ok=True)
    nums = [int(m.group(1)) for d in os.listdir(out_root)
            if os.path.isdir(os.path.join(out_root, d)) and (m := re.fullmatch(r"v(\d+)(_.*)?", d))]
    path = os.path.join(out_root, f"v{max(nums, default=0) + 1:02d}" + (f"_{label}" if label else ""))
    os.makedirs(path)
    return path


def snapshot_inputs(scan_dir, out_dir):
    """Copy the design inputs and the generator code used, so a version can be rebuilt or diffed later."""
    dst = os.path.join(out_dir, "inputs")
    os.makedirs(dst)
    for f in os.listdir(scan_dir):
        if f.startswith("shoe_outline_") and f.endswith(".json") or f in ("design_overrides.json", "shoe_profile.json"):
            shutil.copy2(os.path.join(scan_dir, f), dst)
    here = os.path.dirname(os.path.abspath(__file__))
    for f in ("insole_pipeline.py", "generate_clean_bambu_project.py"):
        shutil.copy2(os.path.join(here, f), dst)


def parameters():
    """Module design constants (UPPER_CASE, JSON-serializable) for the version record."""
    return {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, tuple, dict, str))}


SHOE_PROFILE_KEYS = ("SHOE_BED_FRACTION", "BOTTOM_BEVEL", "BEVEL_Y", "POCKET_ML")   # shoe-specific; set per scan, not globally


def load_shoe_profile(scan_dir):
    """Apply <scan_dir>/shoe_profile.json (keys in SHOE_PROFILE_KEYS) over the module defaults (flat underside)."""
    path = os.path.join(scan_dir, "shoe_profile.json")
    if not os.path.exists(path):
        return {}
    prof = json.load(open(path))
    bad = set(prof) - set(SHOE_PROFILE_KEYS)
    if bad:
        raise SystemExit(f"{path}: unknown keys {sorted(bad)}; allowed {SHOE_PROFILE_KEYS}")
    globals().update({k: tuple(v) if isinstance(v, list) else v for k, v in prof.items()})
    return prof


def main(scan_dir, out_root, label=""):
    out_dir = next_version_dir(out_root, label)
    snapshot_inputs(scan_dir, out_dir)
    load_shoe_profile(scan_dir)
    plans = analyze(scan_dir)
    feet, meshes, pocket = [], {}, 0.0
    for side in ("left", "right"):
        plan = plans[side]
        shoe = load_shoe_outline(scan_dir, side)
        mesh, medial_pos, field, outline = build_insole(os.path.join(scan_dir, f"{side}.stl"), plan, shoe)
        mesh.export(os.path.join(out_dir, f"{side}_insole.stl"))
        mods = modifiers_for(mesh, field, outline, plan, medial_pos)
        for i, mod in enumerate(mods):
            mod["mesh"].export(os.path.join(out_dir, f"{side}_zone_{i + 1:02d}_{mod['density'].rstrip('%')}pct.stl"))
        pocket = max(pocket, float(field.bottom.max()))
        plan["medial_side_x"] = "+" if medial_pos else "-"
        plan["sized_to_shoe_outline"] = shoe is not None
        plan["result"] = {"size_mm": (mesh.bounds[1] - mesh.bounds[0]).round(1).tolist(), "volume_cm3": round(mesh.volume / 1000, 1),
                          "arch_pocket_max_mm": round(float(field.bottom.max()), 1),
                          "zones": [f"{m['name']}: {m['density']}" for m in mods]}
        feet.append({"label": f"{side.capitalize()} Insole ({plan['label']})", "mesh": mesh, "modifiers": mods})
        meshes[side] = mesh
    process = {"sparse_infill_density": f"{BASE_DENSITY}%", "top_shell_layers": str(TOP_LAYERS), "bottom_shell_layers": str(BOTTOM_LAYERS)}
    if pocket > 0.3:
        # the arch pocket's ceiling is an overhang: tree supports from the plate only, extra Z gap so TPU peels off
        process.update({"enable_support": "1", "support_type": "tree(auto)", "support_on_build_plate_only": "1",
                        "support_top_z_distance": "0.3"})
    write_3mf(feet, os.path.join(out_dir, "Custom_Insoles_ME3D.3mf"), process)
    # default infill test plate at the forefoot floor thickness (where 94% of the insole sits), same shell layers as the insoles
    write_coupon_plate(out_dir, thicknesses=(FOREFOOT_BASE,), top_layers=TOP_LAYERS, bottom_layers=BOTTOM_LAYERS, base_density=BASE_DENSITY)
    save_preview(meshes, os.path.join(out_dir, "preview.png"))
    json.dump({"version": os.path.basename(out_dir), "parameters": parameters(), "process": process, "plans": plans},
              open(os.path.join(out_dir, "design.json"), "w"), indent=2)
    print(out_dir)
    print(json.dumps(plans, indent=2))
    return out_dir


if __name__ == "__main__":
    main(*sys.argv[1:4])
