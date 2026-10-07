# Custom insole pipeline

Volumental/Fleet Feet foot scan -> printable TPU insoles (STL) + Bambu Studio project (3MF) for a Bambu Lab A1.

```
pip install -r requirements.txt
python acquire_scan.py <scan id> foot_scan_data        # once per person: scan data + pressure steps
python pressure_maps.py foot_scan_data                 # optional: where the load is
python insole_pipeline.py foot_scan_data generated_insoles <label>
```

Each run writes a new folder `generated_insoles/vNN_<label>/`:

| file | what |
|---|---|
| `Custom_Insoles_ME3D.3mf` | two plates (left, right), 45 deg rotated to fit the 256 mm bed; modifiers + process settings embedded |
| `left_insole.stl`, `right_insole.stl` | insole bodies |
| `<side>_zone_NN_<pct>pct.stl` | infill-density modifier volumes (also embedded in the 3MF) |
| `design.json` | every design constant, slicer setting and per-foot plan for that version |
| `preview.png`, `zones_vs_pressure.png` | top-surface render |
| `test_coupons.3mf`, `.json`, `.png` | default infill test plate (see Infill test coupons) |
| `inputs/` | shoe outlines, overrides and the code used, so a version can be rebuilt |

Scan data and generated designs are git-ignored (not in the repository). STL/3MF output can be rebuilt from `inputs/`;
`design.json` records every setting of a version.

## Acquire a scan

```
python acquire_scan.py <scan id or URL containing it> foot_scan_data_<name>
```

Downloads the meshes (`left/right.stl`, `.obj`), `measurements.json`, the gait/pressure summaries and every walking
step's pressure frames from my.volumental.com, validates them, and writes `scan_info.json` (scan id, time, size and
sha256 of each file). Existing files are kept unless `--force`; missing required files exit with code 1.
The module docstring has the Chrome console one-liner that reads the scan id off the Fleet Feet fit id page.

## Inputs (`foot_scan_data/`)

Required: `left.stl`, `right.stl` (mm, Y=0 heel), `measurements.json`, `kinetic_profile.json`,
`running_pressure_measurement.json`, `pressure_measurement.json`.

Optional:
- `shoe_outline_<side>.json` - factory-insole outline in the insole frame. The insole is sized to it minus
  `SHOE_CLEARANCE`. Made by `shoe_outline_from_photo.py`.
- `design_overrides.json` - per-foot plan keys (`fill_medial`, `correction_mm`, `arch_zone_peak`, ...) applied last.
- `shoe_profile.json` - shoe-specific underside geometry (`SHOE_BED_FRACTION` arch pocket depth, `BOTTOM_BEVEL` heel/lateral
  edge bevel, optionally `BEVEL_Y`, `POCKET_ML`). Absent = flat underside. `examples/shoe_profile_altra_fwd_via_2.json`
  has the values used for the Altra FWD VIA 2 (pocket 1.0, bevel 4 mm); another shoe needs its own measurements.
  `examples/design_overrides.example.json` shows the override format.

One folder per person (`foot_scan_data`, `foot_scan_data_2`, ...), each built into its own output root, e.g.
`python insole_pipeline.py foot_scan_data_2 generated_insoles_2 <label>`.

## Tuning

All design constants are at the top of `insole_pipeline.py` (floor thickness, heel cup, arch, pocket, bevel, zones,
densities, shell layers). Change one thing per print, rebuild, and compare `design.json` between versions.

## Shoe outline from a photo

```
python shoe_outline_from_photo.py <photo.jpg> <printed_insole.stl> foot_scan_data/shoe_outline_<side>.json
```

Photo: top view of the printed insole lying on its factory insole on a light-teal cutting mat. The printed STL is the
scale/pose reference (silhouette is registered, IoU must be >= 0.95). If the output file exists, the photo only adds
places where the factory insole sticks out past the print.

Each add-on is recorded in the outline json (`addons`). To give the other foot the same edge (the two factory insoles
are mirror images), copy it instead of photographing again:

```
python shoe_outline_from_photo.py mirror foot_scan_data/shoe_outline_right.json foot_scan_data/shoe_outline_left.json
```

The source outline minus its add-ons is mirrored and registered (shift + rotation) onto the destination outline; it
must match to IoU >= 0.90 (it is 0.94 for the Altra pair, rotation -3.6 deg). The registration is written to the
destination json (`mirror_registration`). Keep a copy of the destination json first if you may want to undo it.

## Infill test coupons

```
python coupon_plate.py [out_dir] [--gyroid 10,12,15,20,28,38] [--crosszag 15,28] [--thickness 4,5] [--size 30]
```

`test_coupons.3mf` is one A1 plate of 30 mm slabs, each with its own sparse infill (modifier), the insoles' shell layers
(3 top + 2 bottom) and wall loops, and a raised label tab (`G15` = gyroid 15%, `Z28` = cross zag 28%; `/5` = 5 mm
thick when several thicknesses are printed). Default thickness is the forefoot floor (4 mm -> 3 mm sparse core); add
`--thickness 4,5` for the heel floor too (16 coupons). Every pipeline run writes the default plate into its version
folder, so the test always matches that version's shell layers. Print in the same TPU profile as the insoles, press
each coupon with a thumb or stand on it, sort soft -> firm, and compare with the densities in `design.json`.
`test_coupons.json` / `.png` give the layout.

## Pressure maps

```
python pressure_maps.py <scan_dir> [--insole-dir generated_insoles/vNN_label] [--out DIR]
```

Averages the walking-step pressure frames (`<i>.bin`, `<i>.json`, `footaxis_<i>.json`, as downloaded by
`acquire_scan.py`) into the insole frame, per foot, each step normalised to its own peak. Writes `pressure_maps.png`,
`pressure_summary.json` (regions, heel/forefoot hotspots, steps used/skipped, per-zone mean pressure and share of
load) and `pressure_maps.npz` to `--out` (default `<scan_dir>/pressure_maps`). With `--insole-dir` the insole
outline and infill zones are overlaid.

- Alignment is good to about one sensor row (~8 mm); feet with < 3 steps are flagged low confidence.
- Partial footprints (pressure crop < 150 mm long) are skipped; a stance far longer than the rest is flagged
  (`long contact`) but kept. Drop steps by hand with `--exclude 1,4`.
- A hotspot is the pressure-weighted centroid of the cells within 90% of the band maximum, so a flat plateau gives a
  stable position.
- A scan without per-step frames gets a scan-level summary and no maps.

## Design rules the generator enforces

- Infill zones never overlap (earlier zones are cut out of later ones).
- A zone part narrower than 2 infill line spacings is dropped; if more than 25% of a zone would be, the build fails.
- Meshes must be watertight with positive volume.
- Arch pocket (underside) needs support: the 3MF enables tree supports from the plate only when a pocket exists.

## Status and limits

- The design constants (floor thickness, heel cup, arch fill, zone positions, infill densities) were tuned on one
  person and one shoe (Altra FWD VIA 2) from fit feedback; treat them as a starting point. The infill densities have
  not been validated by wear - print the coupon plate first.
- Shoe fitting is a print -> photograph -> rebuild loop (`shoe_outline_from_photo.py`); the shoe underside values
  (`shoe_profile.json`) are estimates until measured.
- No automated tests yet. Developed and run on Python 3.13 / Windows.

## Privacy

Scans (3D foot meshes, gait and pressure data, scan ids) are biometric data. `foot_scan_data*/` and
`generated_insoles*/` are git-ignored; keep it that way when sharing a fork, and get a person's consent before
running the pipeline on their scan.
