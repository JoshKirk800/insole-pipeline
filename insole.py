# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 JoshKirk800
"""Fleet Feet scan link -> custom insoles, in the slicer format you pick.

    python insole.py [LINK] [--workdir DIR] [--format bambu|orca|prusa|stl|all] [--label NAME] [--yes]

Asks for your Fleet Feet fit id link (or the scan id), downloads the scan, builds the insoles and the infill test
plate, then asks which format you want. Everything for one scan lives in <workdir>/<first 8 chars of the id>/:
    scan/      downloaded scan data (private: biometric data)        designs/  generated versions (vNN_<label>)
    export/    the chosen format(s)                                  run.log   full output of the steps
Optional inputs copied into scan/ before building: --shoe-outline-left/-right (factory insole outlines made by
shoe_outline_from_photo.py), --shoe-profile (shoe underside values), --overrides (per-foot design overrides).
Without them the insole is sized to the foot and has a flat underside.
"""
import argparse
import contextlib
import json
import os
import shutil
import sys

import acquire_scan
import exporters

SHOE_FILES = {"shoe_outline_left": "shoe_outline_left.json", "shoe_outline_right": "shoe_outline_right.json",
              "shoe_profile": "shoe_profile.json", "overrides": "design_overrides.json"}
CONSOLE_HELP = """The Fleet Feet page removes the scan id from the address bar when it loads. Get it one of two ways:
  - in your results email, right-click "View 3D Scan" > Copy link address (the link contains &scan=<id>), or
  - on the scan page open the browser console (F12) and run:
    (() => { const id = sessionStorage.getItem('ff_fitid_scan') || document.querySelector('iframe.scan-frame')?.src.match(/volumental\\.com\\/([0-9a-f-]+)/i)?.[1] || location.pathname.split('/')[1]; console.log(id); copy(id); })()"""


def ask(prompt, default=None):
    """input() that returns `default` at end of input (non-interactive use)."""
    try:
        answer = input(prompt).strip()
    except EOFError:
        return default
    return answer or default


@contextlib.contextmanager
def logged(path, title):
    """Run a step with its output going to the log file; on failure print the tail of the log."""
    print(f"  {title} ...", end=" ", flush=True)
    with open(path, "a", encoding="utf-8") as log:
        log.write(f"\n===== {title} =====\n")
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                yield
            print("done")
        except SystemExit as e:
            print("FAILED")
            _tail(path)
            if e.code not in (0, None):
                raise
        except Exception as e:
            print(f"FAILED ({type(e).__name__}: {e})")
            _tail(path)
            raise


def _tail(path, n=12):
    lines = open(path, encoding="utf-8").read().splitlines()[-n:]
    print("    " + "\n    ".join(lines))


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("link", nargs="?", help="Fleet Feet fit id link, my.volumental.com link, or the scan id (asked if omitted)")
    ap.add_argument("--workdir", default="insole_work")
    ap.add_argument("--format", help="bambu, orca, prusa, stl, a comma list, or all (asked if omitted)")
    ap.add_argument("--label", help="version label (default scan-<id8>)")
    ap.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    ap.add_argument("--no-pressure-maps", action="store_true")
    for opt, fname in SHOE_FILES.items():
        ap.add_argument("--" + opt.replace("_", "-"), metavar="FILE", help=f"copied to scan/{fname}")
    return ap.parse_args(argv)


def get_scan_id(link):
    for attempt in range(3):
        if link is None:
            link = ask("Paste your Fleet Feet fit id link (or the scan id): ")
        sid = acquire_scan.find_scan_id(link) if link else None
        if sid:
            return sid
        print(f"No scan id found in that link.\n{CONSOLE_HELP}\n")
        link = None
    raise SystemExit("no scan id; giving up")


def describe_scan(scan_dir):
    m = json.load(open(os.path.join(scan_dir, "measurements.json")))["measurements"]
    kin = json.load(open(os.path.join(scan_dir, "kinetic_profile.json")))
    for side in ("left", "right"):
        print(f"    {side:5s} {m[side + '_length'] * 1000:.0f} mm long, {m[side + '_width'] * 1000:.0f} mm wide, "
              f"arch {m[side + '_arch_height'] * 1000:.1f} mm, pressure pattern: {kin[side]['pressurePattern']}")


def choose_formats(arg):
    names = list(exporters.FORMATS)
    if arg:
        picked = names if arg == "all" else [a.strip() for a in arg.split(",") if a.strip()]
        bad = [p for p in picked if p not in exporters.FORMATS]
        if bad:
            raise SystemExit(f"unknown format {bad}; choose from {names + ['all']}")
        return picked
    print("\nWhich format do you want?")
    for i, n in enumerate(names, 1):
        print(f"  {i}) {exporters.FORMATS[n][0]:26s} {exporters.FORMATS[n][1]}")
    print(f"  {len(names) + 1}) All of the above")
    while True:
        a = ask(f"Choice [1-{len(names) + 1}, default 1]: ", "1")
        if a.isdigit() and 1 <= int(a) <= len(names) + 1:
            return names if int(a) == len(names) + 1 else [names[int(a) - 1]]
        if a in names:
            return [a]
        print("  not a valid choice")


NEXT_STEP = {
    "bambu": "open Custom_Insoles_ME3D.3mf in Bambu Studio (plate 1 = left, plate 2 = right); test_coupons.3mf is the infill test plate",
    "orca": "open Custom_Insoles_ME3D.3mf in OrcaSlicer and check the zones show their infill and the printer/filament presets match",
    "prusa": "open each file in PrusaSlicer (bed is set to 256 x 256 in the file: switch to your printer preset and check the arrangement)",
    "stl": "follow slicer_settings.md; zone_map.png shows where each zone STL goes",
}


def main(argv=None):
    args = parse_args(argv)
    print("Insole pipeline: Fleet Feet scan -> custom insoles\n")
    sid = get_scan_id(args.link)
    work = os.path.join(args.workdir, sid[:8])
    scan_dir, designs, export_root, log = (os.path.join(work, d) for d in ("scan", "designs", "export", "run.log"))
    os.makedirs(scan_dir, exist_ok=True)
    print(f"Scan {sid}\nWorking folder: {os.path.abspath(work)}\n")

    for opt, fname in SHOE_FILES.items():
        src = getattr(args, opt)
        if src:
            shutil.copy2(src, os.path.join(scan_dir, fname))
            print(f"  using {src} as {fname}")

    print("Steps:")
    with logged(log, "downloading the scan"):
        acquire_scan.acquire(sid, scan_dir, args.force)
    describe_scan(scan_dir)
    if not args.yes and (ask("Build insoles from this scan? [Y/n] ", "y") or "y").lower().startswith("n"):
        print("Stopped; the scan is saved in", scan_dir)
        return 0

    if not args.no_pressure_maps:
        import pressure_maps
        try:
            with logged(log, "pressure maps"):
                pressure_maps.main(scan_dir)
        except (Exception, SystemExit):
            print("  (continuing without pressure maps)")

    import insole_pipeline
    label = args.label or f"scan-{sid[:8]}"
    with logged(log, "building the insoles and the test plate"):
        version_dir = insole_pipeline.main(scan_dir, designs, label)
    design = exporters.load_design(version_dir)
    print(f"\nDesign {os.path.basename(version_dir)}:")
    for side in ("left", "right"):
        plan = design["plans"][side]
        res = plan["result"]
        print(f"    {side:5s} {plan['label']:12s} {res['size_mm'][0]:.0f} x {res['size_mm'][1]:.0f} x {res['size_mm'][2]:.0f} mm, {len(res['zones'])} infill zones")
    if not os.path.exists(os.path.join(scan_dir, SHOE_FILES["shoe_outline_left"])):
        print("    (sized to the foot, flat underside: no factory-insole outline or shoe profile given)")

    out = {}
    for fmt in choose_formats(args.format):
        out[fmt] = exporters.export(fmt, version_dir, os.path.join(export_root, fmt))
    print("\nDone. Files:")
    for fmt, files in out.items():
        print(f"  {exporters.FORMATS[fmt][0]}  ->  {os.path.abspath(os.path.join(export_root, fmt))}")
        print(f"    next: {NEXT_STEP[fmt]}")
    print(f"\nFull log: {os.path.abspath(log)}\nThe scan folder holds biometric data: keep it private.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
