# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 JoshKirk800
"""Download everything the pipeline needs for one Fleet Feet fit id / Volumental scan.

    python acquire_scan.py <scan id or any URL containing it> <out_dir> [--force]

Scan id: Fleet Feet strips `scan=` from the fit id page URL and keeps it in sessionStorage. On that page, in the
Chrome console (F12):

    (() => { const id = sessionStorage.getItem('ff_fitid_scan') || document.querySelector('iframe.scan-frame')?.src.match(/volumental\\.com\\/([0-9a-f-]+)/i)?.[1] || location.pathname.split('/')[1]; console.log(id); copy(id); })()

(sessionStorage is per tab: use the tab that opened the link. Otherwise right-click "View 3D Scan" in the results
email -> "Copy link address"; the id is the `scan=` value.)

Files written to <out_dir> (names match what insole_pipeline.py / pressure_maps.py read), all from my.volumental.com:
  /uploads/<id>/                  measurements.json*, measurement_descriptions.json, left.stl*, right.stl*, left.obj, right.obj
  /fitstation/scans/<id>/         kinetic_profile.json*, pressure_measurement.json*, running_pressure_measurement.json*,
                                  fitstation.json (raw pressure stream, ~4 MB), and per walking step i:
                                  <i>.bin/.json/.png, footaxis_<i>.json, centerofpressure_<i>.json
  /fitstation/scans/measurements/<id>   -> fitstation_summary.json
  /capture_meta/<id>, /scan_class/<id>  -> scan_info.json (with sha256 + size of every file)
(* = required: the run fails if one is missing or malformed.) Existing files are kept unless --force.
Per-step frames are optional for the insole itself; without them pressure_maps.py can only give scan-level metrics.
"""
import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from urllib.parse import unquote

HOST = "https://my.volumental.com"
UA = {"User-Agent": "Mozilla/5.0"}
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)

# (url path template, local name, required)
FILES = [
    ("/uploads/{id}/measurements.json", "measurements.json", True),
    ("/uploads/{id}/measurement_descriptions.json", "measurement_descriptions.json", False),
    ("/uploads/{id}/left.stl", "left.stl", True),
    ("/uploads/{id}/right.stl", "right.stl", True),
    ("/uploads/{id}/left.obj", "left.obj", False),
    ("/uploads/{id}/right.obj", "right.obj", False),
    ("/fitstation/scans/{id}/kinetic_profile.json", "kinetic_profile.json", True),
    ("/fitstation/scans/{id}/pressure_measurement.json", "pressure_measurement.json", True),
    ("/fitstation/scans/{id}/running_pressure_measurement.json", "running_pressure_measurement.json", True),
    ("/fitstation/scans/{id}/fitstation.json", "fitstation.json", False),
    ("/fitstation/scans/measurements/{id}", "fitstation_summary.json", False),
]


class Missing(Exception):
    pass


def find_scan_id(text):
    """Scan id from a Fleet Feet fit id link (`...&scan=<id>`), a my.volumental.com/<id>/ link, or the id itself; None
    if there is none. (The fit id page strips `scan=` from the address bar, so a link copied from the browser after
    loading usually has no id - use the email's "View 3D Scan" link or the console snippet in this file's docstring.)"""
    text = unquote(text.strip())
    m = re.search(r"[?&#]scan=([0-9a-f-]{8,64})(?![0-9a-z-])", text, re.I)    # not `scanned=...`
    if m:
        return m.group(1).lower()
    m = UUID.search(text)
    if m:
        return m.group(0).lower()
    return text.lower() if re.fullmatch(r"[0-9a-f-]{8,64}", text, re.I) else None


def parse_scan_id(text):
    sid = find_scan_id(text)
    if not sid:
        raise SystemExit(f"no scan id found in {text!r}")
    return sid


def fetch(path, retries=3):
    """GET HOST+path -> bytes. The server answers 500 'Could not fetch ...' for files that don't exist."""
    err = None
    for k in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(HOST + path, headers=UA), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (404, 500):
                body = e.read(200).decode("utf-8", "replace").strip()
                raise Missing(f"HTTP {e.code}" + (f" ({body})" if body and not body.startswith("<") else ""))
            err = e
        except (urllib.error.URLError, TimeoutError) as e:
            err = e
        time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"GET {path} failed: {err}")


def validate(name, data):
    """Raises ValueError if `data` is not what `name` should be (the site serves its SPA html for unknown paths)."""
    if not data:
        raise ValueError("empty")
    if name.endswith(".json"):
        j = json.loads(data)
        if name == "measurements.json" and "left_length" not in j.get("measurements", {}):
            raise ValueError("no measurements.left_length")
        if name == "kinetic_profile.json" and not {"left", "right"} <= set(j):
            raise ValueError("no left/right")
    elif name.endswith(".stl"):
        if len(data) < 84 or 84 + 50 * struct.unpack("<I", data[80:84])[0] != len(data):
            raise ValueError("not a binary STL (size does not match its triangle count)")
    elif name.endswith(".obj"):
        if not data.lstrip().startswith((b"#", b"v ")):
            raise ValueError("not an OBJ")
    elif name.endswith(".png"):
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("not a PNG")


def save(out_dir, name, data):
    tmp = os.path.join(out_dir, name + ".part")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, os.path.join(out_dir, name))


def acquire(scan, out_dir, force=False):
    sid = parse_scan_id(scan)
    os.makedirs(out_dir, exist_ok=True)
    got, kept, missing, problems = {}, [], [], []

    def get_file(path, name, required):
        dst = os.path.join(out_dir, name)
        if os.path.exists(dst) and not force:
            kept.append(name)
            return open(dst, "rb").read()
        try:
            data = fetch(path.format(id=sid))
            validate(name, data)
        except Missing as e:
            (problems if required else missing).append(f"{name}: {e}")
            return None
        except (ValueError, json.JSONDecodeError) as e:
            problems.append(f"{name}: invalid ({e})")
            return None
        save(out_dir, name, data)
        got[name] = data
        return data

    for path, name, required in FILES:
        get_file(path, name, required)

    # walking steps: the listing is the authority; footaxis/centerofpressure exist per step but are not always listed
    steps = []
    try:
        listing = json.loads(fetch(f"/fitstation/scans/{sid}/?steps_per_foot=equal"))
        idx = sorted({int(m.group(1)) for f in listing if (m := re.match(r"(\d+)\.", f))})
    except (Missing, ValueError, json.JSONDecodeError):
        idx = []
    for i in idx:
        meta_raw = None
        for tmpl, name in (("/fitstation/scans/{id}/%d.json" % i, f"{i}.json"), ("/fitstation/scans/{id}/%d.bin" % i, f"{i}.bin"),
                           ("/fitstation/scans/{id}/%d.png" % i, f"{i}.png"), ("/fitstation/scans/{id}/footaxis_%d.json" % i, f"footaxis_{i}.json"),
                           ("/fitstation/scans/{id}/centerofpressure_%d.json" % i, f"centerofpressure_{i}.json")):
            d = get_file(tmpl, name, False)
            if name == f"{i}.json":
                meta_raw = d
        if meta_raw:
            m = json.loads(meta_raw)
            b = os.path.join(out_dir, f"{i}.bin")
            ok = os.path.exists(b) and os.path.getsize(b) == 4 * m["nbFrames"] * m["height"] * m["width"]
            steps.append({"step": i, "foot": m["type"], "frames": m["nbFrames"], "crop_sensors": f"{m['height']}x{m['width']}",
                          "has_axis": os.path.exists(os.path.join(out_dir, f"footaxis_{i}.json")), "bin_ok": ok})

    info_path = os.path.join(out_dir, "scan_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) and not force else {}
    for key, path in (("capture_meta", f"/capture_meta/{sid}"), ("scan_class", f"/scan_class/{sid}")):
        if key not in info:
            try:
                raw = fetch(path)
                info[key] = json.loads(raw) if key == "capture_meta" else raw.decode().strip()
            except (Missing, json.JSONDecodeError):
                pass
    files = {f: {"bytes": os.path.getsize(os.path.join(out_dir, f)), "sha256": hashlib.sha256(open(os.path.join(out_dir, f), "rb").read()).hexdigest()}
             for f in sorted(os.listdir(out_dir)) if f != "scan_info.json" and os.path.isfile(os.path.join(out_dir, f))}
    info.update({"scan_id": sid, "source": HOST, "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": files,
                 "steps": steps, "not_available": missing})
    json.dump(info, open(info_path, "w"), indent=2)

    print(f"scan {sid} -> {out_dir}")
    print(f"  downloaded {len(got)} files, kept {len(kept)} existing, {len(steps)} walking steps")
    for s in steps:
        flags = ("" if s["has_axis"] else "  [no footaxis: pressure_maps skips it]") + ("" if s["bin_ok"] else "  [.bin size mismatch]")
        print(f"    step {s['step']}: {s['foot']:5s} {s['frames']:4d} frames, crop {s['crop_sensors']}{flags}")
    if not steps:
        print("  no per-step pressure frames on the server for this scan")
    for m in missing:
        print("  not available:", m)
    if problems:
        print("FAILED required files:")
        for p in problems:
            print("  ", p)
        raise SystemExit(1)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan", help="scan id (uuid) or a URL containing it")
    ap.add_argument("out_dir")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    a = ap.parse_args()
    acquire(a.scan, a.out_dir, a.force)
