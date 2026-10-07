# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 JoshKirk800
"""Bambu Studio 3MF writers.

write_3mf: two plates (left, right), each insole rotated 45 deg to fit the A1 bed.
write_plate_3mf: one unrotated plate of arbitrary objects with per-part slicer settings (used by coupon_plate.py).

write_3mf(feet, out_path): feet = [left, right]; each foot is
    {"label": str, "mesh": Trimesh,
     "modifiers": [{"name": str, "mesh": Trimesh, "density": "38%"}, ...]}
Modifier meshes share the body's coordinate system.
process: project-level slicer keys, e.g. {"sparse_infill_density": "12%", "top_shell_layers": "3", "bottom_shell_layers": "2"}.
"""
import json
import os
import uuid
import zipfile

import numpy as np

BED = 256.0
LAYER_HEIGHT, WALL_LOOPS, MAX_VOLUMETRIC_SPEED = "0.2", "2", "3.5"   # strings: Bambu config values; shared with exporters.py
PLATE_PITCH = 1.2 * BED  # Bambu Studio lays plates out 1.2x bed width apart
ROT = np.radians(45)
C, S = np.cos(ROT), np.sin(ROT)


def _mesh_xml(mesh):
    v = "\n".join(f'     <vertex x="{p[0]:.4f}" y="{p[1]:.4f}" z="{p[2]:.4f}"/>' for p in mesh.vertices)
    t = "\n".join(f'     <triangle v1="{f[0]}" v2="{f[1]}" v3="{f[2]}"/>' for f in mesh.faces)
    return f"    <mesh>\n     <vertices>\n{v}\n     </vertices>\n     <triangles>\n{t}\n     </triangles>\n    </mesh>"


def _placement(mesh, plate_index):
    """Translation that centres the 45-deg-rotated body on its plate and drops it to Z=0."""
    R = np.array([[C, -S, 0], [S, C, 0], [0, 0, 1]])
    rot = mesh.vertices @ R.T
    lo, hi = rot.min(axis=0), rot.max(axis=0)
    cx = BED / 2 + PLATE_PITCH * plate_index
    tx, ty, tz = cx - (lo[0] + hi[0]) / 2, BED / 2 - (lo[1] + hi[1]) / 2, -lo[2]
    local = np.array([lo[0] + tx - PLATE_PITCH * plate_index, lo[1] + ty, hi[0] + tx - PLATE_PITCH * plate_index, hi[1] + ty])
    if local[0] < 0 or local[1] < 0 or local[2] > BED or local[3] > BED:
        raise RuntimeError(f"Plate {plate_index + 1}: insole does not fit the {BED:.0f} mm bed: {local.round(1)}")
    return tx, ty, tz

def _package(out_path, resources, build, settings_objs, plates, asm_ids, process, title):
    """Assemble the 3MF zip: model, Bambu model_settings (objects, parts, plates) and project_settings."""
    uid = lambda: str(uuid.uuid4())
    model_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06" xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p" xmlns:BambuStudio="http://schemas.bambulab.com/package/2021">\n'
        ' <metadata name="Application">BambuStudio-02.08.02.61</metadata>\n'
        ' <metadata name="BambuStudio:3mfVersion">1</metadata>\n'
        f' <metadata name="Title">{title}</metadata>\n'
        ' <resources>\n' + "\n".join(resources) + '\n </resources>\n'
        f' <build p:UUID="{uid()}">\n' + "\n".join(build) + '\n </build>\n'
        '</model>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
        ' <Default Extension="png" ContentType="image/png"/>\n'
        ' <Default Extension="gcode" ContentType="text/x.gcode"/>\n'
        '</Types>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        ' <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
        '</Relationships>'
    )
    model_settings = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<config>\n' + "\n".join(settings_objs) + "\n" + "\n".join(plates)
        + "\n  <assemble>\n" + "".join(f'    <assemble_item object_id="{a}" instance_id="0" offset="0 0 0"/>\n' for a in asm_ids)
        + "  </assemble>\n</config>"
    )
    # Bambu Studio's validator requires: print_settings_id (not process_settings_id), scalar strings for
    # process keys, arrays for filament keys, and subtype="modifier_part".
    project_settings = {
        "different_settings_to_system": [
            "".join(f";{k}" for k in ["sparse_infill_pattern", *process]),
            ";filament_max_volumetric_speed", "", "",
        ],
        "curr_bed_type": "Textured PEI Plate",
        "filament_colour": ["#00AEFF"],
        "filament_diameter": ["1.75"],
        "filament_is_support": ["0"],
        "filament_type": ["TPU"],
        "layer_height": LAYER_HEIGHT,
        "wall_loops": WALL_LOOPS,
        "sparse_infill_pattern": "gyroid",
        **process,
        "filament_max_volumetric_speed": [MAX_VOLUMETRIC_SPEED],
        "printable_area": ["0x0", "256x0", "256x256", "0x256"],
        "printable_height": "256",
        "bed_exclude_area": [],
        "filament_settings_id": ["Bambu TPU 95A HF @BBL A1"],
        "printer_model": "Bambu Lab A1",
        "wipe_tower_x": ["15"],
        "wipe_tower_y": ["220"],
        "print_settings_id": "0.20mm Standard @BBL A1",
        "printer_settings_id": "Bambu Lab A1 0.4 nozzle",
        "printer_variant": "0.4",
        "nozzle_diameter": ["0.4"],
    }
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("3D/3dmodel.model", model_xml)
        zf.writestr("Metadata/model_settings.config", model_settings)
        zf.writestr("Metadata/project_settings.config", json.dumps(project_settings, indent=2))
    print(f"[3MF] {out_path} ({os.path.getsize(out_path)} bytes)")


def write_plate_3mf(items, out_path, process, plate_name="Plate", title="Test plate"):
    """One plate, no rotation. items: [{"name": str, "x": mm, "y": mm,
        "parts": [{"name": str, "mesh": Trimesh, "subtype": "normal_part"|"modifier_part", "settings": {slicer key: value}}]}]
    Part meshes are in the item's own frame (x, y >= 0, z >= 0); the item is translated by (x, y) on the bed."""
    uid = lambda: str(uuid.uuid4())
    resources, build, settings_objs, instances, asm_ids = [], [], [], [], []
    next_id = 1
    for item in items:
        lo = np.min([p["mesh"].bounds[0] for p in item["parts"]], axis=0)
        hi = np.max([p["mesh"].bounds[1] for p in item["parts"]], axis=0)
        if lo[0] + item["x"] < 0 or lo[1] + item["y"] < 0 or hi[0] + item["x"] > BED or hi[1] + item["y"] > BED or lo[2] < -1e-6:
            raise RuntimeError(f"{item['name']}: outside the {BED:.0f} mm bed at ({item['x']}, {item['y']})")
        part_ids = []
        for p in item["parts"]:
            resources.append(f'  <object id="{next_id}" p:UUID="{uid()}" name="{p["name"]}" type="model">\n{_mesh_xml(p["mesh"])}\n  </object>')
            part_ids.append(next_id)
            next_id += 1
        asm_id = next_id
        next_id += 1
        asm_ids.append(asm_id)
        comps = "".join(f'    <component objectid="{pid}" p:UUID="{uid()}"/>\n' for pid in part_ids)
        resources.append(f'  <object id="{asm_id}" p:UUID="{uid()}" name="{item["name"]}" type="model">\n   <components>\n{comps}   </components>\n  </object>')
        build.append(f'  <item objectid="{asm_id}" p:UUID="{uid()}" transform="1 0 0 0 1 0 0 0 1 {item["x"]:.4f} {item["y"]:.4f} 0" printable="1"/>')
        part_xml = ""
        for pid, p in zip(part_ids, item["parts"]):
            meta = "".join(f'      <metadata key="{k}" value="{v}"/>\n' for k, v in p.get("settings", {}).items())
            extruder = '      <metadata key="extruder" value="1"/>\n' if p["subtype"] == "normal_part" else ""
            part_xml += f'    <part id="{pid}" subtype="{p["subtype"]}">\n      <metadata key="name" value="{p["name"]}"/>\n{extruder}{meta}    </part>\n'
        settings_objs.append(f'  <object id="{asm_id}">\n    <metadata key="name" value="{item["name"]}"/>\n    <metadata key="extruder" value="1"/>\n{part_xml}  </object>')
        instances.append(f'    <model_instance>\n      <metadata key="object_id" value="{asm_id}"/>\n      <metadata key="instance_id" value="0"/>\n    </model_instance>')
    plates = [f'  <plate>\n    <metadata key="plater_id" value="1"/>\n    <metadata key="plater_name" value="{plate_name}"/>\n' + "\n".join(instances) + "\n  </plate>"]
    _package(out_path, resources, build, settings_objs, plates, asm_ids, process, title)



def write_3mf(feet, out_path, process):
    uid = lambda: str(uuid.uuid4())
    resources, next_id = [], 1
    parts = []  # per foot: (body_id, [(mod_id, mod)])
    for foot in feet:
        body_id = next_id
        next_id += 1
        resources.append(f'  <object id="{body_id}" p:UUID="{uid()}" name="{foot["label"]} Body" type="model">\n{_mesh_xml(foot["mesh"])}\n  </object>')
        mods = []
        for mod in foot["modifiers"]:
            mods.append((next_id, mod))
            resources.append(f'  <object id="{next_id}" p:UUID="{uid()}" name="{mod["name"]} (Modifier)" type="model">\n{_mesh_xml(mod["mesh"])}\n  </object>')
            next_id += 1
        parts.append((body_id, mods))

    asm_ids, build, settings_objs, plates = [], [], [], []
    for i, (foot, (body_id, mods)) in enumerate(zip(feet, parts)):
        asm_id = next_id
        next_id += 1
        asm_ids.append(asm_id)
        comps = "".join(f'    <component objectid="{cid}" p:UUID="{uid()}"/>\n' for cid in [body_id] + [m[0] for m in mods])
        resources.append(f'  <object id="{asm_id}" p:UUID="{uid()}" name="{foot["label"]}" type="model">\n   <components>\n{comps}   </components>\n  </object>')
        tx, ty, tz = _placement(foot["mesh"], i)
        build.append(f'  <item objectid="{asm_id}" p:UUID="{uid()}" transform="{C:.6f} {S:.6f} 0 {-S:.6f} {C:.6f} 0 0 0 1 {tx:.4f} {ty:.4f} {tz:.4f}" printable="1"/>')

        part_xml = (
            f'    <part id="{body_id}" subtype="normal_part">\n'
            f'      <metadata key="name" value="{foot["label"]} Body"/>\n'
            f'      <metadata key="extruder" value="1"/>\n'
            f'    </part>\n'
        )
        for mid, mod in mods:
            part_xml += (
                f'    <part id="{mid}" subtype="modifier_part">\n'
                f'      <metadata key="name" value="{mod["name"]}"/>\n'
                f'      <metadata key="sparse_infill_density" value="{mod["density"]}"/>\n'
                f'      <metadata key="sparse_infill_pattern" value="gyroid"/>\n'
                f'    </part>\n'
            )
        settings_objs.append(
            f'  <object id="{asm_id}">\n'
            f'    <metadata key="name" value="{foot["label"]}"/>\n'
            f'    <metadata key="extruder" value="1"/>\n{part_xml}  </object>'
        )
        plates.append(
            f'  <plate>\n    <metadata key="plater_id" value="{i + 1}"/>\n'
            f'    <metadata key="plater_name" value="{foot["label"]}"/>\n'
            f'    <model_instance>\n      <metadata key="object_id" value="{asm_id}"/>\n'
            f'      <metadata key="instance_id" value="0"/>\n    </model_instance>\n  </plate>'
        )

    _package(out_path, resources, build, settings_objs, plates, asm_ids, process, "Custom ME3D Insoles")
