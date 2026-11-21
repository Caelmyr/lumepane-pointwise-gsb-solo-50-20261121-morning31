"""Follow-up reshoot plans built from real per-light photograph coverage."""

from dataclasses import dataclass
from io import StringIO
import csv
import math

import numpy as np
from PIL import Image, ImageDraw

from .catalog import capture_context
from .contracts import (
    DeskError, ROLES, dimensions, flag, identifier, identifiers,
    integer, number, obj, revision, word,
)
from .fracture import line_coverage
from .geometry import mask_for_marks, polygon_mask
from .imaging import canvas_picture, encode, label, resize_rgba
from .visuals import legend_band

PLAN_LIGHTS = ("transmitted", "reflected")
FRAME_COLORS = {"transmitted": (86, 168, 196), "reflected": (226, 169, 92)}
GAP_COLORS = {"transmitted": (68, 132, 176), "reflected": (205, 124, 84)}
OUTLINE_COLOR = (255, 236, 178)
MIN_BOX_SIDE = 0.004
MAX_TARGETS = 24
MERGE_GAP_MM = 35.0


@dataclass(frozen=True)
class PlanRequest:
    title: str
    targets: tuple
    lights: tuple
    frame_mm: tuple
    budget: dict
    minimum_gap_mm2: float
    readability: float

    @classmethod
    def parse(cls, raw):
        data = obj(raw, "reshoot plan")
        frame = dimensions(data.get("frame_coverage_mm"), "single frame coverage", low=5, high=12000)
        budget = obj(data.get("remaining_shots"), "remaining photographs")
        lights = obj(data.get("lights"), "lighting requests")
        picked = tuple(
            light for light in PLAN_LIGHTS
            if flag(lights.get(light), f"{light} follow-up lighting")
        )
        if not picked:
            raise DeskError("Request transmitted light, reflected light, or both.", field="lights")
        return cls(
            word(data.get("title"), "plan title", limit=100, default="Follow-up reshoot plan"),
            tuple(identifiers(data.get("targets"), "marked priority regions", high=MAX_TARGETS)),
            picked,
            tuple(frame),
            {light: integer(budget.get(light), f"remaining {light} photographs", 0, 60) for light in PLAN_LIGHTS},
            number(data.get("minimum_gap_mm2"), "minimum gap area", 0.01, 200000, default=4.0),
            number(data.get("readability_fraction"), "readable coverage target", .05, 1.0, default=.80),
        )


def role_surfaces(db, window_id):
    photos = db.list("captures", parent=window_id, limit=36, oldest=True)
    surfaces = {light: [] for light in PLAN_LIGHTS}
    for photo in photos:
        if photo["role"] not in PLAN_LIGHTS:
            continue
        try:
            surfaces[photo["role"]].append(capture_context(db, photo["id"], registered=True))
        except DeskError:
            continue
    return surfaces


def role_coverage(contexts, excluded):
    if not contexts:
        return None
    support = readable = None
    for ctx in contexts:
        layer = ctx.surface.valid & ~excluded
        clean = layer & ~ctx.surface.dark & ~ctx.surface.saturated
        support = layer if support is None else support | layer
        readable = clean if readable is None else readable | clean
    return {"support": support, "readable": readable}


def target_mask(extent, mark):
    if mark["shape"] == "polygon":
        return polygon_mask(extent, mark["points"], mark["holes"])
    return line_coverage(extent, [mark], width=3)


def clamp_box(box):
    x0, y0, x1, y1 = box
    if x1 - x0 < MIN_BOX_SIDE:
        center = (x0 + x1) / 2
        x0, x1 = max(0.0, center - MIN_BOX_SIDE / 2), min(1.0, center + MIN_BOX_SIDE / 2)
    if y1 - y0 < MIN_BOX_SIDE:
        center = (y0 + y1) / 2
        y0, y1 = max(0.0, center - MIN_BOX_SIDE / 2), min(1.0, center + MIN_BOX_SIDE / 2)
    return [x0, y0, x1, y1]


def mask_box(mask, extent):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    width, height = extent
    return clamp_box([
        xs.min() / width, ys.min() / height, (xs.max() + 1) / width, (ys.max() + 1) / height,
    ])


def box_size_mm(box, physical):
    return [(box[2] - box[0]) * physical[0], (box[3] - box[1]) * physical[1]]


def box_gap_mm(first, second, physical):
    horizontal = max(0.0, max(first[0], second[0]) - min(first[2], second[2])) * physical[0]
    vertical = max(0.0, max(first[1], second[1]) - min(first[3], second[3])) * physical[1]
    return math.hypot(horizontal, vertical)


def merge_boxes(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def mm2_per_pixel(extent, physical):
    return float(np.prod(np.asarray(physical, dtype=np.float64) / np.asarray(extent)))


def assess_targets(window, marks, req, surfaces, excluded):
    extent = next((ctx.surface.extent for group in surfaces.values() for ctx in group), None)
    if extent is None:
        raise DeskError("Register at least one transmitted or reflected photograph to the window plane.")
    chosen = {mark["id"]: mark for mark in marks}
    missing = [key for key in req.targets if key not in chosen]
    if missing:
        raise DeskError("A selected marked region is no longer present on this window.", field="targets")
    pixel_area = mm2_per_pixel(extent, window["physical_mm"])
    coverages = {light: role_coverage(surfaces[light], excluded) for light in req.lights}
    rows = []
    for mark in (chosen[key] for key in req.targets):
        mask = target_mask(extent, mark)
        total = int(mask.sum())
        if not total:
            raise DeskError(f"The region {mark['title']} has no measurable area.")
        lights = {}
        for light in req.lights:
            coverage = coverages[light]
            if coverage is None:
                gap = mask.copy()
                seen = clear = 0
            else:
                gap = mask & ~coverage["readable"]
                seen = int(np.count_nonzero(mask & coverage["support"]))
                clear = int(np.count_nonzero(mask & coverage["readable"]))
            gap_mm2 = round(int(gap.sum()) * pixel_area, 3)
            readable_fraction = clear / total
            lights[light] = {
                "registered_photographs": len(surfaces[light]),
                "photographed_percent": round(100 * seen / total, 3),
                "readable_percent": round(100 * readable_fraction, 5),
                "gap_pixels": int(gap.sum()),
                "gap_mm2": gap_mm2,
                "gap_fraction": round(float(gap.sum()) / total, 5),
                "needs_followup": bool(
                    coverage is None
                    or (gap_mm2 >= req.minimum_gap_mm2 and readable_fraction < req.readability)
                ),
                "gap_mask": gap,
            }
        rows.append({"mark": mark, "mask": mask, "area_mm2": round(total * pixel_area, 3), "lights": lights})
    return rows, extent, pixel_area


def build_frames(rows, req, physical, pixel_area, extent):
    demands = []
    for row in rows:
        for light in req.lights:
            reading = row["lights"].get(light)
            if not reading or not reading["needs_followup"]:
                continue
            demands.append({
                "light": light, "target": row["mark"]["id"], "title": row["mark"]["title"],
                "box": mask_box(row["mask"] & reading["gap_mask"], extent) or [0.0, 0.0, 0.0, 0.0],
                "gap_mm2": reading["gap_mm2"], "gap_mask": reading["gap_mask"],
            })
    demands.sort(key=lambda item: -item["gap_mm2"])
    clusters = []

    def fits(box):
        return all(side <= cap for side, cap in zip(box_size_mm(box, physical), req.frame_mm))

    for demand in demands:
        if not fits(demand["box"]):
            continue
        for cluster in clusters:
            if cluster["light"] != demand["light"]:
                continue
            candidate = merge_boxes([cluster["box"], demand["box"]])
            near = box_gap_mm(cluster["box"], demand["box"], physical) <= MERGE_GAP_MM
            if near and fits(candidate):
                cluster["box"] = candidate
                cluster["members"].append(demand)
                break
        else:
            clusters.append({"light": demand["light"], "box": demand["box"], "members": [demand]})
    frames, scheduled = [], set()
    for cluster in clusters:
        light = cluster["light"]
        used = sum(1 for frame in frames if frame["light"] == light)
        if used >= req.budget[light]:
            continue
        gap_union = np.logical_or.reduce([member["gap_mask"] for member in cluster["members"]])
        frames.append({
            "code": f"{light[0].upper()}{used + 1}",
            "light": light,
            "lighting_role": light,
            "lighting_label": ROLES[light],
            "frame_box": cluster["box"],
            "frame_size_mm": [round(v, 2) for v in box_size_mm(cluster["box"], physical)],
            "coverage_capacity_mm": [round(req.frame_mm[0], 2), round(req.frame_mm[1], 2)],
            "targets": sorted({member["target"] for member in cluster["members"]}),
            "target_titles": sorted({member["title"] for member in cluster["members"]}),
            "expected_gap_pixels": int(gap_union.sum()),
            "expected_fill_mm2": round(int(gap_union.sum()) * pixel_area, 3),
        })
        scheduled.update((light, member["target"]) for member in cluster["members"])
    return frames, scheduled


def dashed_rectangle(pen, box, size, color, width):
    x0, y0, x1, y1 = box
    corners = [(x0 * size[0], y0 * size[1]), (x1 * size[0], y0 * size[1]),
               (x1 * size[0], y1 * size[1]), (x0 * size[0], y1 * size[1])]
    for start, end in zip(corners, corners[1:] + corners[:1]):
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        steps = max(1, int(distance / 9))
        for step in range(steps):
            if step % 2:
                continue
            a, b = step / steps, min(1.0, (step + 1) / steps)
            pen.line(
                (start[0] + (end[0] - start[0]) * a, start[1] + (end[1] - start[1]) * a,
                 start[0] + (end[0] - start[0]) * b, start[1] + (end[1] - start[1]) * b),
                fill=color, width=width,
            )


def gap_overlay(image, masks, extent, color):
    union = np.logical_or.reduce(masks) if masks else None
    if union is None or not union.any():
        return image
    resized = np.asarray(Image.fromarray(union.astype(np.uint8) * 255).resize(image.size, Image.Resampling.NEAREST), dtype=bool)
    layer = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    layer[resized] = np.array([*color, 190], dtype=np.uint8)
    return Image.alpha_composite(image.convert("RGBA"), Image.fromarray(layer, "RGBA")).convert("RGB")


def plan_map(surfaces, rows, frames, scheduled, req, extent):
    base_ctx = next(ctx for group in surfaces.values() for ctx in group)
    image = resize_rgba(canvas_picture(base_ctx.surface), 1200).convert("RGB")
    for light in req.lights:
        image = gap_overlay(image, [row["lights"][light]["gap_mask"] for row in rows], extent, GAP_COLORS[light])
    pen = ImageDraw.Draw(image, "RGBA")
    for row in rows:
        points = [(x * image.width, y * image.height) for x, y in row["mark"]["points"]]
        if row["mark"]["shape"] == "polygon":
            points.append(points[0])
        pen.line(points, fill=OUTLINE_COLOR + (200,), width=2)
    for frame in frames:
        box = frame["frame_box"]
        rectangle = [box[0] * image.width, box[1] * image.height, box[2] * image.width, box[3] * image.height]
        pen.rectangle(rectangle, outline=FRAME_COLORS[frame["light"]] + (255,), width=4)
        label(pen, (rectangle[0] + 4, max(2, rectangle[1] - 19)), frame["code"],
              fill="#fff6df", background="#222a36", size=14)
    for row in rows:
        for light in req.lights:
            reading = row["lights"].get(light)
            if not reading or not reading["needs_followup"] or (light, row["mark"]["id"]) in scheduled:
                continue
            box = mask_box(row["mask"] & reading["gap_mask"], extent)
            dashed_rectangle(pen, box, image.size, GAP_COLORS[light] + (255,), 2)
    legend = [(f"{ROLES[light]} follow-up frame", FRAME_COLORS[light]) for light in req.lights]
    legend += [(f"{ROLES[light]} unresolved gap", GAP_COLORS[light]) for light in req.lights]
    legend.append(("Marked priority region", OUTLINE_COLOR))
    return legend_band(image, "Planned follow-up coverage on the window plane", legend, lines=2)


def target_rows(rows, scheduled, req):
    result = []
    for row in rows:
        lights = {}
        for light in req.lights:
            reading = row["lights"].get(light)
            lights[light] = (
                {key: value for key, value in reading.items() if key != "gap_mask"}
                | {"scheduled": (light, row["mark"]["id"]) in scheduled}
                if reading else None
            )
        result.append({
            "mark": row["mark"]["id"], "title": row["mark"]["title"],
            "category": row["mark"]["category"], "shape": row["mark"]["shape"],
            "area_mm2": row["area_mm2"], "lights": lights,
        })
    return result


def prepare_plan(library, window_id, raw):
    req = PlanRequest.parse(raw)
    with library.use() as db:
        window = db.one(window_id, "windows")
        if not window["physical_mm"]:
            raise DeskError("Record the physical window dimensions before planning frame coverage.")
        marks = [item for item in db.list("marks", parent=window_id, limit=160, oldest=True) if not item["retired"]]
        surfaces = role_surfaces(db, window_id)
        first_ctx = next((ctx for group in surfaces.values() for ctx in group), None)
        excluded = (
            mask_for_marks(first_ctx.surface.extent, first_ctx.marks, {"ignore", "occluded"})
            if first_ctx else None
        )
        rows, extent, pixel_area = assess_targets(window, marks, req, surfaces, excluded)
        frames, scheduled = build_frames(rows, req, window["physical_mm"], pixel_area, extent)
        map_image = plan_map(surfaces, rows, frames, scheduled, req, extent)
        payload = {
            "title": req.title,
            "window_version": window["version"],
            "frame_coverage_mm": list(req.frame_mm),
            "remaining_shots": dict(req.budget),
            "readability_fraction": req.readability,
            "minimum_gap_mm2": req.minimum_gap_mm2,
            "requested_lights": list(req.lights),
            "targets": target_rows(rows, scheduled, req),
            "frames": frames,
            "photograph_basis": {
                light: [ctx.capture["id"] for ctx in surfaces[light]] for light in PLAN_LIGHTS
            },
            "window_basis": {
                key: window[key] for key in ("id", "version", "physical_mm", "working_extent", "title")
            },
            "note": "Gaps use registered photographs that are in frame, not excluded, and neither dark nor highlight clipped.",
        }
        return payload, map_image


def create_plan(library, window_id, raw):
    payload, map_image = prepare_plan(library, window_id, raw)
    with library.use(write=True) as db:
        saved = db.add("resurvey_plans", payload, parent=window_id, action="planned follow-up reshoot frames")
        db.link(saved["id"], db.one(window_id, "windows"), "planned window")
        db.attach(saved["id"], "map", encode(map_image), "image/png")
        return db.public(saved)


def revise_plan(library, key, raw):
    data = obj(raw)
    window_id = identifier(data.get("window_id"), "window")
    payload, map_image = prepare_plan(library, window_id, raw)
    with library.use(write=True) as db:
        existing = db.one(key, "resurvey_plans")
        if existing["parent"] != window_id:
            raise DeskError("This plan belongs to a different window.")
        saved = db.revise(key, "resurvey_plans", revision(raw), payload, "revised the follow-up reshoot plan")
        db.replace_asset(key, "map", encode(map_image), "image/png")
        return db.public(saved)


def plan_list(library, window_id):
    with library.use() as db:
        db.one(window_id, "windows")
        return {"plans": db.list("resurvey_plans", parent=window_id, limit=40)}


def plan_csv(record):
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(("Frame", "Lighting", "Frame width mm", "Frame height mm", "Coverage cap mm", "Priority regions"))
    for frame in record["frames"]:
        writer.writerow((
            frame["code"], frame["lighting_label"], frame["frame_size_mm"][0], frame["frame_size_mm"][1],
            " x ".join(str(v) for v in frame["coverage_capacity_mm"]), "; ".join(frame["target_titles"]),
        ))
    writer.writerow(())
    writer.writerow(("Priority region", "Lighting", "Photographed %", "Readable %", "Gap mm2", "Scheduled"))
    for target in record["targets"]:
        for light, reading in target["lights"].items():
            if reading is None:
                continue
            writer.writerow((
                target["title"], ROLES[light], reading["photographed_percent"], reading["readable_percent"],
                reading["gap_mm2"], "yes" if reading["scheduled"] else "no",
            ))
    return output.getvalue().encode()
