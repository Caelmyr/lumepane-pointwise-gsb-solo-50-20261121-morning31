"""Reviewable crack path map: branches, junctions, and lengths from visible evidence only."""

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .contracts import number, obj
from .fracture import (
    NEIGHBOURS, FractureSettings, candidates_for, fracture_field,
    hessian_response, label_candidates, observation_match,
)
from .imaging import canvas_picture, label
from .visuals import AQUA, GOLD, PALE, ROSE, SKY, legend_band, rgb_triplet

SEGMENT_CAP = 400
JUNCTION_CAP = 250
ENDPOINT_CAP = 300
BREAK_CAP = 150
CAME_CAP = 150
CANDIDATE_BOX = (240, 184, 145)
EVIDENCE_OVERLAP = 0.3


@dataclass(frozen=True)
class PathSettings:
    base: FractureSettings
    bridge_gap: float
    bridge_angle: float
    min_segment: float

    @classmethod
    def parse(cls, raw):
        data = obj(raw)
        return cls(
            FractureSettings.parse(data),
            number(data.get("bridge_gap"), "review gap reach", 4, 160, default=40),
            number(data.get("bridge_angle"), "facing tolerance", 10, 80, default=38),
            number(data.get("min_segment"), "minimum segment length", 1, 80, default=4),
        )

    def json(self):
        return self.base.json() | {
            "bridge_gap": self.bridge_gap, "bridge_angle": self.bridge_angle,
            "min_segment": self.min_segment,
        }


def zhang_suen(mask):
    image = np.pad(mask.astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for phase in (0, 1):
            p2 = image[:-2, 1:-1]
            p3 = image[:-2, 2:]
            p4 = image[1:-1, 2:]
            p5 = image[2:, 2:]
            p6 = image[2:, 1:-1]
            p7 = image[2:, :-2]
            p8 = image[1:-1, :-2]
            p9 = image[:-2, :-2]
            ring = (p2, p3, p4, p5, p6, p7, p8, p9)
            neighbours = sum(ring)
            transitions = sum((ring[i] == 0) & (ring[(i + 1) % 8] == 1) for i in range(8))
            if phase == 0:
                vanish = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                vanish = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (image[1:-1, 1:-1] == 1) & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1) & vanish
            if remove.any():
                image[1:-1, 1:-1][remove] = 0
                changed = True
    return image[1:-1, 1:-1].astype(bool)


def skeletonize(mask):
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=bool)
    ys, xs = np.nonzero(mask)
    y0, y1 = max(0, ys.min() - 3), min(mask.shape[0], ys.max() + 4)
    x0, x1 = max(0, xs.min() - 3), min(mask.shape[1], xs.max() + 4)
    out = np.zeros(mask.shape, dtype=bool)
    out[y0:y1, x0:x1] = zhang_suen(mask[y0:y1, x0:x1])
    return out


def trace_graph(skel):
    density = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant")
    degree = (density - 1) * skel
    node_mask = skel & (degree != 2)
    node_of, node_count = ndimage.label(node_mask, structure=NEIGHBOURS)
    ys, xs = np.nonzero(node_mask)
    tags = node_of[node_mask]
    sizes = np.bincount(tags, minlength=node_count + 1)
    sums_y = np.bincount(tags, weights=ys, minlength=node_count + 1)
    sums_x = np.bincount(tags, weights=xs, minlength=node_count + 1)
    kinds, positions, node_pixels = {}, {}, {}
    for node in range(1, node_count + 1):
        cluster = node_of == node
        peak = int(degree[cluster].max())
        kinds[node] = "junction" if peak >= 3 else "endpoint" if peak == 1 else "isolated"
        positions[node] = (float(sums_x[node] / sizes[node]), float(sums_y[node] / sizes[node]))
        node_pixels[node] = list(zip(*np.nonzero(cluster)))
    height, width = skel.shape

    def neighbours(y, x):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and skel[ny, nx]:
                    yield ny, nx

    consumed = np.zeros(skel.shape, dtype=bool)
    segments = []

    def walk(start, first):
        path = [start, first]
        prev, cur = start, first
        while not node_mask[cur]:
            consumed[cur] = True
            following = [p for p in neighbours(*cur) if p != prev]
            if not following:
                break
            prev, cur = cur, following[0]
            path.append(cur)
        end = int(node_of[cur]) if node_mask[cur] else 0
        return path, end

    for y, x in zip(ys, xs):
        origin = int(node_of[y, x])
        for ny, nx in neighbours(y, x):
            if node_mask[ny, nx] or consumed[ny, nx]:
                continue
            path, end = walk((int(y), int(x)), (ny, nx))
            segments.append({"nodes": (origin, end), "pixels": path})
    rest = skel & ~node_mask & ~consumed
    loops, loop_count = ndimage.label(rest, structure=NEIGHBOURS)
    for loop_id in range(1, loop_count + 1):
        ly, lx = np.nonzero(loops == loop_id)
        start = (int(ly[0]), int(lx[0]))
        path = [start]
        prev, cur = None, start
        while True:
            following = [p for p in neighbours(*cur) if p != prev and loops[p] == loop_id]
            if not following:
                break
            prev, cur = cur, following[0]
            path.append(cur)
            if cur == start:
                break
        segments.append({"nodes": (0, 0), "pixels": path})
    return kinds, positions, node_pixels, segments


def segment_length(pixels, scale):
    xy = np.asarray(pixels, dtype=np.float64)[:, ::-1]
    steps = np.diff(xy, axis=0)
    length_px = float(np.linalg.norm(steps, axis=1).sum()) if len(steps) else 0.0
    length_mm = None
    if scale is not None and len(steps):
        length_mm = float(np.linalg.norm(steps * np.asarray(scale, dtype=np.float64), axis=1).sum())
    return length_px, length_mm


def majority_candidate(index_image, pixels):
    votes = index_image[tuple(np.asarray(pixels, dtype=int).T)]
    votes = votes[votes > 0]
    if not len(votes):
        return None
    return int(np.bincount(votes).argmax())


def node_degrees(segments):
    degree = {}
    for seg in segments:
        a, b = seg["nodes"]
        for node in {a, b}:
            if node:
                degree[node] = degree.get(node, 0) + (2 if a == b else 1)
    return degree


def simplify_segments(segments, prune_px):
    """Prune short spurs and dissolve degree-2 nodes so junctions are real branch points."""
    segs = [dict(seg) for seg in segments]
    pruned = 0
    changed = True
    while changed:
        changed = False
        degree = node_degrees(segs)
        keep = []
        for seg in segs:
            a, b = seg["nodes"]
            if seg["length_px"] < prune_px and (degree.get(a, 0) == 1 or degree.get(b, 0) == 1):
                pruned += 1
                changed = True
                continue
            keep.append(seg)
        segs = keep
        if changed:
            continue
        incident = {}
        for idx, seg in enumerate(segs):
            for node in set(seg["nodes"]):
                if node:
                    incident.setdefault(node, []).append(idx)
        for node, idxs in incident.items():
            if len(idxs) != 2:
                continue
            first, second = (segs[i] for i in idxs)
            if first["nodes"][0] == first["nodes"][1] or second["nodes"][0] == second["nodes"][1]:
                continue
            a_pix = first["pixels"] if first["nodes"][1] == node else first["pixels"][::-1]
            a_nodes = first["nodes"] if first["nodes"][1] == node else first["nodes"][::-1]
            b_pix = second["pixels"] if second["nodes"][0] == node else second["pixels"][::-1]
            b_nodes = second["nodes"] if second["nodes"][0] == node else second["nodes"][::-1]
            merged = {
                "nodes": (a_nodes[0], b_nodes[1]),
                "pixels": a_pix + b_pix[1:],
                "length_px": first["length_px"] + second["length_px"],
                "length_mm": None if first["length_mm"] is None or second["length_mm"] is None
                else first["length_mm"] + second["length_mm"],
            }
            segs = [seg for k, seg in enumerate(segs) if k not in idxs] + [merged]
            changed = True
            break
    return segs, pruned


def outward_tangent(segment, node):
    pixels = segment["pixels"]
    if segment["nodes"][0] == node:
        a, b = pixels[0], pixels[min(len(pixels) - 1, 6)]
    elif segment["nodes"][1] == node:
        a, b = pixels[-1], pixels[max(0, len(pixels) - 7)]
    else:
        return None
    dx, dy = a[1] - b[1], a[0] - b[0]
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return None
    return dx / norm, dy / norm


def facing_breaks(endpoints, tangent_of, positions, unknown, detections, settings, extent):
    width, height = extent
    cos_limit = math.cos(math.radians(settings.bridge_angle))
    pairs = []
    for i, first in enumerate(endpoints):
        ta = tangent_of.get(first)
        if ta is None:
            continue
        pa = positions[first]
        for second in endpoints[i + 1:]:
            tb = tangent_of.get(second)
            if tb is None:
                continue
            pb = positions[second]
            dx, dy = pb[0] - pa[0], pb[1] - pa[1]
            gap = math.hypot(dx, dy)
            if gap < 3 or gap > settings.bridge_gap:
                continue
            ux, uy = dx / gap, dy / gap
            if ta[0] * ux + ta[1] * uy < cos_limit:
                continue
            if -(tb[0] * ux + tb[1] * uy) < cos_limit:
                continue
            ts = np.linspace(0, 1, max(3, int(gap)) + 1)[1:-1]
            xs = np.clip(np.rint(pa[0] + dx * ts).astype(int), 0, width - 1)
            ys = np.clip(np.rint(pa[1] + dy * ts).astype(int), 0, height - 1)
            occluded = float(unknown[ys, xs].mean())
            if occluded < 0.5 or float(detections[ys, xs].mean()) > 0.2:
                continue
            pairs.append((gap, first, second, occluded))
    pairs.sort(key=lambda item: item[0])
    paired, chosen = set(), []
    for gap, first, second, occluded in pairs:
        if first in paired or second in paired:
            continue
        paired |= {first, second}
        chosen.append({"ends": (first, second), "gap_px": gap, "occluded": occluded})
    return chosen


def came_band_lines(ctx, field, base, scale, min_segment):
    band = field["selection"] & field["came"] & field["visible"]
    if not np.any(band):
        return []
    support = ndimage.binary_erosion(field["core"] & band, iterations=max(1, base.feature_width // 2))
    response = hessian_response(field["signal"], max(1, base.feature_width), base.polarity)
    response[~support] = 0
    skel = skeletonize(support & (response >= field["threshold"]))
    _, _, _, raw = trace_graph(skel)
    width, height = ctx.surface.extent
    fracture_marks = [m for m in ctx.marks if m["category"] == "fracture"]
    lines = []
    for item in raw:
        length_px, length_mm = segment_length(item["pixels"], scale)
        if length_px < min_segment:
            continue
        mask = np.zeros((height, width), dtype=bool)
        mask[tuple(np.asarray(item["pixels"]).T)] = True
        dilated = ndimage.binary_dilation(mask, iterations=2)
        total = max(1, int(np.count_nonzero(dilated)))
        evidence = [m for m in observation_match(width, height, dilated, fracture_marks)
                    if m["overlapping_pixels"] >= 6 and m["overlapping_pixels"] / total >= EVIDENCE_OVERLAP]
        lines.append({
            "pixels": item["pixels"], "length_px": length_px, "length_mm": length_mm,
            "evidence": evidence, "counted": bool(evidence),
        })
    lines.sort(key=lambda line: -line["length_px"])
    return lines


def dashed_polyline(pen, points, color, width, dash=6.0, gap=4.0):
    pattern = dash + gap
    dist = 0.0
    for a, b in zip(points, points[1:]):
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        length = math.hypot(bx - ax, by - ay)
        if length < 1e-9:
            continue
        ux, uy = (bx - ax) / length, (by - ay) / length
        t = 0.0
        while t < length:
            pos = (dist + t) % pattern
            step = min(length - t, dash - pos if pos < dash else pattern - pos)
            if pos < dash:
                pen.line([(ax + ux * t, ay + uy * t), (ax + ux * (t + step), ay + uy * (t + step))],
                         fill=color, width=width)
            t += step
        dist += length


def draw_path_map(ctx, field, segments, positions, names, breaks, came_lines, candidates):
    base = np.asarray(canvas_picture(ctx.surface).convert("RGB"), dtype=np.float32)
    base[~field["selection"]] *= .45
    came = field["selection"] & field["came"]
    base[came] = base[came] * .72 + np.asarray(rgb_triplet(AQUA), dtype=np.float32) * .28
    picture = Image.fromarray(np.uint8(np.clip(base, 0, 255)), "RGB")
    pen = ImageDraw.Draw(picture)
    rose, sky, gold, pale, aqua = (rgb_triplet(c) for c in (ROSE, SKY, GOLD, PALE, AQUA))
    width, _ = picture.size
    dot = max(2, width // 340)

    def xy(pixel):
        return pixel[1], pixel[0]

    for line in came_lines:
        points = [xy(p) for p in line["pixels"]]
        if line["counted"]:
            pen.line(points, fill=aqua, width=2)
        else:
            dashed_polyline(pen, points, aqua, 2)
        mx, my = xy(line["pixels"][len(line["pixels"]) // 2])
        label(pen, (mx + 4, my + 4), f"c{line['cid']}", size=10, fill=AQUA)
    for seg in segments:
        pen.line([xy(p) for p in seg["pixels"]], fill=rose, width=2)
        mx, my = xy(seg["pixels"][len(seg["pixels"]) // 2])
        label(pen, (mx + 3, my - 12), f"s{seg['sid']}", size=10, fill=ROSE)
    for node, name in names["junction"].items():
        x, y = positions[node]
        pen.ellipse((x - dot, y - dot, x + dot, y + dot), fill=gold)
        label(pen, (x + 5, y - 6), name, size=10, fill=GOLD)
    broken = {node for brk in breaks for node in brk["ends"]}
    for node, name in names["endpoint"].items():
        x, y = positions[node]
        pen.ellipse((x - dot, y - dot, x + dot, y + dot), outline=pale, width=max(1, dot // 2))
        if node in broken:
            label(pen, (x + 5, y + 3), name, size=10, fill=PALE)
    for brk in breaks:
        a, b = (positions[node] for node in brk["ends"])
        dashed_polyline(pen, [a, b], sky, 2)
        label(pen, ((a[0] + b[0]) / 2 + 4, (a[1] + b[1]) / 2 - 12), f"b{brk['bid']} ?", size=10, fill=SKY)
    picture = label_candidates(picture, candidates)
    return legend_band(picture, "Crack path review · visible evidence", [
        ("Counted path", rose), ("Junction", gold), ("Endpoint", pale),
        ("Review break", sky), ("Came-edge line", aqua), ("Candidate box", CANDIDATE_BOX),
    ])


def limited(rows, cap, readout, key):
    if len(rows) > cap:
        readout[key + "_omitted"] = len(rows) - cap
        return rows[:cap]
    return rows


def normalized(point, extent):
    width, height = extent
    return [round(point[0] / width, 6), round(point[1] / height, 6)]


def crack_path_map(ctx, raw):
    settings = PathSettings.parse(raw)
    base = settings.base
    field = fracture_field(ctx, base)
    candidates, candidate_mask, raw_count, index_image = candidates_for(ctx, field, base)
    extent = ctx.surface.extent
    scale = ctx.scale if ctx.alignment else None

    _, positions, node_pixels, raw_segments = trace_graph(skeletonize(candidate_mask))
    measured = []
    for item in raw_segments:
        length_px, length_mm = segment_length(item["pixels"], scale)
        measured.append(item | {"length_px": length_px, "length_mm": length_mm})
    segments, spurs = simplify_segments(measured, settings.min_segment)
    listed = []
    for seg in segments:
        if seg["length_px"] < settings.min_segment:
            spurs += 1
            continue
        seg["candidate"] = majority_candidate(index_image, seg["pixels"])
        listed.append(seg)
    segments = listed
    segments.sort(key=lambda seg: (seg["candidate"] or 1 << 30, -seg["length_px"]))
    for i, seg in enumerate(segments):
        seg["sid"] = i + 1

    degree = node_degrees(segments)
    used = {node for seg in segments for node in seg["nodes"] if node}
    ordered = sorted(used, key=lambda node: (positions[node][1], positions[node][0]))
    names = {"junction": {}, "endpoint": {}}
    for node in ordered:
        deg = degree.get(node, 0)
        if deg >= 3:
            kind = "junction"
        elif deg == 1:
            kind = "endpoint"
        else:
            continue
        names[kind][node] = f"{kind[0]}{len(names[kind]) + 1}"
    segment_of = {}
    for seg in segments:
        for node in seg["nodes"]:
            if node:
                segment_of.setdefault(node, seg)
    tangent_of = {node: outward_tangent(segment_of[node], node) for node in names["endpoint"]}

    breaks = facing_breaks(
        list(names["endpoint"]), tangent_of, positions,
        ~field["eligible"], field["detections"], settings, extent,
    )
    if len(breaks) > BREAK_CAP:
        readout_extra_breaks = len(breaks) - BREAK_CAP
        breaks = breaks[:BREAK_CAP]
    else:
        readout_extra_breaks = 0
    for i, brk in enumerate(breaks):
        brk["bid"] = i + 1
        if scale is not None:
            (ax, ay), (bx, by) = (positions[node] for node in brk["ends"])
            ux, uy = (bx - ax) / brk["gap_px"], (by - ay) / brk["gap_px"]
            brk["gap_mm"] = brk["gap_px"] * math.hypot(ux * scale[0], uy * scale[1])
        else:
            brk["gap_mm"] = None
    break_of = {node: brk["bid"] for brk in breaks for node in brk["ends"]}

    came_lines = came_band_lines(ctx, field, base, scale, settings.min_segment)
    if len(came_lines) > CAME_CAP:
        readout_extra_came = len(came_lines) - CAME_CAP
        came_lines = came_lines[:CAME_CAP]
    else:
        readout_extra_came = 0
    for i, line in enumerate(came_lines):
        line["cid"] = i + 1

    visible_px = sum(seg["length_px"] for seg in segments)
    visible_mm = sum(seg["length_mm"] for seg in segments) if scale is not None else None
    counted = [line for line in came_lines if line["counted"]]
    visible_px += sum(line["length_px"] for line in counted)
    if visible_mm is not None:
        visible_mm += sum(line["length_mm"] for line in counted)

    readout = {
        "settings": settings.json(),
        "candidate_count": len(candidates), "raw_component_count": raw_count,
        "segment_count": len(segments), "junction_count": len(names["junction"]),
        "endpoint_count": len(names["endpoint"]),
        "visible_path_length_px": round(visible_px, 4),
        "visible_path_length_mm": round(visible_mm, 4) if visible_mm is not None else None,
        "longest_segment_px": round(max((s["length_px"] for s in segments), default=0), 4) or None,
        "spurs_below_minimum": spurs,
        "review_break_count": len(breaks),
        "occluded_gap_px_under_review": round(sum(b["gap_px"] for b in breaks), 4),
        "came_edge_line_count": len(came_lines), "came_edge_counted": len(counted),
    }
    if readout_extra_breaks:
        readout["review_breaks_omitted"] = readout_extra_breaks
    if readout_extra_came:
        readout["came_edge_lines_omitted"] = readout_extra_came

    def mm(value):
        return round(value, 4) if value is not None else None

    seg_rows = [{
        "segment": seg["sid"], "candidate": seg["candidate"],
        "length_px": round(seg["length_px"], 4), "length_mm": mm(seg["length_mm"]),
        "from_node": names["junction"].get(seg["nodes"][0]) or names["endpoint"].get(seg["nodes"][0]) or "loop",
        "to_node": names["junction"].get(seg["nodes"][1]) or names["endpoint"].get(seg["nodes"][1]) or "loop",
        "midpoint": normalized(seg["pixels"][len(seg["pixels"]) // 2][::-1], extent),
    } for seg in segments]
    junction_rows = [{
        "junction": names["junction"][node], "at": normalized(positions[node], extent),
        "branches": degree[node],
        "segments": [seg["sid"] for seg in segments if node in seg["nodes"]],
        "candidate": majority_candidate(index_image, node_pixels[node]),
    } for node in names["junction"]]
    endpoint_rows = [{
        "endpoint": names["endpoint"][node], "at": normalized(positions[node], extent),
        "segment": segment_of[node]["sid"], "candidate": segment_of[node]["candidate"],
        "review_break": break_of.get(node),
    } for node in names["endpoint"]]
    break_rows = [{
        "break_point": brk["bid"], "status": "review",
        "ends": [normalized(positions[node], extent) for node in brk["ends"]],
        "endpoints": [names["endpoint"][node] for node in brk["ends"]],
        "gap_px": round(brk["gap_px"], 4), "gap_mm": mm(brk["gap_mm"]),
        "occluded_fraction": round(brk["occluded"], 4),
        "candidates": sorted({segment_of[node]["candidate"] for node in brk["ends"]} - {None}),
        "note": "Facing ends across an occluded gap; continuity and gap length stay unverified.",
    } for brk in breaks]
    came_rows = [{
        "line": line["cid"], "counted": line["counted"],
        "length_px": round(line["length_px"], 4), "length_mm": mm(line["length_mm"]),
        "midpoint": normalized(line["pixels"][len(line["pixels"]) // 2][::-1], extent),
        "independent_evidence": line["evidence"],
    } for line in came_lines]
    candidate_rows = []
    for cand in candidates:
        own = [seg for seg in segments if seg["candidate"] == cand["index"]]
        length_px = sum(seg["length_px"] for seg in own)
        candidate_rows.append({
            "candidate": cand["index"], "bounds": cand["bounds"], "centroid": cand["centroid"],
            "segments": [seg["sid"] for seg in own],
            "junctions": [row["junction"] for row in junction_rows if row["candidate"] == cand["index"]],
            "visible_length_px": round(length_px, 4),
            "visible_length_mm": mm(sum(seg["length_mm"] for seg in own)) if scale is not None else None,
            "source_marks": cand["source_marks"],
        })

    readout.update({
        "segments": limited(seg_rows, SEGMENT_CAP, readout, "segments"),
        "junctions": limited(junction_rows, JUNCTION_CAP, readout, "junctions"),
        "endpoints": limited(endpoint_rows, ENDPOINT_CAP, readout, "endpoints"),
        "review_breaks": break_rows,
        "came_edge_lines": came_rows,
        "candidates": candidate_rows,
        "note": "Segments, junctions, and lengths use only visible ridge evidence inside listed candidates; "
                "candidate numbers match the Fine fracture candidates examination at the same settings. "
                "Dashed breaks span occluded or came-covered gaps and stay unverified and uncounted. "
                "Lines inside the came clearance count only with an independent recorded fracture.",
    })
    art = draw_path_map(ctx, field, segments, positions, names, breaks, came_lines, candidates)
    return readout, {"map": art}
