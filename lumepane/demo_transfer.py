"""Synthetic re-framed window: an old frame record and its replacement frame.

The glass artwork is rendered once and then photographed through two
different frame apertures. The new aperture is shifted and slightly scaled,
and one lower-right area is masked by the new mount, so some old
observations carry over cleanly while others cross the new boundary or land
in an occluded region and must be held for review.
"""

from threading import Lock

import numpy as np
from PIL import Image

from .catalog import new_capture
from .demo import panel_board, regular_patterns
from .geometry import CORNERS, native_crop_registration, output_plane
from .imaging import encode, image_rgb, photo_decode

TRANSFER_LOCK = Lock()
OLD_APERTURE = (0.0, 0.0, 1.0, 1.0)
NEW_APERTURE = (-0.035, -0.025, 0.945, 0.985)
OCCLUSION_BOX = (0.80, 0.55, 0.97, 0.78)
EXTENT = 688, 498

FRAME_SCHEME = {
    "title": "Hallowell · Re-framed quatrefoil",
    "code": "HF–17",
    "physical_mm": [440, 318.5],
    "location": "Synthetic east cloister",
    "central_color": (35, 104, 128),
    "secondary_color": (196, 132, 58),
    "seed": 24601,
}

# Shared, recognisable features expressed in the coordinates of each frame.
LANDMARKS = [
    {"name": "Upper lead joint", "old": [0.245, 0.205], "new": [0.2857, 0.2277]},
    {"name": "Right came crossing", "old": [0.715, 0.300], "new": [0.7653, 0.3218]},
    {"name": "Lower lead joint", "old": [0.330, 0.735], "new": [0.3724, 0.7525]},
    {"name": "Left rim knot", "old": [0.150, 0.520], "new": [0.1888, 0.5396]},
    {"name": "Center boss", "old": [0.500, 0.505], "new": [0.5459, 0.5248]},
    {"name": "Upper right petal tip", "old": [0.845, 0.180], "new": [0.898, 0.203]},
]

OLD_MARKS = [
    ("Upper ornamental pane", "pane", "polygon",
     [[.12, .16], [.44, .16], [.43, .42], [.14, .42]], []),
    ("Central came spine", "came", "line",
     [[.50, .13], [.50, .36], [.50, .55], [.50, .72]], []),
    ("Fissure by the boss", "fracture", "line",
     [[.56, .29], [.583, .34], [.616, .39], [.66, .45]], []),
    ("Lower pane crossing the new edge", "pane", "polygon",
     [[.30, .84], [.62, .84], [.63, 1.0], [.31, 1.0]], []),
    ("Right pane lost to the mount", "pane", "polygon",
     [[.80, .56], [.95, .57], [.95, .76], [.80, .76]], []),
    ("Stable reference patch", "reference", "polygon",
     [[.37, .33], [.417, .331], [.417, .39], [.37, .39]], []),
]


def aperture_view(scheme, aperture, occluded, visit):
    """Render the artwork, then resample it through the new frame aperture.

    The aperture is expressed in old-image coordinates. Values outside
    ``0..1`` mean the replacement frame looks past the edge of the old
    photograph, so those output pixels have no photographed support.
    """
    x0, y0, x1, y1 = aperture
    width, height = EXTENT
    optics = panel_board(scheme)
    base, grain = regular_patterns(scheme)
    image = optics.palette * base[:, :, None]
    image += grain[:, :, None] * .65
    image -= (.23 * optics.cracks + .18 * optics.painted)[:, :, None]
    image[optics.frame] = (.035, .037, .043)
    area = np.ones((height, width), dtype=bool)
    if occluded:
        ox0, oy0, ox1, oy1 = OCCLUSION_BOX
        area[round(oy0 * height):round(oy1 * height),
             round(ox0 * width):round(ox1 * width)] = False
        image[~area] = (.08, .07, .09)
    full = image_rgb(np.clip(image, .006, 1.0), area, from_linear=False)

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    source_x = (x0 + (x1 - x0) * grid_x / (width - 1)) * (width - 1)
    source_y = (y0 + (y1 - y0) * grid_y / (height - 1)) * (height - 1)
    supported = ((source_x >= 0) & (source_x <= width - 1)
                 & (source_y >= 0) & (source_y <= height - 1))
    clipped_x = np.clip(source_x, 0, width - 1)
    clipped_y = np.clip(source_y, 0, height - 1)
    rgb = np.asarray(full.convert("RGB"), dtype=np.uint8)
    source_alpha = np.asarray(full.getchannel("A"))
    sampled_rgb = rgb[np.rint(clipped_y).astype(int), np.rint(clipped_x).astype(int)]
    sampled_alpha = source_alpha[np.rint(clipped_y).astype(int),
                                  np.rint(clipped_x).astype(int)]
    alpha = np.where(supported, sampled_alpha, 0).astype(np.uint8)
    out = Image.fromarray(sampled_rgb, "RGB")
    out.putalpha(Image.fromarray(alpha))
    return out


def save_frame_photo(db, window, scheme, aperture, occluded, role, visit, date, caption):
    image = aperture_view(scheme, aperture, occluded, visit)
    raw = encode(image)
    photo = photo_decode(raw, 1)
    note = {
        "role": role, "visit": visit, "day": date, "page": 1,
        "caption": caption, "notes": "Synthetic re-framing demonstration photograph.",
    }
    source, _ = new_capture(db, window, photo, caption.lower().replace(" ", "-") + ".png",
                            note, synthetic=True)
    extent = output_plane(window, image.size)
    info = native_crop_registration(list(CORNERS), source, extent)
    info.update({
        "capture_id": source["id"], "capture_version": source["version"],
        "support_fraction": float(np.mean(np.asarray(image.getchannel("A")) >= 250)),
        "whole_photo_assumed": False,
    })
    align = db.add("alignments", info, parent=window["id"],
                   action="registered a framed photograph")
    db.link(align["id"], source, "registered photo")
    db.link(align["id"], window, "window dimensions")
    db.attach(align["id"], "plane", encode(image.resize(extent)), "image/png")
    return source, align


def attach_frame_mark(db, window, handle, transform, outline):
    title, kind, shape, points, holes = outline
    mark = db.add("marks", {
        "title": title, "category": kind, "shape": shape, "points": points, "holes": holes,
        "state": "observed",
        "notes": "Observation recorded against the original framed photograph.",
        "alignment": transform["id"], "source_capture": handle["id"],
        "coordinate_basis": "window", "plane_extent": transform["plane_extent"],
        "measure": {}, "retired": False,
    }, parent=window["id"], action="saved an observation on the old frame")
    db.link(mark["id"], handle, "marked photograph")
    db.link(mark["id"], transform, "marked plane")
    return mark


def attach_new_occlusion(db, window, handle, transform):
    x0, y0, x1, y1 = OCCLUSION_BOX
    points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    mark = db.add("marks", {
        "title": "New mount occlusion", "category": "occluded", "shape": "polygon",
        "points": points, "holes": [], "state": "observed",
        "notes": "Area covered by the replacement mount.",
        "alignment": transform["id"], "source_capture": handle["id"],
        "coordinate_basis": "window", "plane_extent": transform["plane_extent"],
        "measure": {}, "retired": False,
    }, parent=window["id"], action="recorded an occlusion on the new frame")
    db.link(mark["id"], handle, "marked photograph")
    db.link(mark["id"], transform, "marked plane")
    return mark


def frame_transfer_demo(library):
    from .catalog import window_fields
    with TRANSFER_LOCK, library.use(write=True) as db:
        existing = [item for item in db.list("windows", limit=80)
                    if item.get("code") == FRAME_SCHEME["code"]]
        if existing:
            return existing
        document = window_fields(FRAME_SCHEME) | {
            "synthetic": True, "working_extent": list(EXTENT),
            "notes": "Synthetic re-framing scenario: old and replacement frame apertures.",
        }
        window = db.add("windows", document, action="created synthetic re-framing window")
        old_photo, old_align = save_frame_photo(
            db, window, FRAME_SCHEME, OLD_APERTURE, False,
            "transmitted", "original framing visit", "2025-03-18",
            "Old frame transmitted photograph · Synthetic")
        new_photo, new_align = save_frame_photo(
            db, window, FRAME_SCHEME, NEW_APERTURE, True,
            "transmitted", "re-framing visit", "2026-10-02",
            "Replacement frame transmitted photograph · Synthetic")
        for outline in OLD_MARKS:
            attach_frame_mark(db, window, old_photo, old_align, outline)
        attach_new_occlusion(db, window, new_photo, new_align)
        return [db.public(window)]
