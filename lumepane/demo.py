"""Repeatable synthetic stained-glass panels for examining real image evidence."""

from dataclasses import dataclass
from threading import Lock

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .catalog import new_capture, window_fields
from .contracts import DeskError
from .geometry import CORNERS, native_crop_registration, output_plane
from .imaging import encode, image_rgb, linear_to_srgb, photo_decode

DEMO_LOCK = Lock()
EXTENT = 688, 498
DEMO_WINDOWS = (
    {"title": "Vesper · Thistle roundel", "code": "VP–01",
     "physical_mm": [440, 318.5], "location": "Synthetic north bay",
     "central_color": (24, 115, 135), "secondary_color": (187, 122, 43), "seed": 135850},
    {"title": "Aster · Shore light", "code": "VP–02",
     "physical_mm": [440, 318.5], "location": "Synthetic east bay",
     "central_color": (45, 148, 104), "secondary_color": (155, 53, 77), "seed": 8384},
)
ILLUMINATION_CASES = (
    ("transmitted", "initial light table", "2026-05-12"),
    ("transmitted", "late autumn check", "2026-11-11"),
    ("reflected", "initial light table", "2026-05-12"),
    ("rake_left", "late autumn check", "2026-11-11"),
    ("incident", "initial light table", "2026-05-12"),
    ("dark", "initial light table", "2026-05-12"),
)


@dataclass(frozen=True)
class ShapeBoard:
    palette: np.ndarray
    frame: np.ndarray
    painted: np.ndarray
    cracks: np.ndarray
    unstable: np.ndarray
    aperture: np.ndarray


def grid():
    w, h = EXTENT
    x, y = np.meshgrid(np.arange(w, dtype=np.float32) / w, np.arange(h, dtype=np.float32) / h)
    return x, y


def icon_palette(r, angle, x, y, scheme):
    focus = np.asarray(scheme["central_color"], dtype=np.float32) / 255
    alt = np.asarray(scheme["secondary_color"], dtype=np.float32) / 255
    accents = np.array([
        (252, 179, 80), (81, 126, 145), (186, 89, 57), (112, 174, 150),
        (226, 186, 90), (144, 83, 111), (247, 211, 130), (45, 104, 142),
    ], dtype=np.float32) / 255
    petals = ((angle * 8 / (2 * np.pi) + 16) % 8).astype(int)
    amber = (np.sin(6 * x + 13 * y) + 1) * .5
    field = alt * (.45 + .5 * amber[..., None]) + focus * (.35 + .16 * (1 - amber[..., None]))
    band = (r > .31) & (r < .48)
    field[band] = accents[((petals + 2) % 8)[band]]
    center = r < .297
    field[center] = focus
    core = r <= .102
    field[core] = alt
    highlights = (r > .134) & (r < .237) & (np.cos(6 * angle) > .16)
    field[highlights] = accents[((petals + 4) % 8)[highlights]]
    outer = (np.abs(x - .5) > .425) | (np.abs(y - .5) > .415)
    field[outer] = accents[1] * .7 + alt[None, :] * .26
    return np.clip(field, .04, .95)


def distance_to_polyline(x, y, vertices):
    best = np.full(x.shape, np.inf, dtype=np.float32)
    points = list(zip(vertices[:-1], vertices[1:]))
    for (ax, ay), (bx, by) in points:
        dx, dy = bx - ax, by - ay
        bracket = np.clip(((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy), 0, 1)
        distance = np.hypot(x - (ax + bracket * dx), y - (ay + bracket * dy))
        best = np.minimum(best, distance)
    return best


def panel_board(scheme):
    x, y = grid()
    radius = np.hypot((x - .5) * 1.22, y - .51)
    angle = np.arctan2(y - .51, (x - .5) * 1.22)
    color = icon_palette(radius, angle, x, y, scheme)
    seam = np.zeros(x.shape, dtype=bool)
    for boundary in (.104, .30, .49):
        seam |= np.abs(radius - boundary) < .007
    angular = np.abs(np.sin(4 * angle))
    seam |= (angular < .050) & (radius > .308) & (radius < .49)
    seam |= (np.abs(np.sin(3 * angle)) < .038) & (radius > .11) & (radius < .293)
    seam |= np.abs(x - .50) < .004
    seam |= np.abs(y - .51) < .005
    rim = (np.abs(x - .5) > .478) | (np.abs(y - .5) > .47)
    frame = rim | seam
    stroke = distance_to_polyline(x, y, [(0.56, .29), (.583, .34), (.616, .39), (.66, .45)])
    second = distance_to_polyline(x, y, [(.19, .55), (.23, .57), (.27, .615), (.295, .67), (.34, .682)])
    fissures = np.exp(-.5 * (np.minimum(stroke, second) / .0015) ** 2)
    paint = np.clip((np.sin(43 * x - 12 * y + 3 * np.sin(22 * y)) - .79) / .18, 0, 1)
    paint *= ((radius > .15) & (radius < .25))
    deposits = np.exp(-((x - .67) / .094) ** 2 - ((y - .64) / .064) ** 2)
    return ShapeBoard(color, frame, paint, fissures, deposits, np.ones(x.shape, dtype=bool))


def regular_patterns(scheme, eta=0.):
    x, y = grid()
    rnd = np.random.default_rng(scheme["seed"])
    noise = ndimage.gaussian_filter(rnd.normal(size=x.shape), .53).astype(np.float32)
    soft = ndimage.gaussian_filter(rnd.random(x.shape), 7).astype(np.float32) - .5
    flow = .012 * np.sin(130 * x + 21 * y + eta * 1.7) + .011 * np.cos(190 * y - 35 * x)
    bg = .72 + .098 * np.cos(np.pi * (x - .43)) - .048 * np.cos(np.pi * (y - .76))
    return bg, flow + .035 * noise + .35 * soft


def scene(scheme, role, visit):
    x, y = grid()
    optics = panel_board(scheme)
    base, grain = regular_patterns(scheme, eta=float("autumn" in visit))
    frame = optics.frame
    area = optics.aperture.copy()
    if role == "dark":
        z = .025 + .06 * np.abs(grain)
        image = np.stack([z, z, z], axis=-1)
    elif role == "incident":
        z = .91 + .032 * np.sin(4 * x) - .033 * np.cos(2 * y)
        image = np.stack([z, z * .98, z * .95], axis=-1)
    else:
        if role == "transmitted":
            color = optics.palette * base[:, :, None]
            image = color + grain[:, :, None] * .65
            image -= (.23 * optics.cracks + .18 * optics.painted + .085 * optics.unstable)[:, :, None]
            image[frame] = (.035, .037, .043)
            if "autumn" in visit:
                sheen = .08 * np.exp(-((x - .69) / .068) ** 2 - ((y - .64) / .074) ** 2)
                image += sheen[..., None]
                image += .03 * (x - .35)[..., None]
                hidden = (x > .875) & (y > .52)
                area[hidden] = False
        elif role == "reflected":
            image = optics.palette * .23 + .21 + grain[:, :, None] * .58
            image += (.15 * optics.painted - .10 * optics.cracks)[..., None]
            image[frame] = (.21, .23, .23)
        else:
            light = .65 if role == "rake_left" else .76
            image = optics.palette * .20 + light * base[..., None] / 2 + grain[..., None]
            image += (.27 * optics.cracks + .13 * optics.painted + .13 * optics.unstable)[..., None]
            image[frame] = (.28, .41, .43)
    clipped = np.clip(image, .006, 1.0)
    return image_rgb(clipped, area, from_linear=False)


def save_demo_photo(db, window, scheme, role, visit, date):
    image = scene(scheme, role, visit)
    raw = encode(image)
    photo = photo_decode(raw, 1)
    names = {
        "transmitted": "Transmitted panel",
        "reflected": "Reflected panel",
        "rake_left": "Light grazing the panel",
        "incident": "Open light-table reference",
        "dark": "Dark exposure reference",
    }
    note = {"role": role, "visit": visit, "day": date, "page": 1, "caption": names[role] + " · Synthetic",
            "notes": "Reproducible synthetic demonstration captured in a known panel plane."}
    source, _ = new_capture(db, window, photo, names[role].lower().replace(" ", "-") + ".png", note, synthetic=True)
    if role not in ("incident", "dark"):
        extent = output_plane(window, image.size)
        info = native_crop_registration(list(CORNERS), source, extent)
        info.update({
            "capture_id": source["id"], "capture_version": source["version"],
            "support_fraction": float(np.mean(np.asarray(image.getchannel("A")) >= 250)),
            "whole_photo_assumed": False,
        })
        align = db.add("alignments", info, parent=window["id"], action="registered the synthetic photo")
        db.link(align["id"], source, "registered photo")
        db.link(align["id"], window, "window dimensions")
        db.attach(align["id"], "plane", encode(image.resize(extent)), "image/png")
    else:
        align = None
    return source, align


def attach_demo_mark(db, window, handle, transform, outline):
    title, kind, shape, points, holes = outline
    mark = db.add("marks", {
        "title": title, "category": kind, "shape": shape, "points": points, "holes": holes,
        "state": "observed", "notes": "Synthetic labels added at the known drawn coordinates.",
        "alignment": transform["id"], "source_capture": handle["id"], "coordinate_basis": "window",
        "plane_extent": transform["plane_extent"], "measure": {}, "retired": False,
    }, parent=window["id"], action="saved a synthetic source annotation")
    db.link(mark["id"], handle, "marked photograph")
    db.link(mark["id"], transform, "marked plane")
    return mark


def insert_demo_marks(db, window, handle, transform):
    data = [
        ("Left ornamental pane", "pane", "polygon", [[.12,.16],[.44,.16],[.43,.46],[.14,.46]], []),
        ("Right artisan pane", "pane", "polygon", [[.55,.16],[.84,.17],[.84,.45],[.55,.45]], []),
        ("Lower archived pane", "pane", "polygon", [[.12,.61],[.43,.6],[.44,.82],[.13,.83]], [[[.19,.65],[.22,.65],[.22,.68],[.19,.68]]]),
        ("East central came", "came", "line", [[.50,.13],[.50,.36],[.50,.55],[.50,.87]], []),
        ("Cross-bar", "came", "line", [[.09,.51],[.30,.51],[.59,.51],[.92,.51]], []),
        ("Observed short fissure", "fracture", "line", [[.56,.29],[.583,.34],[.616,.39],[.66,.45]], []),
        ("Locally masked catchlight", "occluded", "polygon", [[.44,.20],[.49,.20],[.49,.31],[.44,.31]], []),
        ("Stable center reference", "reference", "polygon", [[.37,.33],[.417,.331],[.417,.39],[.37,.39]], []),
    ]
    return [attach_demo_mark(db, window, handle, transform, row) for row in data]


def demo_library(library):
    with DEMO_LOCK, library.use(write=True) as query:
        exists = [item for item in query.list("windows", limit=80) if item["synthetic"]]
        if exists:
            return exists
        made = []
        for model in DEMO_WINDOWS:
            document = window_fields(model) | {"synthetic": True, "working_extent": list(EXTENT),
                                                "notes": "Procedurally rendered condition and illumination; no real window is shown."}
            window = query.add("windows", document, action="created synthetic training window")
            roles = [save_demo_photo(query, window, model, role, visit, date)
                     for role, visit, date in ILLUMINATION_CASES]
            insert_demo_marks(query, window, *roles[0])
            made.append(query.public(window))
        return made


def showcase():
    image = scene(DEMO_WINDOWS[0], "transmitted", "initial light table")
    return encode(image)
