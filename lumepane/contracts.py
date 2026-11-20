"""Validation primitives shared by the image-facing entry points."""

import math
import re
from collections.abc import Mapping
from datetime import datetime, timezone

ID_PATTERN = re.compile(r"^[0-9a-f]{24}$")
SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_FRAME_EDGE = 4500
MAX_UPLOAD_PIXELS = 12_000_000
MAX_IMAGE_BYTES = 28 * 1024**2
MAX_WORK_EDGE = 1800
MAX_POLYGON_VERTICES = 80
MAX_REPORT_RECORDS = 24

ROLES = {
    "transmitted": "Transmitted light",
    "reflected": "Reflected front",
    "rake_left": "Raking from left",
    "rake_right": "Raking from right",
    "incident": "Incident light reference",
    "dark": "Dark reference",
}
MARK_KINDS = {
    "pane": "Glass piece",
    "came": "Came or solder",
    "fracture": "Observed fracture",
    "paint": "Paint detail",
    "occluded": "Occluded",
    "ignore": "Exclude from measurement",
    "reference": "Reference patch",
}
OBSERVATION_STATES = {
    "observed": "Observed",
    "review": "Needs review",
    "resolved": "Resolved",
}


class DeskError(ValueError):
    """An operator action could not be applied as requested."""

    def __init__(self, message, *, field=None, status=422):
        super().__init__(message)
        self.field = field
        self.status = status


class Missing(DeskError):
    def __init__(self, message):
        super().__init__(message, status=404)


class Stale(DeskError):
    def __init__(self, message="The item changed. Reopen it before saving."):
        super().__init__(message, status=409)


class Capacity(DeskError):
    def __init__(self, message, *, field=None):
        super().__init__(message, field=field, status=413)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def obj(data, label="request"):
    if not isinstance(data, Mapping):
        raise DeskError(f"Expected an object for {label}.", field=label)
    return data


def word(value, label, *, default=None, limit=120, multiline=False):
    if value is None and default is not None:
        value = default
    if not isinstance(value, str):
        raise DeskError(f"Enter text for {label}.", field=label)
    result = value.strip()
    allowed = {"\n", "\r", "\t"} if multiline else set()
    if any(not (c.isprintable() or c in allowed) for c in result):
        raise DeskError(f"Remove control characters from {label}.", field=label)
    if len(result) > limit or not result and default is None:
        raise DeskError(f"Enter between 1 and {limit} characters for {label}.", field=label)
    return result


def enum(value, label, choices, *, default=None):
    if value is None:
        value = default
    if not isinstance(value, str) or value not in choices:
        raise DeskError(f"Choose a valid {label}.", field=label)
    return value


def number(value, label, low, high, *, default=None):
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeskError(f"Enter a number for {label}.", field=label)
    try:
        cast = float(value)
    except OverflowError as error:
        raise DeskError(f"{label} is too large.", field=label) from error
    if not math.isfinite(cast) or not low <= cast <= high:
        raise DeskError(f"{label} must be between {low} and {high}.", field=label)
    return cast


def integer(value, label, low, high, *, default=None):
    if value is None:
        value = default
    if type(value) is not int or not low <= value <= high:
        raise DeskError(f"{label} must be a whole number between {low} and {high}.", field=label)
    return value


def flag(value, label, *, default=False):
    if value is None:
        return default
    if type(value) is not bool:
        raise DeskError(f"{label} must be on or off.", field=label)
    return value


def identifier(value, label="record"):
    if not isinstance(value, str) or ID_PATTERN.fullmatch(value) is None:
        raise DeskError(f"Choose a valid {label}.", field=label)
    return value


def digest(value, label):
    if not isinstance(value, str) or SHA_PATTERN.fullmatch(value) is None:
        raise DeskError(f"Invalid {label}.", field=label)
    return value


def collection(value, label, low=0, high=100):
    if not isinstance(value, (tuple, list)) or not low <= len(value) <= high:
        raise DeskError(f"Provide {low} to {high} values for {label}.", field=label)
    return value


def identifiers(value, label, *, low=1, high=MAX_REPORT_RECORDS):
    result = [identifier(k, label) for k in collection(value, label, low, high)]
    if len(set(result)) != len(result):
        raise DeskError(f"Choose each {label} only once.", field=label)
    return result


def point(value, label="point", *, low=0.0, high=1.0):
    x, y = collection(value, label, 2, 2)
    return [number(x, f"{label} x", low, high), number(y, f"{label} y", low, high)]


def polygon(value, label="outline", *, minimum=3):
    return [point(item, label) for item in collection(value, label, minimum, MAX_POLYGON_VERTICES)]


def region_box(value, label="area", *, fallback=None):
    if value is None:
        return fallback
    x0, y0, x1, y1 = [number(v, label, 0, 1) for v in collection(value, label, 4, 4)]
    if x1 - x0 < 0.001 or y1 - y0 < 0.001:
        raise DeskError(f"{label} must have positive width and height.", field=label)
    return [x0, y0, x1, y1]


def dimensions(value, label="dimensions", *, low=1, high=12000):
    w, h = collection(value, label, 2, 2)
    return [number(w, f"{label} width", low, high), number(h, f"{label} height", low, high)]


def window_setting(payload, *, default=None):
    data = obj(payload, "display")
    if not data:
        return default
    lower = number(data.get("low"), "display low", 0, 1, default=0)
    upper = number(data.get("high"), "display high", 0, 1, default=1)
    if upper <= lower + 0.005:
        raise DeskError("Set a wider display window.", field="display")
    return {
        "low": lower,
        "high": upper,
        "gamma": number(data.get("gamma"), "display gamma", 0.2, 4, default=1),
        "grayscale": flag(data.get("grayscale"), "grayscale"),
    }


def iso_day(value, label):
    if value is None or value == "":
        return None
    string = word(value, label, limit=10)
    try:
        parsed = datetime.strptime(string, "%Y-%m-%d").date()
    except ValueError as error:
        raise DeskError("Use a calendar date as YYYY-MM-DD.", field=label) from error
    return parsed.isoformat()


def one_page(payload, limit=24):
    data = obj(payload)
    return integer(data.get("page"), "page", 1, limit, default=1)


def revision(payload):
    return integer(obj(payload).get("version"), "version", 1, 2_000_000_000)


def support_limit(requested, occupied, label, *, minimum):
    if requested <= 0 or occupied < minimum:
        raise DeskError(f"{label} needs at least {minimum} usable image samples.")
    return occupied / requested
