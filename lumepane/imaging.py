"""Photograph normalization with preserved original bytes and support."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import warnings

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from scipy import ndimage

from .contracts import Capacity, DeskError, MAX_FRAME_EDGE, MAX_IMAGE_BYTES, MAX_UPLOAD_PIXELS, MAX_WORK_EDGE

Image.MAX_IMAGE_PIXELS = MAX_UPLOAD_PIXELS
THUMB_EDGE = 420
CLIP_BRIGHT = 252 / 255
CLIP_DARK = 5 / 255
RGBA_BACKGROUND = (26, 31, 40)
GOOD_PNG_MODES = {"RGB", "RGBA", "L", "LA", "P"}


@dataclass(frozen=True)
class DraftCapture:
    raw: bytes
    picture: Image.Image
    media_type: str
    metadata: dict
    warnings: tuple[str, ...]

    @property
    def digest(self):
        return sha256(self.raw).hexdigest()


@dataclass(frozen=True)
class Surface:
    frame: dict
    linear: np.ndarray
    valid: np.ndarray
    saturated: np.ndarray
    dark: np.ndarray

    @property
    def extent(self):
        return self.linear.shape[1], self.linear.shape[0]

    def subset(self, box):
        x0, y0, x1, y1 = box
        section = np.s_[y0:y1, x0:x1]
        return Surface(
            self.frame,
            self.linear[section],
            self.valid[section],
            self.saturated[section],
            self.dark[section],
        )


def safe_filename(name):
    cleaned = (name or "capture").replace("\\", "/").rsplit("/", 1)[-1]
    stripped = "".join(c for c in cleaned if c.isprintable() and c not in "<>:\"|?*")
    return stripped.strip()[:120] or "capture"


def read_upload(file):
    if file is None or not file.filename:
        raise DeskError("Choose a local image to import.", field="image")
    data = file.stream.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise DeskError("The selected file is empty.", field="image")
    if len(data) > MAX_IMAGE_BYTES:
        raise Capacity("Use an image up to 28 MiB.", field="image")
    return data, safe_filename(file.filename)


def encode(image, *, format="PNG", **settings):
    data = BytesIO()
    image.save(data, format=format, **settings)
    return data.getvalue()


def opened_picture(raw):
    with Image.open(BytesIO(raw)) as p:
        return p.copy()


def capture_tags(pic):
    exif = pic.getexif()
    fields = (
        (274, "orientation"),
        (36867, "camera_time"),
        (271, "make"),
        (272, "model"),
        (33434, "seconds"),
        (33437, "aperture"),
        (34855, "iso"),
    )
    out = {}
    for tag, key in fields:
        value = exif.get(tag)
        if isinstance(value, str):
            out[key] = "".join(x for x in value if x.isprintable())[:80]
        elif isinstance(value, int):
            out[key] = value
        elif value is not None and hasattr(value, "numerator") and hasattr(value, "denominator"):
            if value.denominator:
                out[key] = float(value)
    return out


def preview_inches(image, dpi):
    return [round(image.width / dpi, 4), round(image.height / dpi, 4)]


def rgba_depth_frame(image):
    if image.mode in ("I;16", "I;16L", "I;16B", "I"):
        value = np.asarray(image, dtype=np.float64)
        level = np.rint(np.clip(value, 0, 65535) * (255 / 65535)).astype(np.uint8)
        return Image.fromarray(level, "L").convert("RGBA"), "16-bit grayscale preview"
    if image.mode in ("F", "1"):
        mono = image.convert("L")
        return mono.convert("RGBA"), "converted monochrome preview"
    shades = image.convert("RGBA")
    return shades, "8-bit working sRGB"


def convert_to_srgb(image, rgba):
    embedded = image.info.get("icc_profile")
    if not embedded:
        return rgba, "assumed sRGB"
    try:
        source = ImageCms.getOpenProfile(BytesIO(embedded))
        destination = ImageCms.createProfile("sRGB")
        if image.mode in ("RGB", "RGBA", "CMYK", "L", "LA"):
            common = image.mode
            if common == "RGBA":
                common = "RGB"
            elif common == "LA":
                common = "L"
        else:
            common = "RGB"
        out = ImageCms.profileToProfile(image.convert(common), source, destination, outputMode="RGB")
    except (OSError, ValueError, TypeError, ImageCms.PyCMSError):
        return rgba, "embedded profile not usable"
    rgb = out.convert("RGBA")
    rgb.putalpha(rgba.getchannel("A"))
    return rgb, "embedded profile converted to sRGB"


def resize_rgba(image, edge):
    if max(image.size) <= edge:
        return image.copy()
    ratio = edge / max(image.size)
    shape = tuple(max(1, int(round(x * ratio))) for x in image.size)
    return image.resize(shape, Image.Resampling.LANCZOS)


def photo_decode(raw, page):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as file:
                fmt = file.format
                if fmt not in {"JPEG", "PNG", "TIFF", "WEBP"}:
                    raise DeskError("Use PNG, JPEG, TIFF, or WebP.")
                count = getattr(file, "n_frames", 1)
                if page > count:
                    raise DeskError(f"This file has {count} frame(s).", field="page")
                file.seek(page - 1)
                if max(file.size) > MAX_FRAME_EDGE or file.width * file.height > MAX_UPLOAD_PIXELS:
                    raise Capacity("The selected page is larger than the supported photo frame.")
                if min(file.size) < 16:
                    raise DeskError("Use a photo of at least 16 pixels on each side.")
                source_size = list(file.size)
                exif = capture_tags(file)
                oriented = ImageOps.exif_transpose(file)
                rgba, depth = rgba_depth_frame(oriented)
                rgba, space = convert_to_srgb(oriented, rgba)
                working = resize_rgba(rgba, MAX_WORK_EDGE)
    except (UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning,
            SyntaxError, OSError, EOFError) as error:
        raise DeskError("The selected image page could not be decoded.", field="image") from error
    alpha = np.asarray(working.getchannel("A"))
    coverage = round(float(np.mean(alpha >= 250)), 6)
    if not coverage:
        raise DeskError("The photo contains no visible pixels.", field="image")
    conditions = []
    if list(working.size) != list(rgba.size):
        conditions.append("Reduced only for the analysis plane. The original is retained.")
    if space == "embedded profile not usable":
        conditions.append("Profile could not be applied. The working colors use the stored RGB.")
    if depth != "8-bit working sRGB":
        conditions.append("The working view has been normalized to 8-bit; the original is retained.")
    mime = Image.MIME.get(fmt, "application/octet-stream")
    meta = {
        "encoding": fmt,
        "source_size": source_size,
        "upright_size": list(oriented.size),
        "extent": list(working.size),
        "analysis_space": space,
        "working_depth": depth,
        "camera": exif,
        "page": page,
        "page_count": count,
        "usable_alpha_fraction": coverage,
    }
    return DraftCapture(raw, working, mime, meta, tuple(conditions))


def srgb_to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(rgb):
    value = np.clip(rgb, 0, 1)
    return np.where(value <= 0.0031308, value * 12.92, 1.055 * value ** (1 / 2.4) - 0.055)


def luminance(rgb):
    return rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def surface_from_picture(record, image):
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    valid = alpha >= 0.98
    high_channel = np.max(rgb, axis=2)
    positive = srgb_to_linear(rgb)
    positive[~valid] = 0
    return Surface(record, positive, valid, valid & (high_channel >= CLIP_BRIGHT), valid & (high_channel <= CLIP_DARK))


def open_surface(query, capture):
    return surface_from_picture(capture, opened_picture(query.media(capture["id"], "working")))


def image_rgb(values, alpha=None, *, from_linear=False):
    values = np.asarray(values)
    if from_linear:
        values = linear_to_srgb(values)
    integer = np.rint(np.clip(values, 0, 1) * 255.0).astype(np.uint8)
    if integer.ndim == 2:
        integer = np.repeat(integer[:, :, None], 3, axis=2)
    if alpha is None:
        return Image.fromarray(integer, "RGB")
    alpha_bytes = np.rint(np.clip(alpha, 0, 1) * 255.0).astype(np.uint8)
    rgba = np.dstack((integer, alpha_bytes))
    return Image.fromarray(rgba, "RGBA")


def canvas_picture(surface):
    return image_rgb(surface.linear, surface.valid, from_linear=True)


def srgb_display(surface, settings):
    if not settings:
        return canvas_picture(surface)
    source = luminance(surface.linear) if settings["grayscale"] else surface.linear
    view = linear_to_srgb(source)
    zones = np.clip((view - settings["low"]) / (settings["high"] - settings["low"]), 0, 1)
    return image_rgb(zones ** (1 / settings["gamma"]), surface.valid)


def marks_display(image, marks):
    overlay = image.convert("RGBA")
    pen = ImageDraw.Draw(overlay)
    width, height = image.size
    colors = {
        "pane": "#ffe49b", "came": "#a7d8ea", "fracture": "#ed9f87",
        "paint": "#efb6ef", "occluded": "#c3c8e1", "ignore": "#c6cbd4", "reference": "#bfe9b0",
    }
    for mark in marks:
        points = [(min(width - 1, x * width), min(height - 1, y * height)) for x, y in mark["points"]]
        if mark["shape"] == "polygon":
            points.append(points[0])
        color = colors[mark["category"]]
        pen.line(points, fill=color, width=max(1, round(max(width, height) / 410)), joint="curve")
        x, y = points[0]
        label(pen, (x + 3, y + 3), mark["title"], fill=color, size=13, limit=23)
    return overlay


def checker(size, *, tile=10):
    width, height = size
    y, x = np.ogrid[:height, :width]
    weights = (x // tile + y // tile) % 2
    slate = np.array([29, 36, 46], dtype=np.uint8)
    blue = np.array([47, 56, 69], dtype=np.uint8)
    return Image.fromarray(np.where(weights[..., None], slate, blue), "RGB")


def display_support(image, accepted=None):
    image = image.convert("RGBA")
    if accepted is not None:
        actual = np.asarray(image.getchannel("A"), dtype=np.uint8)
        limit = np.where(accepted, actual, 0).astype(np.uint8)
        image.putalpha(Image.fromarray(limit, "L"))
    board = checker(image.size)
    board.paste(image, mask=image.getchannel("A"))
    return board


def font(size, *, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default(size=size)


def label(draw, pos, text, *, fill="#f4f4ed", background="#222a36", size=14, limit=68):
    text = str(text)
    if len(text) > limit:
        text = text[:limit - 1] + "…"
    face = font(size, bold=True)
    width = draw.textlength(text, font=face)
    x, y = (round(p) for p in pos)
    draw.rounded_rectangle((x - 3, y - 1, x + width + 6, y + size + 5), radius=4, fill=background)
    draw.text((x, y), text, font=face, fill=fill)


def percentile(values, probabilities=(0.05, 0.5, 0.95)):
    if not len(values):
        return [None] * len(probabilities)
    return [round(float(i), 6) for i in np.quantile(values, probabilities)]


def grid_bounds(width, height, columns, rows):
    x = np.rint(np.linspace(0, width, columns + 1)).astype(int)
    y = np.rint(np.linspace(0, height, rows + 1)).astype(int)
    for row in range(rows):
        for column in range(columns):
            yield row, column, (int(x[column]), int(y[row]), int(x[column + 1]), int(y[row + 1]))


def masked_gaussian(values, valid, sigma, *, support_floor=.66):
    float_mask = valid.astype(np.float32)
    weight = ndimage.gaussian_filter(float_mask, sigma=sigma, mode="constant")
    filtered = ndimage.gaussian_filter(np.where(valid, values, 0), sigma=sigma, mode="constant")
    out = np.divide(filtered, weight, out=np.zeros_like(filtered), where=weight > 1e-6)
    return out, valid & (weight >= support_floor)


def gradient(values, valid):
    north = np.pad(valid[:-1], ((1, 0), (0, 0)))
    south = np.pad(valid[1:], ((0, 1), (0, 0)))
    west = np.pad(valid[:, :-1], ((0, 0), (1, 0)))
    east = np.pad(valid[:, 1:], ((0, 0), (0, 1)))
    accessible = valid & north & south & west & east
    gy, gx = np.gradient(values)
    return gx, gy, accessible


def fitted_image(image, size):
    small = image.copy()
    small.thumbnail(size, Image.Resampling.LANCZOS)
    base = checker(size, tile=12)
    placement = ((size[0] - small.width) // 2, (size[1] - small.height) // 2)
    if small.mode == "RGBA":
        base.paste(small, placement, small.getchannel("A"))
    else:
        base.paste(small, placement)
    return base
