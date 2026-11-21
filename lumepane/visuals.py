"""Maps and charts for measurement evidence rather than image adjustment."""

from io import BytesIO
import math

import numpy as np
from PIL import Image, ImageDraw

from .imaging import canvas_picture, checker, fitted_image, font, label

INK = "#232a37"
PALE = "#f8f5e8"
PALE_EDGE = "#d8d7cc"
SKY = "#86bdcf"
AQUA = "#63b8ae"
ROSE = "#ed8b81"
GOLD = "#edc36f"
QUIET = "#626675"
VOID = (10, 14, 22)
PALETTE = {
    "unknown": (127, 129, 156),
    "measured": (79, 155, 161),
    "dark": (90, 115, 190),
    "clip": (248, 150, 95),
    "difference": (249, 149, 108),
    "excluded": (105, 103, 120),
    "weak": (197, 166, 128),
    "bright": (250, 205, 110),
}


def rgb_triplet(swatch):
    value = swatch.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def overlay_colors(surface, tags, categories, *, strength=.58):
    image = canvas_picture(surface)
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    for idx, color in enumerate(categories):
        if color is None:
            continue
        choose = surface.valid & (tags == idx)
        base[choose] = base[choose] * (1 - strength) + np.asarray(color) * strength
    return Image.fromarray(np.uint8(np.clip(base, 0, 255)), "RGB")


def color_ramp(values, low, high, first, last):
    t = np.clip((values - low) / max(high - low, 1e-9), 0, 1).astype(np.float32)
    return np.array(first)[None, None] * (1 - t[:, :, None]) + np.array(last)[None, None] * t[:, :, None]


def map_colors(surface, values, accepted, limits, color=(48, 181, 165), *, highlight=None, unknown=None):
    base = np.asarray(canvas_picture(surface).convert("RGB"), dtype=np.float32)
    mask = accepted & np.isfinite(values)
    ramp = color_ramp(values, limits[0], limits[1], (42, 63, 99), color)
    base[mask] = .40 * base[mask] + .60 * ramp[mask]
    if unknown is not None:
        yy, xx = np.ogrid[:accepted.shape[0], :accepted.shape[1]]
        hatch = unknown & ((xx + yy) % 7 < 2)
        base[unknown] *= .53
        base[hatch] = PALETTE["unknown"]
    if highlight is not None:
        base[highlight] = base[highlight] * .22 + np.asarray(PALETTE["clip"]) * .78
    base[~surface.valid] = VOID
    return Image.fromarray(np.uint8(np.clip(base, 0, 255)), "RGB")


def signed_map(surface, values, accepted, scale, unknown, outside=None, outside_color=(181, 132, 190)):
    normalized = values / max(1e-8, scale)
    base = np.asarray(canvas_picture(surface).convert("RGB"), dtype=np.float32)
    positive = accepted & (values >= 0)
    negative = accepted & ~positive
    weight = .22 + .55 * np.minimum(np.abs(normalized), 1.0)
    for zone, color in ((positive, (244, 163, 102)), (negative, (92, 147, 212))):
        base[zone] = base[zone] * (1 - weight[zone, None]) + np.asarray(color) * weight[zone, None]
    hatch = unknown & ((np.indices(values.shape).sum(axis=0) % 8) <= 2)
    base[unknown] *= .56
    base[hatch] = PALETTE["unknown"]
    if outside is not None and outside.any():
        lattice = np.indices(values.shape).sum(axis=0)
        cross_hatch = outside & ((lattice % 11 <= 2) | ((lattice + 6) % 11 <= 2))
        base[outside] *= .48
        base[cross_hatch] = outside_color
    return Image.fromarray(np.uint8(np.clip(base, 0, 255)), "RGB")


def legend_band(figure, title, labels, *, lines=2):
    width, height = figure.size
    reserved = 36 + 24 * lines
    page = Image.new("RGB", (width, height + reserved), (26, 33, 46))
    photo = figure.convert("RGBA")
    page.paste(photo, (0, reserved), photo.getchannel("A"))
    pen = ImageDraw.Draw(page)
    pen.text((12, 8), title, font=font(17, bold=True), fill="#fcf4dc")
    x, y = 13, 37
    for name, swatch in labels:
        if isinstance(swatch, tuple) and len(swatch) != 3:
            continue
        size = int(pen.textlength(name, font=font(13))) + 41
        if x + size > width - 7 and x > 13:
            x, y = 13, y + 22
        if y >= reserved - 8:
            break
        pen.rounded_rectangle((x, y, x + 15, y + 11), radius=2, fill=swatch)
        pen.text((x + 22, y - 2), name, font=font(13), fill="#dce0e7")
        x += size
    return page


def draw_bars(size, title, labels, values, colors):
    width, height = size
    paper = Image.new("RGB", size, PALE)
    pen = ImageDraw.Draw(paper)
    pen.text((22, 16), title, font=font(21, bold=True), fill=INK)
    interval = max(25, (height - 70) / max(1, len(values)))
    peak = max(values, default=0)
    left = 136
    for i, (name, value, color) in enumerate(zip(labels, values, colors, strict=True)):
        y = int(57 + i * interval)
        if y > height - 20:
            break
        pen.text((21, y - 1), name[:16], font=font(13), fill=INK)
        length = (width - left - 68) * value / max(peak, 1)
        if length > 0:
            pen.rounded_rectangle((left, y, left + length, y + 16), radius=3, fill=color)
        pen.text((left + length + 8, y), f"{value:,}", font=font(12), fill=QUIET)
    return paper


def proper_segments(samples):
    buffer = []
    for x, value in samples:
        if value is None:
            if buffer:
                yield buffer
                buffer = []
        else:
            buffer.append((float(x), float(value)))
    if buffer:
        yield buffer


def readable_axis(low, high):
    if high - low < 1e-9:
        low, high = low - .01, high + .01
    size = high - low
    base_power = 10 ** math.floor(math.log10(size / 5))
    normalized = size / (5 * base_power)
    interval = (1 if normalized < 1.5 else 2 if normalized < 3 else 5) * base_power
    lower = math.floor(low / interval) * interval
    upper = math.ceil(high / interval) * interval
    n = round((upper - lower) / interval)
    return lower, upper, [lower + i * interval for i in range(n + 1)]


def series_chart(channels, title, unit, *, width=790, height=275):
    paper = Image.new("RGB", (width, height), PALE)
    pen = ImageDraw.Draw(paper)
    pen.text((23, 14), title, font=font(18, bold=True), fill=INK)
    y_values = [point[1] for row in channels for point in row["samples"] if point[1] is not None]
    all_x = [point[0] for row in channels for point in row["samples"]]
    if not y_values or not all_x:
        pen.text((23, 60), "No measured stations", font=font(17), fill=QUIET)
        return paper
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y, lines = readable_axis(min(y_values), max(y_values))
    box = (64, 54, width - 31, height - 42)

    def map_point(x, y):
        xrat = (x - min_x) / max(1e-8, max_x - min_x)
        yrat = (y - min_y) / max(1e-8, max_y - min_y)
        return box[0] + xrat * (box[2] - box[0]), box[3] - yrat * (box[3] - box[1])

    for level in lines:
        _, y = map_point(min_x, level)
        pen.line((box[0], y, box[2], y), fill=PALE_EDGE)
        pen.text((13, y - 6), f"{level:.4g}", font=font(11), fill=QUIET)
    pen.rectangle(box, outline=PALE_EDGE)
    for row in channels:
        for segment in proper_segments(row["samples"]):
            line = [map_point(x, y) for x, y in segment]
            if len(line) >= 2:
                pen.line(line, fill=row["color"], width=2)
            elif line:
                x, y = line[0]
                pen.ellipse((x - 2, y - 2, x + 2, y + 2), fill=row["color"])
    pen.text((box[0], box[3] + 7), f"{min_x:.4g}", font=font(11), fill=QUIET)
    pen.text((box[2] - 43, box[3] + 7), f"{max_x:.4g} {unit}", font=font(11), fill=QUIET)
    return paper


def photo_board(title, panes, *, column_width=510, image_height=415):
    width = column_width * len(panes)
    paper = Image.new("RGB", (width, image_height + 101), (30, 36, 49))
    draw = ImageDraw.Draw(paper)
    draw.text((20, 14), title, fill="#f9edcf", font=font(22, bold=True))
    for i, (name, photo) in enumerate(panes):
        x = i * column_width
        paper.paste(fitted_image(photo, (column_width - 28, image_height - 12)), (x + 14, 65))
        draw.text((x + 18, 40), name, fill="#dce0e7", font=font(15, bold=True))
    return paper


def mark_bounds(image, bounds, number, shade=GOLD):
    pen = ImageDraw.Draw(image)
    p = [round(bounds[i] * image.size[i % 2]) for i in range(4)]
    pen.rounded_rectangle(p, radius=3, outline=shade, width=max(1, image.width // 420))
    label(pen, (p[0] + 2, p[1] + 2), str(number), size=12, fill=shade)
    return image


def png_reply_bytes(image):
    buffer = BytesIO()
    image.save(buffer, "PNG", optimize=False)
    return buffer.getvalue()


def masked_thumbnail(image, polygon):
    left, top, right, bottom = polygon.getbbox() or (0, 0, image.width, image.height)
    a = image.crop((left, top, right, bottom)).convert("RGBA")
    a.putalpha(polygon.crop((left, top, right, bottom)).convert("L"))
    out = checker(a.size)
    out.paste(a, mask=a.getchannel("A"))
    return out
