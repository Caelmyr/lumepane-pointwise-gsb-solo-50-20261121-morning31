"""Surface grain and patterned glass as a recorded image property."""

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, signal

from .contracts import DeskError, enum, integer, number, obj, support_limit
from .geometry import analysis_box, bounds_px
from .imaging import font, gradient, luminance, masked_gaussian, percentile
from .visuals import legend_band, map_colors, series_chart


@dataclass(frozen=True)
class GrainSettings:
    box: list | None
    background: float
    min_repeat: float
    max_repeat: float
    direction: str

    @classmethod
    def parse(cls, raw):
        wanted = obj(raw)
        small = number(wanted.get("minimum_repeat"), "shortest spacing", 2.5, 20, default=3.0)
        wide = number(wanted.get("maximum_repeat"), "longest spacing", 6., 130, default=45.)
        if wide <= small + 1:
            raise DeskError("Use a wider range of permitted spacings.")
        return cls(
            analysis_box(wanted),
            number(wanted.get("background"), "background smoothing", 2., 50., default=12.0),
            small, wide, enum(wanted.get("grain_mode"), "direction mode", ("average", "principal"), default="principal"),
        )

    def json(self):
        return {
            "box": self.box,
            "background": self.background, "minimum_repeat": self.min_repeat,
            "maximum_repeat": self.max_repeat, "grain_mode": self.direction,
        }


def weighted_spectrum(intensity, mask):
    height, width = intensity.shape
    if min(height, width) < 24:
        raise DeskError("Choose a grain region at least 24 pixels across each edge.")
    support_limit(mask.size, int(mask.sum()), "Texture analysis", minimum=144)
    weight = np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)
    effective = weight * mask
    center = float(np.sum(intensity * effective) / max(effective.sum(), .0001))
    filtered = (intensity - center) * effective
    power = np.abs(np.fft.fftshift(np.fft.fft2(filtered))) ** 2
    kx = np.fft.fftshift(np.fft.fftfreq(width))
    ky = np.fft.fftshift(np.fft.fftfreq(height))
    u, v = np.meshgrid(kx, ky)
    return power.astype(np.float32), u, v


def axis_hint(values, support):
    dx, dy, valid = gradient(values, support)
    if np.count_nonzero(valid) < 36:
        return None
    a = float(np.mean(dx[valid] ** 2))
    b = float(np.mean(dy[valid] ** 2))
    c = float(np.mean(dx[valid] * dy[valid]))
    energy = a + b
    if energy < 1e-7:
        return None
    length = math.hypot(a - b, 2 * c)
    normal = math.degrees(.5 * math.atan2(2 * c, a - b)) % 180
    return {
        "ridge_axis_degrees": round((normal + 90) % 180, 4),
        "across_axis_degrees": round(normal, 4),
        "coherence": round(length / energy, 5),
        "energy": round(energy, 8),
    }


def spectrum_buckets(power, u, v, repeat_low, repeat_high, orientation):
    frequencies = np.hypot(u, v)
    supported = (frequencies >= 1 / repeat_high) & (frequencies <= 1 / repeat_low)
    if orientation:
        theta = np.arctan2(v, u) * 180 / math.pi
        discrepancy = np.abs((theta - orientation["across_axis_degrees"] + 90) % 180 - 90)
        supported &= discrepancy <= 22
    freq_edges = np.linspace(1.0 / repeat_high, 1.0 / repeat_low, 46)
    bin_id = np.searchsorted(freq_edges, frequencies, side="right") - 1
    values = np.zeros(len(freq_edges) - 1)
    counts = np.zeros(values.shape, dtype=int)
    for i in range(len(values)):
        eligible = (bin_id == i) & supported
        if np.any(eligible):
            values[i] = np.mean(power[eligible])
            counts[i] = int(np.count_nonzero(eligible))
    frequency = (freq_edges[:-1] + freq_edges[1:]) / 2
    return values, counts, frequency


def repeat_peaks(values, freq, counts, direction, scale):
    allowed = counts > 0
    if not np.any(allowed):
        return [], []
    ceiling = max(.0001, float(np.max(values[allowed])))
    relative = values / ceiling
    peaks, details = signal.find_peaks(relative, prominence=.08, distance=2)
    ordered = sorted(peaks, key=lambda index: -relative[index])
    results = []
    for i in ordered[:6]:
        if not allowed[i]:
            continue
        pixels = float(1 / freq[i])
        normal = None
        if direction:
            radians = math.radians(direction["across_axis_degrees"])
            normal = np.array([math.cos(radians), math.sin(radians)])
        converted = pixels * float(np.linalg.norm(scale * normal)) if scale is not None and normal is not None else None
        results.append({"spacing_px": round(pixels, 5), "spacing_mm_along_normal": round(converted, 5) if converted else None,
                        "relative_power": round(float(relative[i]), 6), "sampled_frequencies": int(counts[i])})
    curve = [
        [round(float(1 / f), 6), float(round(v, 6)) if ok else None]
        for f, v, ok in reversed(list(zip(freq, relative, allowed, strict=True)))
    ]
    return results, curve


def texture_report(ctx, raw):
    settings = GrainSettings.parse(raw)
    surface = ctx.surface
    extent = surface.extent
    x0, y0, x1, y1 = bounds_px(settings.box, extent)
    if (x1 - x0) * (y1 - y0) > 2_000_000:
        raise DeskError("Choose a smaller analysis region for the texture spectrum.")
    patch = np.s_[y0:y1, x0:x1]
    linear = luminance(surface.linear)
    eligible = ctx.usable & ~surface.saturated & ~surface.dark
    background, measured = masked_gaussian(linear, eligible, settings.background)
    grain = linear - background
    safe_values = grain[patch]
    mask = measured[patch]
    direction = axis_hint(safe_values, mask)
    if settings.direction == "average" or direction and direction["coherence"] < .22:
        directional_filter = None
    else:
        directional_filter = direction
    spectrum, u, v = weighted_spectrum(safe_values, mask)
    values, counts, freq = spectrum_buckets(spectrum, u, v, settings.min_repeat, settings.max_repeat, directional_filter)
    px_scale = ctx.scale if ctx.alignment else None
    peaks, curve = repeat_peaks(values, freq, counts, directional_filter, px_scale)
    chart = series_chart([{"samples": curve, "color": "#2d736a"}], "Image repeat strength", "px")
    sigma = max(.001, float(np.quantile(np.abs(safe_values[mask]), .92)))
    total = np.zeros(eligible.shape, dtype=bool)
    total[patch] = True
    art = map_colors(surface, np.abs(grain), measured & total, (0, sigma), unknown=total & ~measured)
    report = {
        "settings": settings.json(), "support_pixels": int(mask.sum()),
        "orientation": direction, "peaks": peaks, "spectrum": curve,
        "residual_contrast": percentile(np.abs(safe_values[mask])),
        "note": "Strong repeated photograph patterns are candidates. Multiple peaks or directional ambiguity should be checked in the source.",
    }
    return report, {"map": legend_band(art, "Image grain above smooth illumination", [
        ("Local contrast", (48, 181, 165)), ("Unavailable", (127, 129, 156)),
    ]), "chart": chart}
