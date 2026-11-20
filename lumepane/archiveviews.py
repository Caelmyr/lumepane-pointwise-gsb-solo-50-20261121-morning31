"""Contact sheets, scaled inspection strips and image source snapshots."""

from dataclasses import dataclass
from io import BytesIO
import math
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

from .contracts import DeskError, enum, flag, integer, number, obj, word
from .imaging import checker, fitted_image, font, label
from .visuals import INK, PALE, QUIET

PAPERS = {"a4": (210, 297), "letter": (215.9, 279.4), "a3": (297, 420)}


@dataclass(frozen=True)
class Paper:
    name: str
    landscape: bool
    resolution: int
    margin_mm: float

    @classmethod
    def parse(cls, raw):
        form = obj(raw, "paper")
        return cls(
            enum(form.get("paper"), "paper", PAPERS, default="a4"),
            flag(form.get("landscape"), "landscape"),
            integer(form.get("dpi"), "resolution", 96, 200, default=144),
            number(form.get("margin"), "paper margin", 7, 30, default=12),
        )

    @property
    def dims(self):
        width, height = PAPERS[self.name]
        return (height, width) if self.landscape else (width, height)

    @property
    def extent(self):
        return tuple(self.pixels(d) for d in self.dims)

    def pixels(self, millimeters):
        return int(round(self.resolution * millimeters / 25.4))

    def manifest(self):
        return {"paper": self.name, "size_mm": self.dims, "dpi": self.resolution, "margin_mm": self.margin_mm,
                "orientation": "landscape" if self.landscape else "portrait"}


def paginate(collection, size):
    for start in range(0, len(collection), size):
        yield start, collection[start:start + size]


def annual_header(draw, paper, title, subtitle, index, pages):
    start = paper.pixels(paper.margin_mm)
    width, height = paper.extent
    draw.text((start, start), title[:68], fill=INK, font=font(paper.pixels(4.9), bold=True))
    draw.text((start, start + paper.pixels(7.6)), subtitle[:97], fill=QUIET, font=font(paper.pixels(2.9)))
    draw.line((start, start + paper.pixels(14), width - start, start + paper.pixels(14)), fill="#9d9da5")
    draw.text((start, height - start - paper.pixels(4.6)), f"Page {index} of {pages}", fill=QUIET, font=font(paper.pixels(2.9)))


def subplot_grid(paper, columns, rows):
    width, height = paper.extent
    left = paper.pixels(paper.margin_mm)
    top = left + paper.pixels(18)
    bottom = height - left - paper.pixels(8)
    right = width - left
    gap = paper.pixels(3)
    cell_w = (right - left - max(0, columns - 1) * gap) // columns
    cell_h = (bottom - top - max(0, rows - 1) * gap) // rows
    if min(cell_w, cell_h) < 50:
        raise DeskError("Use fewer rows or columns so each page has readable images.")
    for y in range(rows):
        for x in range(columns):
            yield left + x * (gap + cell_w), top + y * (gap + cell_h), cell_w, cell_h


def context_lines(note, drawer, width, face, lines=2):
    current = []
    words = str(note or "").split()
    result = []
    for word in words:
        if drawer.textlength(" ".join(current + [word]), font=face) <= width:
            current.append(word)
        elif not current:
            letters = ""
            for character in word:
                if drawer.textlength(letters + character + "…", font=face) > width:
                    break
                letters += character
            result.append(letters + "…")
        else:
            result.append(" ".join(current))
            current = [word]
        if len(result) == lines:
            return result
    if current:
        result.append(" ".join(current))
    return result[:lines]


def contact_card(picture, paper, image_cell, caption, note, index):
    x, y, w, h = image_cell
    pad = max(7, paper.pixels(2))
    labels = paper.pixels(12)
    board = fitted_image(picture, (w - 2 * pad, h - labels - pad * 2))
    calender = ImageDraw.Draw(board)
    text_font = max(11, paper.pixels(2.8))
    label(calender, (6, 4), str(index), size=text_font, fill="#fbf2ce")
    return board, (x + pad, y + labels), (x + pad, y + pad), text_font


def contact_pages(paper, title, subtitle, entries, rows, columns):
    per = rows * columns
    total = math.ceil(len(entries) / per)
    for start, batch in paginate(entries, per):
        image = Image.new("RGB", paper.extent, PALE)
        pen = ImageDraw.Draw(image)
        annual_header(pen, paper, title, subtitle, start // per + 1, total)
        for index, ((record, picture, note), cell) in enumerate(zip(batch, subplot_grid(paper, columns, rows)), start + 1):
            x, y, w, h = cell
            pen.rounded_rectangle((x, y, x + w, y + h), radius=max(4, text_size(paper)),
                                  fill="#e0e1da", outline="#b3b7b7", width=1)
            caption = record.get("caption") or record["title"]
            board, origin, top, text_font = contact_card(picture, paper, cell, caption, note, index)
            image.paste(board, origin)
            body_font = font(text_font, bold=True)
            description = context_lines(caption, pen, w - paper.pixels(4), body_font, lines=1)
            if description:
                pen.text(top, description[0], fill=INK, font=body_font)
            body = context_lines(note, pen, w - paper.pixels(4), font(text_font), lines=1)
            if body:
                pen.text((top[0], top[1] + paper.pixels(4.6)), body[0], fill=QUIET, font=font(text_font))
        yield image


def text_size(paper):
    return max(6, paper.pixels(1.0))


def print_pdf(pages, paper):
    workbook = list(pages)
    if not workbook:
        raise DeskError("Choose at least one view.")
    encoded = BytesIO()
    first, *others = workbook
    first.save(encoded, format="PDF", save_all=True, append_images=others, resolution=paper.resolution)
    return encoded.getvalue(), first


def physical_strip(ctx, image, paper, title, real_size):
    width, height = image.size
    box = paper.extent
    gutter = paper.pixels(paper.margin_mm)
    top = gutter + paper.pixels(19)
    bottom = box[1] - gutter - paper.pixels(18)
    if real_size is None:
        draw_dims = fitted_image(image, (box[0] - gutter * 2, bottom - top))
        overlay = draw_dims
    else:
        exact = tuple(paper.pixels(v) for v in real_size)
        if min(exact) < 5:
            raise DeskError("The study crop is too small for this scale check.")
        if exact[0] > box[0] - gutter * 2 or exact[1] > bottom - top:
            raise DeskError("The selected crop does not fit at 100% on this paper.")
        overlay = image.resize(exact, Image.Resampling.LANCZOS)
    page = Image.new("RGB", box, PALE)
    pen = ImageDraw.Draw(page)
    footer = "Observed image has 1:1 scale" if real_size is not None else "Screen-reference image without physical scale"
    annual_header(pen, paper, title, footer, 1, 1)
    if overlay.mode == "RGBA":
        page.paste(overlay, (gutter, top), overlay.getchannel("A"))
    else:
        page.paste(overlay, (gutter, top))
    first = gutter
    last = gutter + paper.pixels(20)
    y = bottom + paper.pixels(7)
    pen.line((first, y, last, y), fill=INK, width=max(1, paper.pixels(.4)))
    for edge in (first, last):
        pen.line((edge, y - paper.pixels(2), edge, y + paper.pixels(2)), fill=INK, width=max(1, paper.pixels(.4)))
    pen.text((last + paper.pixels(4), y - paper.pixels(2)), "20 mm (print at 100%)", fill=INK, font=font(paper.pixels(3)))
    return page, (gutter, top, overlay.width, overlay.height)
