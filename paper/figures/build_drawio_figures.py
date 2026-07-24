#!/usr/bin/env python3
"""Build self-contained Draw.io source for the two Audio-CVR paper figures."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
import xml.etree.ElementTree as ET


NAVY = "#183153"
TEXT = "#263746"
MUTED = "#667788"
BLUE = "#357ABD"
CORAL = "#D95F4F"
GREEN = "#20806B"
AMBER = "#D9A441"
HAIRLINE = "#D8E0E8"
PALE_BLUE = "#EDF4FA"
PALE_CORAL = "#FCEDEA"
PALE_GREEN = "#EAF5F1"
PALE_AMBER = "#FFF7E6"


def _data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    # mxGraph style values are semicolon-delimited. Percent-encoding the
    # data-URI separator keeps the complete payload inside image=... while
    # remaining a valid browser data URL after draw.io exports the SVG.
    return f"data:{mime}%3Bbase64,{encoded}"


class Page:
    def __init__(self, width: int = 1800, height: int = 650) -> None:
        self.model = ET.Element(
            "mxGraphModel",
            {
                "dx": str(width),
                "dy": str(height),
                "grid": "0",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(width),
                "pageHeight": str(height),
                "math": "1",
                "shadow": "0",
                "adaptiveColors": "auto",
            },
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})
        self._counter = 10
        self.vertex(
            "",
            0,
            0,
            width,
            height,
            "rounded=0;html=1;fillColor=#FFFFFF;strokeColor=none;",
            "background",
        )

    def next_id(self, prefix: str = "n") -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def vertex(
        self,
        value: str,
        x: float,
        y: float,
        width: float,
        height: float,
        style: str,
        cell_id: str | None = None,
    ) -> str:
        cell_id = cell_id or self.next_id()
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cell_id,
                "value": value,
                "style": style,
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(x),
                "y": str(y),
                "width": str(width),
                "height": str(height),
                "as": "geometry",
            },
        )
        return cell_id

    def text(
        self,
        value: str,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        size: int = 18,
        color: str = TEXT,
        bold: bool = False,
        align: str = "left",
        cell_id: str | None = None,
    ) -> str:
        style = (
            "text;html=1;whiteSpace=wrap;verticalAlign=middle;"
            f"align={align};fontFamily=Helvetica;fontSize={size};"
            f"fontColor={color};fontStyle={1 if bold else 0};spacing=0;"
        )
        return self.vertex(value, x, y, width, height, style, cell_id)

    def image(
        self,
        path: Path,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        cell_id: str | None = None,
        border: str = "none",
        border_width: int = 0,
    ) -> str:
        style = (
            "shape=image;html=1;imageAspect=0;aspect=fixed;"
            f"image={_data_uri(path)};strokeColor={border};strokeWidth={border_width};"
        )
        return self.vertex("", x, y, width, height, style, cell_id)

    def line(
        self,
        x: float,
        y: float,
        width: float,
        height: float = 2,
        color: str = HAIRLINE,
        cell_id: str | None = None,
    ) -> str:
        return self.vertex(
            "",
            x,
            y,
            width,
            height,
            f"rounded=0;html=1;fillColor={color};strokeColor=none;",
            cell_id,
        )

    def edge(
        self,
        source: str,
        target: str,
        *,
        color: str = NAVY,
        width: int = 3,
        dashed: bool = False,
        cell_id: str | None = None,
    ) -> str:
        cell_id = cell_id or self.next_id("e")
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;"
            f"strokeColor={color};strokeWidth={width};endArrow=classic;endFill=1;"
            f"dashed={1 if dashed else 0};"
        )
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cell_id,
                "value": "",
                "style": style,
                "edge": "1",
                "source": source,
                "target": target,
                "parent": "1",
            },
        )
        ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        return cell_id

    def xml(self) -> str:
        return ET.tostring(self.model, encoding="unicode")


def _label(page: Page, panel: str, title: str, x: int, width: int) -> None:
    page.text(panel, x, 24, 32, 34, size=28, color=NAVY, bold=True)
    page.text(title, x + 38, 24, width - 38, 34, size=22, color=NAVY, bold=True)


def _thumbnail_frame(
    page: Page,
    image: Path,
    x: int,
    y: int,
    width: int,
    height: int,
    color: str,
    label: str,
) -> str:
    page.vertex(
        "",
        x - 4,
        y - 4,
        width + 8,
        height + 8,
        f"rounded=1;arcSize=4;html=1;fillColor=#FFFFFF;strokeColor={color};strokeWidth=3;",
    )
    image_id = page.image(image, x, y, width, height)
    page.vertex(
        label,
        x + 8,
        y + 8,
        92,
        26,
        (
            f"rounded=1;arcSize=20;html=1;fillColor={color};strokeColor={color};"
            "fontFamily=Helvetica;fontSize=13;fontStyle=1;fontColor=#FFFFFF;"
        ),
    )
    return image_id


def _pill(
    page: Page,
    value: str,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str,
    color: str,
    size: int = 14,
    bold: bool = True,
) -> str:
    return page.vertex(
        value,
        x,
        y,
        width,
        height,
        (
            f"rounded=1;arcSize=30;html=1;fillColor={fill};strokeColor={stroke};"
            f"strokeWidth=2;fontFamily=Helvetica;fontSize={size};"
            f"fontStyle={1 if bold else 0};fontColor={color};"
        ),
    )


def _rank_card(
    page: Page,
    rank: str,
    label: str,
    x: float,
    y: float,
    *,
    fill: str,
    stroke: str,
    rank_fill: str,
    label_color: str = TEXT,
    width: float = 184,
    height: float = 84,
) -> str:
    card = page.vertex(
        "",
        x,
        y,
        width,
        height,
        f"rounded=1;arcSize=8;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;",
    )
    page.vertex(
        rank,
        x + 12,
        y + 20,
        44,
        44,
        (
            f"ellipse;html=1;fillColor={rank_fill};strokeColor=#FFFFFF;strokeWidth=2;"
            "fontFamily=Helvetica;fontSize=18;fontStyle=1;fontColor=#FFFFFF;"
        ),
    )
    page.text(
        label,
        x + 64,
        y + 13,
        width - 76,
        height - 26,
        size=15,
        color=label_color,
        bold=True,
    )
    return card


def _gate_lane(
    page: Page,
    *,
    y: float,
    title: str,
    line_one: str,
    line_two: str,
    badge: str,
    fill: str,
    stroke: str,
    media: Path,
    media_height: float,
) -> str:
    lane = page.vertex(
        "",
        765,
        y,
        640,
        142,
        f"rounded=1;arcSize=6;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;",
    )
    page.image(media, 783, y + (142 - media_height) / 2, 180, media_height)
    page.text(title, 985, y + 14, 230, 28, size=18, color=stroke, bold=True)
    page.text(line_one, 985, y + 50, 255, 28, size=14, color=TEXT, bold=True)
    page.text(line_two, 985, y + 82, 285, 42, size=13, color=MUTED)
    _pill(
        page,
        badge,
        1288,
        y + 50,
        98,
        42,
        fill="#FFFFFF",
        stroke=stroke,
        color=stroke,
        size=13,
    )
    return lane


def _stroke(
    page: Page,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    color: str,
    rotation: float = 0,
) -> str:
    return page.vertex(
        "",
        x,
        y,
        width,
        height,
        (
            f"rounded=1;arcSize=50;html=1;fillColor={color};strokeColor=none;"
            f"rotation={rotation};"
        ),
    )


def _icon_badge(
    page: Page,
    kind: str,
    x: float,
    y: float,
    size: float = 54,
    *,
    color: str = NAVY,
    fill: str = "#FFFFFF",
    stroke: str | None = None,
) -> str:
    """Draw a compact, editable vector icon inside a circular badge."""
    stroke = stroke or color
    badge = page.vertex(
        "",
        x,
        y,
        size,
        size,
        (
            f"ellipse;html=1;fillColor={fill};strokeColor={stroke};"
            "strokeWidth=2;"
        ),
    )
    cx = x + size / 2
    cy = y + size / 2
    scale = size / 54

    if kind in {"video", "av", "query"}:
        page.vertex(
            "",
            x + 12 * scale,
            y + 16 * scale,
            30 * scale,
            22 * scale,
            (
                f"rounded=1;arcSize=12;html=1;fillColor=none;"
                f"strokeColor={color};strokeWidth=2;"
            ),
        )
        page.vertex(
            "",
            x + 25 * scale,
            y + 22 * scale,
            8 * scale,
            10 * scale,
            (
                f"triangle;html=1;direction=east;fillColor={color};"
                "strokeColor=none;"
            ),
        )
    if kind in {"audio", "av", "query", "mute"}:
        base_x = x + (11 if kind == "audio" else 8) * scale
        if kind in {"av", "query"}:
            base_x = x + 7 * scale
        heights = (10, 20, 28, 18, 12)
        for index, bar_height in enumerate(heights):
            _stroke(
                page,
                base_x + index * 6 * scale,
                cy - bar_height * scale / 2,
                3 * scale,
                bar_height * scale,
                color=color,
            )
    if kind == "query":
        page.vertex(
            "+",
            x + 34 * scale,
            y + 4 * scale,
            16 * scale,
            16 * scale,
            (
                f"ellipse;html=1;fillColor={AMBER};strokeColor=#FFFFFF;"
                "strokeWidth=1;fontFamily=Helvetica;fontSize=12;"
                "fontStyle=1;fontColor=#FFFFFF;"
            ),
        )
    elif kind == "eye":
        page.vertex(
            "",
            x + 10 * scale,
            y + 19 * scale,
            34 * scale,
            17 * scale,
            (
                f"ellipse;html=1;fillColor=none;strokeColor={color};"
                "strokeWidth=2;"
            ),
        )
        page.vertex(
            "",
            cx - 5 * scale,
            cy - 5 * scale,
            10 * scale,
            10 * scale,
            f"ellipse;html=1;fillColor={color};strokeColor=none;",
        )
    elif kind == "mute":
        _stroke(
            page,
            x + 10 * scale,
            y + 25 * scale,
            35 * scale,
            3 * scale,
            color=CORAL,
            rotation=-38,
        )
    elif kind == "ledger":
        page.vertex(
            "",
            x + 15 * scale,
            y + 10 * scale,
            26 * scale,
            34 * scale,
            (
                f"rounded=1;arcSize=5;html=1;fillColor=#FFFFFF;"
                f"strokeColor={color};strokeWidth=2;"
            ),
        )
        for offset in (0, 9, 18):
            page.vertex(
                "",
                x + 19 * scale,
                y + (16 + offset) * scale,
                5 * scale,
                5 * scale,
                f"ellipse;html=1;fillColor={GREEN};strokeColor=none;",
            )
            _stroke(
                page,
                x + 28 * scale,
                y + (18 + offset) * scale,
                9 * scale,
                2 * scale,
                color=color,
            )
    elif kind == "repeat":
        page.text("20%", x + 7 * scale, y + 13 * scale, 40 * scale, 24 * scale, size=12, color=color, bold=True, align="center")
    elif kind == "dedup":
        for offset in (0, 7):
            page.vertex(
                "",
                x + (12 + offset) * scale,
                y + (13 + offset) * scale,
                25 * scale,
                25 * scale,
                (
                    f"rounded=1;arcSize=6;html=1;fillColor=none;"
                    f"strokeColor={color};strokeWidth=2;"
                ),
            )
    elif kind == "lock":
        page.vertex(
            "",
            x + 17 * scale,
            y + 8 * scale,
            20 * scale,
            24 * scale,
            (
                f"ellipse;html=1;fillColor=none;strokeColor={color};"
                "strokeWidth=3;"
            ),
        )
        page.vertex(
            "",
            x + 12 * scale,
            y + 24 * scale,
            30 * scale,
            23 * scale,
            (
                f"rounded=1;arcSize=10;html=1;fillColor={color};"
                f"strokeColor={color};strokeWidth=2;"
            ),
        )
        page.vertex(
            "",
            cx - 2 * scale,
            y + 31 * scale,
            4 * scale,
            8 * scale,
            "rounded=1;html=1;fillColor=#FFFFFF;strokeColor=none;",
        )
    elif kind == "scissors":
        for offset in (-7, 7):
            page.vertex(
                "",
                cx + offset * scale - 5 * scale,
                y + 31 * scale,
                10 * scale,
                10 * scale,
                f"ellipse;html=1;fillColor=none;strokeColor={color};strokeWidth=2;",
            )
        _stroke(page, x + 18 * scale, y + 15 * scale, 24 * scale, 3 * scale, color=color, rotation=35)
        _stroke(page, x + 18 * scale, y + 15 * scale, 24 * scale, 3 * scale, color=color, rotation=-35)
    elif kind == "filter":
        page.vertex(
            "",
            x + 12 * scale,
            y + 12 * scale,
            30 * scale,
            23 * scale,
            (
                f"shape=trapezoid;direction=south;html=1;fillColor=none;"
                f"strokeColor={color};strokeWidth=2;"
            ),
        )
        _stroke(page, cx - 2 * scale, y + 34 * scale, 4 * scale, 10 * scale, color=color)
    elif kind == "database":
        page.vertex(
            "",
            x + 11 * scale,
            y + 13 * scale,
            32 * scale,
            27 * scale,
            f"rounded=0;html=1;fillColor={fill};strokeColor={color};strokeWidth=2;",
        )
        for yy in (10, 33):
            page.vertex(
                "",
                x + 11 * scale,
                y + yy * scale,
                32 * scale,
                12 * scale,
                f"ellipse;html=1;fillColor={fill};strokeColor={color};strokeWidth=2;",
            )
    return badge


def _panel_header(
    page: Page,
    panel: str,
    title: str,
    icon: str,
    x: float,
    width: float,
    *,
    color: str = NAVY,
) -> None:
    page.text(panel, x, 23, 28, 32, size=26, color=NAVY, bold=True)
    _icon_badge(page, icon, x + 34, 19, 38, color=color, fill="#FFFFFF")
    page.text(title, x + 82, 23, width - 82, 32, size=20, color=NAVY, bold=True)


def _check_badge(page: Page, x: float, y: float, *, passed: bool) -> str:
    color = GREEN if passed else CORAL
    # ASCII labels avoid Type 3 symbol fonts in the exported submission PDF.
    symbol = "OK" if passed else "NO"
    return page.vertex(
        symbol,
        x,
        y,
        30,
        30,
        (
            f"ellipse;html=1;fillColor={color};strokeColor=#FFFFFF;"
            "strokeWidth=2;fontFamily=Helvetica;fontSize=11;"
            "fontStyle=1;fontColor=#FFFFFF;"
        ),
    )


def _legacy_build_figure_one(asset_root: Path, candidate_root: Path) -> Page:
    page = Page()
    _label(page, "a", "A directional audio-primary query", 28, 610)
    _label(page, "b", "One-score counterfactual", 675, 410)
    _label(page, "c", "Cross-model source anchoring", 1135, 635)
    page.line(650, 24, 2, 602)
    page.line(1110, 24, 2, 602)

    page.text(
        "fixed Test1000 example; selected without retrieval scores",
        48,
        62,
        570,
        24,
        size=13,
        color=MUTED,
    )
    ref_strip = page.image(
        asset_root / "reference_filmstrip.jpg",
        48,
        96,
        570,
        105,
        border=BLUE,
        border_width=2,
    )
    _pill(
        page,
        "REFERENCE / PRE-EDIT",
        60,
        108,
        178,
        27,
        fill=BLUE,
        stroke=BLUE,
        color="#FFFFFF",
        size=12,
    )
    page.image(asset_root / "reference_waveform.png", 48, 207, 570, 87)
    edit = _pill(
        page,
        "add the sound of a crowd applauding",
        160,
        304,
        345,
        46,
        fill=PALE_AMBER,
        stroke=AMBER,
        color=NAVY,
        size=15,
    )
    target_strip = page.image(
        asset_root / "target_filmstrip.jpg",
        48,
        363,
        570,
        105,
        border=GREEN,
        border_width=2,
    )
    _pill(
        page,
        "TARGET / POST-EDIT",
        60,
        375,
        172,
        27,
        fill=GREEN,
        stroke=GREEN,
        color="#FFFFFF",
        size=12,
    )
    page.image(asset_root / "target_waveform.png", 48, 474, 570, 87)
    _pill(
        page,
        "EDIT NOT SATISFIED",
        56,
        582,
        185,
        34,
        fill=PALE_CORAL,
        stroke=CORAL,
        color=CORAL,
        size=13,
    )
    page.text(
        "visual context preserved",
        245,
        582,
        188,
        34,
        size=13,
        color=MUTED,
        bold=True,
        align="center",
    )
    _pill(
        page,
        "EDIT SATISFIED",
        438,
        582,
        172,
        34,
        fill=PALE_GREEN,
        stroke=GREEN,
        color=GREEN,
        size=13,
    )

    _pill(
        page,
        "only s(q, own reference) -> -inf",
        700,
        78,
        385,
        44,
        fill=PALE_AMBER,
        stroke=AMBER,
        color=NAVY,
        size=14,
    )
    page.text("WITH REFERENCE", 684, 136, 190, 28, size=13, color=MUTED, bold=True)
    page.text("MASK OWN REFERENCE", 897, 136, 194, 28, size=13, color=MUTED, bold=True)
    exact_ref = _rank_card(
        page,
        "1",
        "own<br>reference",
        682,
        174,
        fill=PALE_BLUE,
        stroke=BLUE,
        rank_fill=CORAL,
    )
    _rank_card(
        page,
        "2",
        "edited<br>target",
        682,
        276,
        fill=PALE_GREEN,
        stroke=GREEN,
        rank_fill=GREEN,
    )
    _rank_card(
        page,
        "3",
        "other<br>candidate",
        682,
        378,
        fill="#F4F6F8",
        stroke="#AAB5BF",
        rank_fill="#AAB5BF",
        label_color=MUTED,
    )
    masked_ref = _rank_card(
        page,
        "-inf",
        "own<br>reference",
        902,
        174,
        fill=PALE_CORAL,
        stroke=CORAL,
        rank_fill=CORAL,
        label_color=CORAL,
    )
    page.line(974, 214, 92, 4, color=CORAL)
    _rank_card(
        page,
        "1",
        "edited<br>target",
        902,
        276,
        fill=PALE_GREEN,
        stroke=GREEN,
        rank_fill=GREEN,
    )
    _rank_card(
        page,
        "2",
        "other<br>candidate",
        902,
        378,
        fill="#F4F6F8",
        stroke="#AAB5BF",
        rank_fill="#AAB5BF",
        label_color=MUTED,
    )
    page.edge(exact_ref, masked_ref, color=AMBER, width=2, dashed=True)
    page.vertex(
        "",
        690,
        492,
        388,
        98,
        "rounded=1;arcSize=8;html=1;fillColor=#FFFFFF;strokeColor=#AAB5BF;strokeWidth=2;",
    )
    page.text(
        "same score row",
        710,
        504,
        348,
        26,
        size=16,
        color=NAVY,
        bold=True,
        align="center",
    )
    page.text(
        "2,000 items -> 1,999 effective<br>1 score masked; 1,999 untouched",
        710,
        532,
        348,
        47,
        size=14,
        color=MUTED,
        align="center",
    )

    page.vertex(
        "+84.5 to +99.2 pp R@1 after masking",
        1148,
        78,
        620,
        54,
        (
            f"rounded=1;arcSize=8;html=1;fillColor={NAVY};strokeColor={NAVY};"
            "fontFamily=Helvetica;fontSize=22;fontStyle=1;fontColor=#FFFFFF;"
        ),
    )
    page.text(
        "Full1000; n = 1,000 queries; V+A+T",
        1148,
        139,
        320,
        26,
        size=13,
        color=MUTED,
        bold=True,
    )
    page.vertex(
        "",
        1490,
        143,
        14,
        14,
        f"ellipse;html=1;fillColor={CORAL};strokeColor=none;",
    )
    page.text("exact", 1510, 136, 62, 28, size=13, color=MUTED)
    page.vertex(
        "",
        1582,
        143,
        14,
        14,
        f"ellipse;html=1;fillColor={GREEN};strokeColor=none;",
    )
    page.text("own reference masked", 1602, 136, 165, 28, size=13, color=MUTED)

    x0, x1 = 1360, 1735
    page.line(x0, 206, x1 - x0, 3, color=HAIRLINE)
    for value, x in ((0, x0), (50, (x0 + x1) / 2), (100, x1)):
        page.line(x - 1, 199, 2, 16, color="#AAB5BF")
        page.text(str(value), x - 22, 216, 44, 22, size=12, color=MUTED, align="center")
    rows = [
        ("E5 adapter", 12.78, 97.26),
        ("ImageBind", 2.5, 98.5),
        ("OmniEmbed", 0.0, 99.2),
    ]
    for index, (name, before, after) in enumerate(rows):
        y = 302 + index * 128
        page.text(name, 1135, y - 19, 202, 38, size=17, color=TEXT, bold=True, align="right")
        bx = x0 + (x1 - x0) * before / 100
        ax = x0 + (x1 - x0) * after / 100
        page.line(bx, y, max(3, ax - bx), 5, color="#AAB5BF")
        page.vertex(
            "",
            bx - 13,
            y - 13,
            26,
            26,
            f"ellipse;html=1;fillColor={CORAL};strokeColor=#FFFFFF;strokeWidth=2;",
        )
        page.vertex(
            "",
            ax - 15,
            y - 15,
            30,
            30,
            f"ellipse;html=1;fillColor={GREEN};strokeColor=#FFFFFF;strokeWidth=2;",
        )
        page.text(f"{before:.1f}", bx - 28, y - 48, 56, 26, size=14, color=CORAL, bold=True, align="center")
        page.text(f"{after:.1f}", ax - 31, y - 48, 62, 26, size=14, color=GREEN, bold=True, align="center")
    page.text("R@1 (%)", 1648, 604, 90, 26, size=14, color=MUTED, bold=True, align="right")
    return page


def _legacy_build_figure_two(asset_root: Path, candidate_root: Path) -> Page:
    page = Page()
    _label(page, "a", "Source-aware clips", 28, 315)
    _label(page, "b", "Audio-first pairing", 385, 325)
    _label(page, "c", "Modality-isolated verification", 760, 640)
    _label(page, "d", "Audit and freeze", 1450, 320)
    for x in (360, 735, 1430):
        page.line(x, 24, 2, 602)

    source = page.image(
        asset_root / "reference_filmstrip.jpg",
        42,
        92,
        286,
        53,
        border=BLUE,
        border_width=2,
    )
    page.text("window A  /  6-9 s", 44, 150, 280, 24, size=13, color=BLUE, bold=True, align="center")
    page.image(
        asset_root / "target_filmstrip.jpg",
        42,
        196,
        286,
        53,
        border=GREEN,
        border_width=2,
    )
    page.text("window B  /  6-9 s", 44, 254, 280, 24, size=13, color=GREEN, bold=True, align="center")
    page.line(58, 296, 252, 4, color=HAIRLINE)
    page.vertex("", 92, 286, 12, 24, f"rounded=1;html=1;fillColor={BLUE};strokeColor=none;")
    page.vertex("", 261, 286, 12, 24, f"rounded=1;html=1;fillColor={GREEN};strokeColor=none;")
    page.text("same raw source", 76, 314, 215, 26, size=13, color=MUTED, bold=True, align="center")
    page.vertex(
        "",
        44,
        363,
        280,
        174,
        "rounded=1;arcSize=6;html=1;fillColor=#F7F9FB;strokeColor=#AAB5BF;strokeWidth=2;",
    )
    page.text("PROVENANCE", 62, 376, 244, 28, size=15, color=NAVY, bold=True)
    for index, label in enumerate(
        ("dataset", "raw source ID", "clip start / end", "source-disjoint group")
    ):
        y = 417 + index * 29
        page.vertex("", 63, y + 6, 10, 10, f"ellipse;html=1;fillColor={BLUE};strokeColor=none;")
        page.text(label, 84, y, 220, 22, size=13, color=MUTED)
    page.text(
        "stable IDs follow every derivative",
        48,
        570,
        272,
        30,
        size=13,
        color=TEXT,
        bold=True,
        align="center",
    )

    pair = page.image(asset_root / "muted_video_pair.jpg", 394, 94, 316, 86)
    page.text("reference", 396, 185, 146, 24, size=13, color=BLUE, bold=True, align="center")
    page.text("target", 562, 185, 146, 24, size=13, color=GREEN, bold=True, align="center")
    page.image(asset_root / "reference_waveform.png", 398, 229, 308, 47)
    page.image(asset_root / "target_waveform.png", 398, 282, 308, 47)
    _pill(
        page,
        "add the sound of a crowd applauding",
        412,
        354,
        280,
        46,
        fill=PALE_AMBER,
        stroke=AMBER,
        color=NAVY,
        size=14,
    )
    _pill(
        page,
        "visual context preserved",
        420,
        435,
        264,
        40,
        fill=PALE_BLUE,
        stroke=BLUE,
        color=BLUE,
        size=13,
    )
    _pill(
        page,
        "audible directional delta",
        420,
        493,
        264,
        40,
        fill=PALE_GREEN,
        stroke=GREEN,
        color=GREEN,
        size=13,
    )
    page.text(
        "reference fails edit; target satisfies it",
        400,
        565,
        304,
        35,
        size=13,
        color=MUTED,
        bold=True,
        align="center",
    )
    page.edge(source, pair, color=NAVY, width=3)

    page.text(
        "each reviewer receives a different view",
        775,
        65,
        610,
        25,
        size=13,
        color=MUTED,
        bold=True,
        align="center",
    )
    gate_audio = _gate_lane(
        page,
        y=96,
        title="AUDIO ONLY",
        line_one="is the change audible?",
        line_two="reference fails; target satisfies the edit",
        badge="DIRECTION",
        fill=PALE_AMBER,
        stroke=AMBER,
        media=asset_root / "audio_only_pair.png",
        media_height=61,
    )
    gate_video = _gate_lane(
        page,
        y=253,
        title="MUTED VIDEO",
        line_one="can vision reveal the target?",
        line_two="reject candidates with a visual shortcut",
        badge="NO SHORTCUT",
        fill=PALE_BLUE,
        stroke=BLUE,
        media=asset_root / "muted_video_pair.jpg",
        media_height=49,
    )
    gate_full = _gate_lane(
        page,
        y=410,
        title="FULL AUDIOVISUAL",
        line_one="does the edit hold in context?",
        line_two="screen contextual and transcript / ASR shortcuts",
        badge="CONSISTENT",
        fill=PALE_GREEN,
        stroke=GREEN,
        media=asset_root / "target_filmstrip.jpg",
        media_height=33,
    )
    all_pass = _pill(
        page,
        "ALL THREE GATES PASS",
        945,
        580,
        280,
        38,
        fill=PALE_GREEN,
        stroke=GREEN,
        color=GREEN,
        size=14,
    )

    ledger = page.vertex(
        "",
        1452,
        90,
        316,
        130,
        "rounded=1;arcSize=6;html=1;fillColor=#F7F9FB;strokeColor=#AAB5BF;strokeWidth=2;",
    )
    page.text("ATOMIC REVIEW LEDGER", 1470, 103, 280, 26, size=15, color=NAVY, bold=True)
    for index, label in enumerate(
        ("decision + confidence", "rejection reason", "media + provenance")
    ):
        y = 139 + index * 27
        page.line(1472, y + 9, 34, 4, color=GREEN)
        page.text(label, 1518, y, 225, 22, size=13, color=MUTED)
    stability = _pill(
        page,
        "20% SAMPLED STABILITY AUDIT",
        1470,
        252,
        280,
        44,
        fill=PALE_AMBER,
        stroke=AMBER,
        color=NAVY,
        size=13,
    )
    page.edge(ledger, stability, color="#AAB5BF", width=2)
    dedup = page.vertex(
        "CANONICAL DEDUPLICATION<br><font style='font-size:13px;color:#667788'>sample / pair / source / inverse</font>",
        1470,
        331,
        280,
        82,
        (
            f"rounded=1;arcSize=6;html=1;fillColor={PALE_BLUE};strokeColor={BLUE};"
            f"strokeWidth=2;fontFamily=Helvetica;fontSize=15;fontStyle=1;fontColor={BLUE};"
        ),
    )
    page.edge(stability, dedup, color="#AAB5BF", width=2)
    frozen = page.vertex(
        "FROZEN FULL1000<br><font style='font-size:15px'>1,000 targets + 1,000 references</font>",
        1462,
        454,
        296,
        96,
        (
            f"rounded=1;arcSize=6;html=1;fillColor={NAVY};strokeColor={NAVY};"
            "fontFamily=Helvetica;fontSize=22;fontStyle=1;fontColor=#FFFFFF;"
        ),
    )
    page.edge(dedup, frozen, color=GREEN, width=3)
    page.text(
        "source-disjoint  |  SHA256 locked",
        1468,
        558,
        284,
        26,
        size=13,
        color=GREEN,
        bold=True,
        align="center",
    )
    page.text(
        "automatically curated and model-verified",
        1448,
        591,
        324,
        26,
        size=12,
        color=MUTED,
        align="center",
    )
    return page


def build_figure_one(asset_root: Path, candidate_root: Path) -> Page:
    """Reference confusion as a media-first query, intervention, and effect plot."""
    page = Page()
    _panel_header(page, "a", "Directional query", "query", 26, 720, color=AMBER)
    _panel_header(page, "b", "Mask one score", "mute", 770, 405, color=CORAL)
    _panel_header(page, "c", "Cross-model effect", "database", 1210, 560, color=GREEN)
    page.line(748, 22, 2, 604)
    page.line(1188, 22, 2, 604)

    page.text(
        "fixed Full1000 example / score-independent",
        48,
        66,
        500,
        23,
        size=12,
        color=MUTED,
    )

    reference_card = page.vertex(
        "",
        46,
        108,
        282,
        222,
        (
            f"rounded=1;arcSize=6;html=1;fillColor={PALE_BLUE};"
            f"strokeColor={BLUE};strokeWidth=2;"
        ),
    )
    page.image(asset_root / "reference_filmstrip.jpg", 58, 122, 258, 88)
    page.image(asset_root / "reference_waveform.png", 58, 220, 258, 56)
    _icon_badge(page, "video", 61, 286, 32, color=BLUE, fill="#FFFFFF")
    page.text("REFERENCE", 101, 287, 118, 28, size=14, color=BLUE, bold=True)
    _check_badge(page, 282, 286, passed=False)

    target_card = page.vertex(
        "",
        438,
        108,
        282,
        222,
        (
            f"rounded=1;arcSize=6;html=1;fillColor={PALE_GREEN};"
            f"strokeColor={GREEN};strokeWidth=2;"
        ),
    )
    page.image(asset_root / "target_filmstrip.jpg", 450, 122, 258, 88)
    page.image(asset_root / "target_waveform.png", 450, 220, 258, 56)
    _icon_badge(page, "video", 453, 286, 32, color=GREEN, fill="#FFFFFF")
    page.text("TARGET", 493, 287, 95, 28, size=14, color=GREEN, bold=True)
    _check_badge(page, 674, 286, passed=True)

    edit_badge = _icon_badge(
        page,
        "query",
        344,
        144,
        76,
        color=AMBER,
        fill=PALE_AMBER,
        stroke=AMBER,
    )
    page.text("ADD", 343, 226, 78, 22, size=12, color=AMBER, bold=True, align="center")
    page.text("DOG BARK", 326, 248, 112, 25, size=15, color=NAVY, bold=True, align="center")
    page.edge(reference_card, edit_badge, color=AMBER, width=3)
    page.edge(edit_badge, target_card, color=AMBER, width=3)

    page.vertex(
        "",
        74,
        380,
        275,
        122,
        "rounded=1;arcSize=8;html=1;fillColor=#FFFFFF;strokeColor=#D8E0E8;strokeWidth=2;",
    )
    _icon_badge(page, "eye", 96, 408, 58, color=BLUE, fill=PALE_BLUE)
    page.text("SAME", 166, 418, 58, 30, size=11, color=NAVY, bold=True, align="center")
    _icon_badge(page, "video", 235, 408, 58, color=GREEN, fill=PALE_GREEN)
    page.text("same scene", 92, 467, 205, 24, size=13, color=MUTED, bold=True, align="center")

    page.vertex(
        "",
        418,
        380,
        275,
        122,
        "rounded=1;arcSize=8;html=1;fillColor=#FFFFFF;strokeColor=#D8E0E8;strokeWidth=2;",
    )
    _icon_badge(page, "audio", 440, 408, 58, color=BLUE, fill=PALE_BLUE)
    page.text("DELTA", 510, 418, 62, 30, size=11, color=AMBER, bold=True, align="center")
    _icon_badge(page, "audio", 580, 408, 58, color=GREEN, fill=PALE_GREEN)
    page.text("sound changes", 442, 467, 196, 24, size=13, color=MUTED, bold=True, align="center")

    _pill(
        page,
        "PRE-EDIT",
        100,
        548,
        150,
        38,
        fill=PALE_CORAL,
        stroke=CORAL,
        color=CORAL,
        size=14,
    )
    page.text("TO", 334, 555, 70, 24, size=12, color=AMBER, bold=True, align="center")
    _pill(
        page,
        "POST-EDIT",
        496,
        548,
        150,
        38,
        fill=PALE_GREEN,
        stroke=GREEN,
        color=GREEN,
        size=14,
    )

    page.text("EXACT", 782, 76, 154, 25, size=13, color=MUTED, bold=True, align="center")
    page.text("MASKED", 1010, 76, 154, 25, size=13, color=MUTED, bold=True, align="center")

    exact_ref = page.vertex(
        "",
        784,
        112,
        154,
        154,
        (
            f"rounded=1;arcSize=6;html=1;fillColor={PALE_CORAL};"
            f"strokeColor={CORAL};strokeWidth=2;"
        ),
    )
    page.image(asset_root / "reference_filmstrip.jpg", 796, 126, 130, 62)
    page.vertex(
        "1",
        796,
        199,
        38,
        38,
        (
            f"ellipse;html=1;fillColor={CORAL};strokeColor=#FFFFFF;strokeWidth=2;"
            "fontFamily=Helvetica;fontSize=18;fontStyle=1;fontColor=#FFFFFF;"
        ),
    )
    _icon_badge(page, "video", 848, 196, 38, color=CORAL, fill="#FFFFFF")
    page.text("own ref", 889, 200, 42, 35, size=12, color=CORAL, bold=True, align="center")

    page.vertex(
        "",
        784,
        288,
        154,
        154,
        (
            f"rounded=1;arcSize=6;html=1;fillColor={PALE_GREEN};"
            f"strokeColor={GREEN};strokeWidth=2;"
        ),
    )
    page.image(asset_root / "target_filmstrip.jpg", 796, 302, 130, 62)
    page.vertex(
        "2",
        796,
        375,
        38,
        38,
        (
            f"ellipse;html=1;fillColor={GREEN};strokeColor=#FFFFFF;strokeWidth=2;"
            "fontFamily=Helvetica;fontSize=18;fontStyle=1;fontColor=#FFFFFF;"
        ),
    )
    _icon_badge(page, "video", 848, 372, 38, color=GREEN, fill="#FFFFFF")
    page.text("target", 889, 376, 42, 35, size=12, color=GREEN, bold=True, align="center")

    masked_target = page.vertex(
        "",
        1012,
        112,
        154,
        154,
        (
            f"rounded=1;arcSize=6;html=1;fillColor={PALE_GREEN};"
            f"strokeColor={GREEN};strokeWidth=2;"
        ),
    )
    page.image(asset_root / "target_filmstrip.jpg", 1024, 126, 130, 62)
    page.vertex(
        "1",
        1024,
        199,
        38,
        38,
        (
            f"ellipse;html=1;fillColor={GREEN};strokeColor=#FFFFFF;strokeWidth=2;"
            "fontFamily=Helvetica;fontSize=18;fontStyle=1;fontColor=#FFFFFF;"
        ),
    )
    _check_badge(page, 1124, 203, passed=True)

    masked_ref = page.vertex(
        "",
        1012,
        288,
        154,
        154,
        (
            f"rounded=1;arcSize=6;html=1;fillColor={PALE_CORAL};"
            f"strokeColor={CORAL};strokeWidth=2;opacity=55;"
        ),
    )
    page.image(asset_root / "reference_filmstrip.jpg", 1024, 302, 130, 62)
    _icon_badge(page, "mute", 1057, 363, 58, color=CORAL, fill="#FFFFFF")
    page.text("MASK", 1018, 390, 48, 30, size=10, color=CORAL, bold=True, align="center")
    _icon_badge(page, "mute", 940, 208, 64, color=CORAL, fill=PALE_CORAL)
    page.text("1 / 2,000", 928, 472, 90, 30, size=18, color=NAVY, bold=True, align="center")
    page.text("scores changed", 909, 501, 128, 24, size=12, color=MUTED, align="center")
    page.text("1,999 fixed", 912, 548, 122, 25, size=13, color=GREEN, bold=True, align="center")

    _icon_badge(page, "lock", 1228, 78, 58, color=CORAL, fill=PALE_CORAL)
    page.text("OWN REFERENCE", 1300, 83, 160, 26, size=15, color=CORAL, bold=True)
    page.text("present", 1300, 108, 82, 22, size=12, color=MUTED)
    _icon_badge(page, "mute", 1490, 78, 58, color=GREEN, fill=PALE_GREEN)
    page.text("MASKED", 1561, 83, 100, 26, size=15, color=GREEN, bold=True)
    page.text("same gallery", 1561, 108, 120, 22, size=12, color=MUTED)

    x0, x1 = 1408, 1707
    page.line(x0, 190, x1 - x0, 3, color=HAIRLINE)
    for value, x in ((0, x0), (50, (x0 + x1) / 2), (100, x1)):
        page.line(x - 1, 183, 2, 16, color="#AAB5BF")
        page.text(str(value), x - 22, 201, 44, 22, size=11, color=MUTED, align="center")

    rows = [
        ("E5 adapter", "query", 12.78, 97.26),
        ("ImageBind", "av", 2.5, 98.5),
        ("OmniEmbed", "database", 0.0, 99.2),
    ]
    for index, (name, icon, before, after) in enumerate(rows):
        y = 300 + index * 126
        _icon_badge(page, icon, 1215, y - 25, 50, color=NAVY, fill="#FFFFFF")
        page.text(name, 1272, y - 20, 124, 40, size=15, color=TEXT, bold=True, align="right")
        bx = x0 + (x1 - x0) * before / 100
        ax = x0 + (x1 - x0) * after / 100
        page.line(bx, y, max(3, ax - bx), 5, color="#AAB5BF")
        page.vertex(
            "",
            bx - 12,
            y - 12,
            24,
            24,
            f"ellipse;html=1;fillColor={CORAL};strokeColor=#FFFFFF;strokeWidth=2;",
        )
        page.vertex(
            "",
            ax - 14,
            y - 14,
            28,
            28,
            f"ellipse;html=1;fillColor={GREEN};strokeColor=#FFFFFF;strokeWidth=2;",
        )
        page.text(f"{before:.1f}", bx - 26, y - 45, 52, 22, size=13, color=CORAL, bold=True, align="center")
        page.text(f"{after:.1f}", ax - 28, y - 45, 56, 22, size=13, color=GREEN, bold=True, align="center")
        page.text(
            f"+{after - before:.1f}",
            1717,
            y - 17,
            60,
            34,
            size=15,
            color=GREEN,
            bold=True,
            align="right",
        )
    page.text("R@1 (%)", 1640, 594, 80, 24, size=13, color=MUTED, bold=True, align="right")
    return page


def _draft_build_figure_two(asset_root: Path, candidate_root: Path) -> Page:
    """A compact icon-led curation pipeline with parallel modality gates."""
    page = Page()
    page.line(82, 319, 1630, 4, color=HAIRLINE)

    stage_specs = (
        ("1", "CLIP", "scissors", 34, 300, BLUE),
        ("2", "PAIR", "query", 358, 330, AMBER),
        ("3", "VERIFY", "eye", 716, 650, GREEN),
        ("4", "FREEZE", "lock", 1396, 370, NAVY),
    )
    for number, title, icon, x, width, color in stage_specs:
        page.vertex(
            number,
            x,
            22,
            34,
            34,
            (
                f"ellipse;html=1;fillColor={color};strokeColor=#FFFFFF;"
                "strokeWidth=2;fontFamily=Helvetica;fontSize=17;"
                "fontStyle=1;fontColor=#FFFFFF;"
            ),
        )
        _icon_badge(page, icon, x + 45, 20, 40, color=color, fill="#FFFFFF")
        page.text(title, x + 96, 23, width - 96, 30, size=20, color=NAVY, bold=True)

    source_card = page.vertex(
        "",
        38,
        104,
        284,
        114,
        (
            f"rounded=1;arcSize=6;html=1;fillColor={PALE_BLUE};"
            f"strokeColor={BLUE};strokeWidth=2;"
        ),
    )
    page.image(asset_root / "reference_filmstrip.jpg", 50, 116, 260, 90)
    _icon_badge(page, "video", 58, 228, 42, color=BLUE, fill="#FFFFFF")
    page.text("RAW SOURCE", 108, 235, 126, 28, size=14, color=BLUE, bold=True)

    scissors = _icon_badge(page, "scissors", 151, 270, 58, color=AMBER, fill=PALE_AMBER)
    page.edge(source_card, scissors, color=AMBER, width=2)
    clip_a = page.vertex(
        "",
        42,
        350,
        130,
        84,
        f"rounded=1;html=1;fillColor={PALE_BLUE};strokeColor={BLUE};strokeWidth=2;",
    )
    page.image(asset_root / "reference_filmstrip.jpg", 50, 358, 114, 48)
    page.text("A", 92, 406, 30, 22, size=14, color=BLUE, bold=True, align="center")
    clip_b = page.vertex(
        "",
        190,
        350,
        130,
        84,
        f"rounded=1;html=1;fillColor={PALE_GREEN};strokeColor={GREEN};strokeWidth=2;",
    )
    page.image(asset_root / "target_filmstrip.jpg", 198, 358, 114, 48)
    page.text("B", 240, 406, 30, 22, size=14, color=GREEN, bold=True, align="center")
    page.edge(scissors, clip_a, color=BLUE, width=2)
    page.edge(scissors, clip_b, color=GREEN, width=2)

    provenance_icons = (
        ("database", "dataset"),
        ("ledger", "source"),
        ("repeat", "time"),
        ("dedup", "group"),
    )
    for index, (icon, label) in enumerate(provenance_icons):
        x = 44 + index * 70
        _icon_badge(page, icon, x, 478, 44, color=BLUE, fill="#FFFFFF")
        page.text(label, x - 5, 528, 54, 22, size=11, color=MUTED, align="center")
    page.text("stable provenance", 76, 570, 206, 26, size=13, color=NAVY, bold=True, align="center")

    pair_card = page.vertex(
        "",
        374,
        122,
        320,
        370,
        "rounded=1;arcSize=6;html=1;fillColor=#FFFFFF;strokeColor=#D8E0E8;strokeWidth=2;",
    )
    page.image(asset_root / "muted_video_pair.jpg", 392, 142, 284, 80)
    page.image(asset_root / "reference_waveform.png", 392, 252, 125, 54)
    page.image(asset_root / "target_waveform.png", 551, 252, 125, 54)
    page.text("REF", 424, 312, 62, 24, size=12, color=BLUE, bold=True, align="center")
    page.text("TGT", 583, 312, 62, 24, size=12, color=GREEN, bold=True, align="center")
    _icon_badge(page, "eye", 420, 348, 48, color=BLUE, fill=PALE_BLUE)
    page.text("≈", 486, 349, 40, 45, size=28, color=NAVY, bold=True, align="center")
    _icon_badge(page, "eye", 545, 348, 48, color=GREEN, fill=PALE_GREEN)
    _icon_badge(page, "query", 494, 414, 58, color=AMBER, fill=PALE_AMBER)
    page.text("DOG BARK", 557, 426, 108, 30, size=14, color=AMBER, bold=True)
    page.edge(clip_a, pair_card, color=NAVY, width=3)
    page.edge(clip_b, pair_card, color=NAVY, width=3)

    page.text("independent views", 820, 82, 448, 25, size=13, color=MUTED, bold=True, align="center")
    gate_specs = (
        ("audio", "AUDIO", AMBER, PALE_AMBER, asset_root / "audio_only_pair.png"),
        ("mute", "MUTED", BLUE, PALE_BLUE, asset_root / "muted_video_pair.jpg"),
        ("av", "FULL AV", GREEN, PALE_GREEN, asset_root / "target_filmstrip.jpg"),
    )
    gate_ids: list[str] = []
    for index, (icon, label, color, fill, media) in enumerate(gate_specs):
        x = 742 + index * 196
        gate = page.vertex(
            "",
            x,
            126,
            168,
            318,
            (
                f"rounded=1;arcSize=7;html=1;fillColor={fill};"
                f"strokeColor={color};strokeWidth=2;"
            ),
        )
        gate_ids.append(gate)
        _icon_badge(page, icon, x + 54, 145, 60, color=color, fill="#FFFFFF")
        page.text(label, x + 24, 214, 120, 28, size=16, color=color, bold=True, align="center")
        page.image(media, x + 18, 258, 132, 58)
        page.text("REF", x + 24, 337, 45, 22, size=11, color=MUTED, bold=True, align="center")
        _check_badge(page, x + 75, 332, passed=index != 1)
        page.text("TGT", x + 24, 386, 45, 22, size=11, color=MUTED, bold=True, align="center")
        _check_badge(page, x + 75, 381, passed=True)
        page.text(
            ("direction" if index == 0 else "no shortcut" if index == 1 else "consistent"),
            x + 18,
            421,
            132,
            22,
            size=11,
            color=color,
            bold=True,
            align="center",
        )
        page.edge(pair_card, gate, color=color, width=2)

    funnel = _icon_badge(page, "filter", 1284, 266, 68, color=GREEN, fill=PALE_GREEN)
    for gate in gate_ids:
        page.edge(gate, funnel, color=GREEN, width=2)
    _check_badge(page, 1303, 342, passed=True)
    page.text("3 / 3", 1291, 381, 58, 25, size=15, color=GREEN, bold=True, align="center")
    page.text("PASS", 1284, 410, 72, 25, size=13, color=GREEN, bold=True, align="center")

    ledger = _icon_badge(page, "ledger", 1414, 108, 66, color=BLUE, fill=PALE_BLUE)
    page.text("LOG", 1410, 181, 74, 24, size=13, color=BLUE, bold=True, align="center")
    repeat = _icon_badge(page, "repeat", 1530, 108, 66, color=AMBER, fill=PALE_AMBER)
    page.text("REPEAT 20%", 1504, 181, 118, 24, size=13, color=AMBER, bold=True, align="center")
    dedup = _icon_badge(page, "dedup", 1648, 108, 66, color=BLUE, fill=PALE_BLUE)
    page.text("DEDUP", 1640, 181, 82, 24, size=13, color=BLUE, bold=True, align="center")
    page.edge(funnel, ledger, color=GREEN, width=3)
    page.edge(ledger, repeat, color="#AAB5BF", width=2)
    page.edge(repeat, dedup, color="#AAB5BF", width=2)

    freeze_card = page.vertex(
        "",
        1416,
        282,
        318,
        246,
        (
            f"rounded=1;arcSize=7;html=1;fillColor={NAVY};"
            f"strokeColor={NAVY};strokeWidth=2;"
        ),
    )
    _icon_badge(page, "lock", 1539, 307, 72, color="#FFFFFF", fill=NAVY, stroke="#FFFFFF")
    page.text("FULL1000", 1466, 390, 218, 42, size=25, color="#FFFFFF", bold=True, align="center")
    page.text("1K TARGET", 1454, 444, 115, 28, size=15, color="#FFFFFF", bold=True, align="center")
    page.text("+", 1569, 444, 24, 28, size=17, color=AMBER, bold=True, align="center")
    page.text("1K REF", 1594, 444, 96, 28, size=15, color="#FFFFFF", bold=True, align="center")
    page.edge(dedup, freeze_card, color=GREEN, width=3)
    page.text("source-disjoint", 1432, 550, 122, 24, size=12, color=GREEN, bold=True, align="center")
    page.text("SHA locked", 1583, 550, 122, 24, size=12, color=GREEN, bold=True, align="center")
    page.text(
        "automatically curated / model-verified",
        1418,
        584,
        316,
        24,
        size=12,
        color=MUTED,
        align="center",
    )
    return page


def build_figure_two(asset_root: Path, candidate_root: Path) -> Page:
    """Four-stage curation flow with short connectors and icon-led gates."""
    page = Page()

    def arrow_right(x: float, y: float, length: float, color: str, width: float = 3) -> None:
        page.line(x, y, length - 10, width, color=color)
        page.vertex(
            "",
            x + length - 13,
            y - 6,
            14,
            14,
            f"triangle;html=1;direction=east;fillColor={color};strokeColor=none;",
        )

    def arrow_down(x: float, y: float, length: float, color: str, width: float = 3) -> None:
        page.line(x, y, width, length - 10, color=color)
        page.vertex(
            "",
            x - 6,
            y + length - 13,
            14,
            14,
            f"triangle;html=1;direction=south;fillColor={color};strokeColor=none;",
        )

    stage_specs = (
        ("1", "CLIP", "scissors", 28, BLUE),
        ("2", "PAIR", "query", 372, AMBER),
        ("3", "VERIFY", "eye", 724, GREEN),
        ("4", "FREEZE", "lock", 1392, NAVY),
    )
    for number, title, icon, x, color in stage_specs:
        page.vertex(
            number,
            x,
            22,
            34,
            34,
            (
                f"ellipse;html=1;fillColor={color};strokeColor=#FFFFFF;"
                "strokeWidth=2;fontFamily=Helvetica;fontSize=17;"
                "fontStyle=1;fontColor=#FFFFFF;"
            ),
        )
        _icon_badge(page, icon, x + 45, 20, 40, color=color, fill="#FFFFFF")
        page.text(title, x + 96, 23, 170, 30, size=20, color=NAVY, bold=True)

    for x in (346, 708, 1376):
        page.line(x, 72, 2, 532, color=HAIRLINE)

    page.vertex(
        "",
        42,
        102,
        264,
        128,
        (
            f"rounded=1;arcSize=6;html=1;fillColor={PALE_BLUE};"
            f"strokeColor={BLUE};strokeWidth=2;"
        ),
    )
    page.image(asset_root / "reference_filmstrip.jpg", 54, 115, 240, 84)
    _icon_badge(page, "video", 57, 188, 38, color=BLUE, fill="#FFFFFF")
    page.text("RAW SOURCE", 103, 195, 130, 24, size=13, color=BLUE, bold=True)

    _icon_badge(page, "scissors", 146, 254, 58, color=AMBER, fill=PALE_AMBER)
    arrow_down(174, 231, 24, AMBER, 3)
    page.line(105, 326, 141, 3, color="#AAB5BF")
    page.line(105, 309, 3, 18, color="#AAB5BF")
    page.line(243, 309, 3, 18, color="#AAB5BF")
    arrow_down(104, 326, 22, BLUE, 2)
    arrow_down(242, 326, 22, GREEN, 2)

    page.vertex(
        "",
        42,
        353,
        126,
        90,
        f"rounded=1;html=1;fillColor={PALE_BLUE};strokeColor={BLUE};strokeWidth=2;",
    )
    page.image(asset_root / "reference_filmstrip.jpg", 50, 361, 110, 50)
    page.text("A", 91, 414, 28, 20, size=13, color=BLUE, bold=True, align="center")
    page.vertex(
        "",
        180,
        353,
        126,
        90,
        f"rounded=1;html=1;fillColor={PALE_GREEN};strokeColor={GREEN};strokeWidth=2;",
    )
    page.image(asset_root / "target_filmstrip.jpg", 188, 361, 110, 50)
    page.text("B", 229, 414, 28, 20, size=13, color=GREEN, bold=True, align="center")

    provenance_icons = (
        ("database", "data"),
        ("ledger", "source"),
        ("repeat", "time"),
        ("dedup", "group"),
    )
    for index, (icon, label) in enumerate(provenance_icons):
        x = 42 + index * 69
        _icon_badge(page, icon, x, 478, 42, color=BLUE, fill="#FFFFFF")
        page.text(label, x - 4, 526, 50, 20, size=10, color=MUTED, align="center")
    page.text("stable provenance", 76, 569, 198, 24, size=13, color=NAVY, bold=True, align="center")

    arrow_right(318, 330, 44, NAVY, 3)

    page.vertex(
        "",
        376,
        108,
        310,
        390,
        "rounded=1;arcSize=6;html=1;fillColor=#FFFFFF;strokeColor=#D8E0E8;strokeWidth=2;",
    )
    page.image(asset_root / "muted_video_pair.jpg", 394, 126, 274, 78)
    page.text("REF", 420, 210, 82, 22, size=12, color=BLUE, bold=True, align="center")
    page.text("TGT", 560, 210, 82, 22, size=12, color=GREEN, bold=True, align="center")
    page.image(asset_root / "reference_waveform.png", 394, 244, 122, 54)
    page.image(asset_root / "target_waveform.png", 546, 244, 122, 54)
    _icon_badge(page, "eye", 416, 324, 50, color=BLUE, fill=PALE_BLUE)
    page.text("SAME", 474, 335, 52, 24, size=10, color=NAVY, bold=True, align="center")
    _icon_badge(page, "eye", 537, 324, 50, color=GREEN, fill=PALE_GREEN)
    _icon_badge(page, "query", 426, 409, 58, color=AMBER, fill=PALE_AMBER)
    page.text("DOG BARK", 500, 423, 128, 30, size=15, color=AMBER, bold=True)
    page.text("VISUAL SAME / AUDIO DELTA", 421, 470, 220, 22, size=10, color=MUTED, bold=True, align="center")

    arrow_right(686, 330, 38, NAVY, 3)

    page.text("three independent views", 816, 82, 438, 24, size=13, color=MUTED, bold=True, align="center")
    lane_specs = (
        (112, "audio", "AUDIO", AMBER, PALE_AMBER, asset_root / "audio_only_pair.png", "direction"),
        (272, "mute", "MUTED", BLUE, PALE_BLUE, asset_root / "muted_video_pair.jpg", "no shortcut"),
        (432, "av", "FULL AV", GREEN, PALE_GREEN, asset_root / "target_filmstrip.jpg", "consistent"),
    )
    collector_x = 1284
    lane_centers: list[float] = []
    for y, icon, label, color, fill, media, verdict in lane_specs:
        page.vertex(
            "",
            738,
            y,
            516,
            126,
            (
                f"rounded=1;arcSize=7;html=1;fillColor={fill};"
                f"strokeColor={color};strokeWidth=2;"
            ),
        )
        _icon_badge(page, icon, 756, y + 31, 60, color=color, fill="#FFFFFF")
        page.image(media, 838, y + 25, 166, 74)
        page.text(label, 1026, y + 18, 118, 26, size=16, color=color, bold=True)
        page.text("REF", 1026, y + 58, 42, 22, size=11, color=MUTED, bold=True)
        _check_badge(page, 1075, y + 53, passed=label != "MUTED")
        page.text("TGT", 1120, y + 58, 42, 22, size=11, color=MUTED, bold=True)
        _check_badge(page, 1167, y + 53, passed=True)
        page.text(verdict, 1026, y + 91, 170, 22, size=11, color=color, bold=True)
        center_y = y + 63
        lane_centers.append(center_y)
        page.line(1254, center_y, collector_x - 1254, 3, color=color)
    page.line(collector_x, lane_centers[0], 3, lane_centers[-1] - lane_centers[0], color="#AAB5BF")
    _icon_badge(page, "filter", 1260, 274, 54, color=GREEN, fill=PALE_GREEN)
    arrow_right(1314, 300, 66, GREEN, 3)

    ledger = _icon_badge(page, "ledger", 1396, 272, 62, color=BLUE, fill=PALE_BLUE)
    repeat = _icon_badge(page, "repeat", 1492, 272, 62, color=AMBER, fill=PALE_AMBER)
    dedup = _icon_badge(page, "dedup", 1588, 272, 62, color=BLUE, fill=PALE_BLUE)
    freeze = _icon_badge(page, "lock", 1684, 264, 78, color="#FFFFFF", fill=NAVY, stroke=NAVY)
    arrow_right(1458, 302, 34, "#AAB5BF", 2)
    arrow_right(1554, 302, 34, "#AAB5BF", 2)
    arrow_right(1650, 302, 34, GREEN, 3)
    page.text("LOG", 1390, 343, 74, 23, size=12, color=BLUE, bold=True, align="center")
    page.text("REPEAT 20%", 1470, 343, 106, 23, size=12, color=AMBER, bold=True, align="center")
    page.text("DEDUP", 1580, 343, 78, 23, size=12, color=BLUE, bold=True, align="center")
    page.text("FULL1000", 1674, 350, 98, 25, size=14, color=NAVY, bold=True, align="center")

    page.vertex(
        "1K TARGET&nbsp;&nbsp;<font color='#D9A441'>+</font>&nbsp;&nbsp;1K REF",
        1432,
        412,
        320,
        78,
        (
            f"rounded=1;arcSize=8;html=1;fillColor={NAVY};strokeColor={NAVY};"
            "fontFamily=Helvetica;fontSize=20;fontStyle=1;fontColor=#FFFFFF;"
        ),
    )
    page.text("source-disjoint", 1435, 512, 132, 23, size=12, color=GREEN, bold=True, align="center")
    page.text("SHA locked", 1614, 512, 132, 23, size=12, color=GREEN, bold=True, align="center")
    page.text(
        "automatically curated / model-verified",
        1408,
        568,
        354,
        24,
        size=12,
        color=MUTED,
        align="center",
    )
    return page


def build_file(asset_root: Path, candidate_root: Path, output: Path) -> None:
    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "modified": "2026-07-23T00:00:00.000Z",
            "agent": "Codex + nature-figure",
            "version": "26.0.0",
            "type": "device",
            "compressed": "false",
        },
    )
    pages = [
        (
            "audio-cvr-figure-1",
            "Figure 1 - Reference Confusion",
            build_figure_one(asset_root, candidate_root),
        ),
        (
            "audio-cvr-figure-2",
            "Figure 2 - Automatic Curation",
            build_figure_two(asset_root, candidate_root),
        ),
    ]
    for page_id, name, page in pages:
        diagram = ET.SubElement(mxfile, "diagram", {"id": page_id, "name": name})
        diagram.append(page.model)
    ET.indent(mxfile, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(mxfile).write(output, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("paper/figures/assets/real_example"),
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("paper/figures/assets/test1000_candidates"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/drawio/audio_cvr_figures.drawio"),
    )
    args = parser.parse_args()
    build_file(args.asset_root, args.candidate_root, args.output)


if __name__ == "__main__":
    main()
