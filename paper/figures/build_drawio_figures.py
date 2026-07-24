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


def build_figure_one(asset_root: Path, candidate_root: Path) -> Page:
    page = Page()
    _label(page, "a", "A real directional audio edit", 28, 650)
    _label(page, "b", "The unchanged source is the trap", 710, 470)
    _label(page, "c", "Exact masking exposes the failure", 1215, 555)
    page.line(690, 24, 2, 602)
    page.line(1195, 24, 2, 602)

    ref_strip = page.image(
        asset_root / "reference_filmstrip.jpg", 44, 82, 610, 112, border=BLUE, border_width=2
    )
    page.vertex(
        "PRE-EDIT",
        56,
        91,
        92,
        26,
        (
            f"rounded=1;arcSize=22;html=1;fillColor={BLUE};strokeColor={BLUE};"
            "fontFamily=Helvetica;fontSize=13;fontStyle=1;fontColor=#FFFFFF;"
        ),
    )
    page.image(asset_root / "reference_waveform.png", 44, 201, 610, 76)
    edit = page.vertex(
        "add audience applause",
        190,
        293,
        320,
        42,
        (
            f"rounded=1;arcSize=40;html=1;fillColor={PALE_AMBER};strokeColor={AMBER};"
            f"strokeWidth=2;fontFamily=Helvetica;fontSize=17;fontStyle=1;fontColor={NAVY};"
        ),
    )
    target_strip = page.image(
        asset_root / "target_filmstrip.jpg", 44, 353, 610, 112, border=GREEN, border_width=2
    )
    page.vertex(
        "POST-EDIT",
        56,
        362,
        94,
        26,
        (
            f"rounded=1;arcSize=22;html=1;fillColor={GREEN};strokeColor={GREEN};"
            "fontFamily=Helvetica;fontSize=13;fontStyle=1;fontColor=#FFFFFF;"
        ),
    )
    page.image(asset_root / "target_waveform.png", 44, 472, 610, 76)
    page.edge(ref_strip, edit, color=AMBER, width=3)
    page.edge(edit, target_strip, color=AMBER, width=3)
    page.vertex(
        "EDIT ABSENT",
        68,
        565,
        178,
        34,
        (
            f"rounded=1;arcSize=25;html=1;fillColor={PALE_CORAL};strokeColor={CORAL};"
            f"fontFamily=Helvetica;fontSize=15;fontStyle=1;fontColor={CORAL};"
        ),
    )
    page.text(
        "same stage and performers",
        250,
        566,
        205,
        32,
        size=14,
        color=MUTED,
        align="center",
    )
    page.vertex(
        "EDIT PRESENT",
        458,
        565,
        178,
        34,
        (
            f"rounded=1;arcSize=25;html=1;fillColor={PALE_GREEN};strokeColor={GREEN};"
            f"fontFamily=Helvetica;fontSize=15;fontStyle=1;fontColor={GREEN};"
        ),
    )

    selected_dir = candidate_root / "05_c27cc098"
    ref_frame = selected_dir / "reference_frame_2.jpg"
    tgt_frame = selected_dir / "target_frame_2.jpg"
    page.text("WITH SOURCE", 718, 78, 205, 28, size=14, color=MUTED, bold=True)
    page.text("MASK OWN SOURCE", 962, 78, 210, 28, size=14, color=MUTED, bold=True)
    _thumbnail_frame(page, ref_frame, 730, 116, 190, 107, CORAL, "TOP-1")
    _thumbnail_frame(page, tgt_frame, 730, 256, 190, 107, GREEN, "TARGET")
    page.text(
        "pre-edit source wins",
        730,
        226,
        190,
        26,
        size=14,
        color=CORAL,
        bold=True,
        align="center",
    )
    masked_ref = _thumbnail_frame(page, ref_frame, 970, 116, 190, 107, CORAL, "MASKED")
    page.line(982, 166, 166, 5, color=CORAL)
    _thumbnail_frame(page, tgt_frame, 970, 256, 190, 107, GREEN, "TOP-1")
    page.text(
        "edited target rises",
        970,
        366,
        190,
        26,
        size=14,
        color=GREEN,
        bold=True,
        align="center",
    )
    mask = page.vertex(
        "mask one score",
        823,
        420,
        245,
        38,
        (
            f"rounded=1;arcSize=35;html=1;fillColor={PALE_AMBER};strokeColor={AMBER};"
            f"fontFamily=Helvetica;fontSize=15;fontStyle=1;fontColor={NAVY};"
        ),
    )
    page.edge(masked_ref, mask, color=AMBER, width=2, dashed=True)
    page.image(asset_root / "real_gallery.jpg", 735, 485, 420, 112)
    page.text(
        "same 2,000-item gallery",
        755,
        603,
        380,
        28,
        size=14,
        color=MUTED,
        bold=True,
        align="center",
    )

    page.vertex(
        "84.5-96.0 pp recovered",
        1230,
        78,
        510,
        58,
        (
            f"rounded=1;arcSize=8;html=1;fillColor={NAVY};strokeColor={NAVY};"
            "fontFamily=Helvetica;fontSize=23;fontStyle=1;fontColor=#FFFFFF;"
        ),
    )
    page.text("R@1 (%)", 1648, 147, 88, 25, size=13, color=MUTED, align="right")
    x0, x1 = 1395, 1710
    page.line(x0, 183, x1 - x0, 3, color=HAIRLINE)
    for value, x in ((0, x0), (50, (x0 + x1) / 2), (100, x1)):
        page.text(str(value), x - 18, 158, 36, 22, size=12, color=MUTED, align="center")
    rows = [
        ("E5  V+T", 5.94, 98.28),
        ("E5  V+A+T", 12.78, 97.26),
        ("ImageBind  V+T", 11.7, 99.3),
        ("ImageBind  V+A+T", 2.5, 98.5),
    ]
    for index, (name, before, after) in enumerate(rows):
        y = 235 + index * 88
        page.text(name, 1218, y - 15, 160, 30, size=15, color=TEXT, bold=True, align="right")
        bx = x0 + (x1 - x0) * before / 100
        ax = x0 + (x1 - x0) * after / 100
        page.line(bx, y, max(3, ax - bx), 4, color="#ABB7C3")
        page.vertex(
            f"{before:.1f}",
            bx - 19,
            y - 19,
            38,
            38,
            (
                f"ellipse;html=1;fillColor={CORAL};strokeColor=#FFFFFF;strokeWidth=2;"
                "fontFamily=Helvetica;fontSize=11;fontStyle=1;fontColor=#FFFFFF;"
            ),
        )
        page.vertex(
            f"{after:.1f}",
            ax - 22,
            y - 22,
            44,
            44,
            (
                f"ellipse;html=1;fillColor={GREEN};strokeColor=#FFFFFF;strokeWidth=2;"
                "fontFamily=Helvetica;fontSize=11;fontStyle=1;fontColor=#FFFFFF;"
            ),
        )
    page.vertex(
        "",
        1245,
        591,
        16,
        16,
        f"ellipse;html=1;fillColor={CORAL};strokeColor=none;",
    )
    page.text("with source", 1268, 585, 120, 28, size=13, color=MUTED)
    page.vertex(
        "",
        1435,
        591,
        16,
        16,
        f"ellipse;html=1;fillColor={GREEN};strokeColor=none;",
    )
    page.text("mask own source", 1458, 585, 155, 28, size=13, color=MUTED)
    return page


def build_figure_two(asset_root: Path, candidate_root: Path) -> Page:
    page = Page()
    page.text(
        "Natural source video",
        30,
        24,
        295,
        34,
        size=21,
        color=NAVY,
        bold=True,
    )
    page.text("Audio-first pair", 365, 24, 280, 34, size=21, color=NAVY, bold=True)
    page.text("Three independent gates", 700, 24, 420, 34, size=21, color=NAVY, bold=True)
    page.text("Audit and split", 1160, 24, 245, 34, size=21, color=NAVY, bold=True)
    page.text("Frozen benchmark", 1445, 24, 330, 34, size=21, color=NAVY, bold=True)
    for x in (340, 675, 1135, 1420):
        page.line(x, 24, 2, 602)

    source = page.image(asset_root / "reference_filmstrip.jpg", 30, 92, 285, 105)
    page.text(
        "6-9 s source clips",
        55,
        205,
        235,
        28,
        size=15,
        color=MUTED,
        bold=True,
        align="center",
    )
    page.image(asset_root / "target_filmstrip.jpg", 30, 252, 285, 105)
    page.line(54, 393, 238, 5, color=HAIRLINE)
    page.vertex(
        "",
        82,
        383,
        12,
        25,
        f"rounded=1;html=1;fillColor={BLUE};strokeColor=none;",
    )
    page.vertex(
        "",
        245,
        383,
        12,
        25,
        f"rounded=1;html=1;fillColor={GREEN};strokeColor=none;",
    )
    page.text("same raw source", 66, 414, 210, 30, size=14, color=MUTED, align="center")
    page.vertex(
        "source-aware sampling",
        54,
        500,
        238,
        40,
        (
            f"rounded=1;arcSize=35;html=1;fillColor={PALE_BLUE};strokeColor={BLUE};"
            f"fontFamily=Helvetica;fontSize=15;fontStyle=1;fontColor={NAVY};"
        ),
    )

    pair = page.image(asset_root / "muted_video_pair.jpg", 372, 92, 265, 142)
    page.text("reference", 370, 238, 125, 24, size=13, color=BLUE, bold=True, align="center")
    page.text("target", 515, 238, 125, 24, size=13, color=GREEN, bold=True, align="center")
    page.image(asset_root / "audio_only_pair.png", 372, 280, 265, 90)
    page.vertex(
        "add audience applause",
        382,
        403,
        245,
        42,
        (
            f"rounded=1;arcSize=35;html=1;fillColor={PALE_AMBER};strokeColor={AMBER};"
            f"fontFamily=Helvetica;fontSize=15;fontStyle=1;fontColor={NAVY};"
        ),
    )
    page.text(
        "source x edit -> target",
        390,
        494,
        230,
        28,
        size=15,
        color=MUTED,
        bold=True,
        align="center",
    )
    page.edge(source, pair, color=NAVY, width=3)

    gate_audio = page.vertex(
        "",
        710,
        80,
        400,
        145,
        f"rounded=1;arcSize=5;html=1;fillColor={PALE_BLUE};strokeColor={BLUE};strokeWidth=2;",
    )
    page.text("1  Audio only", 728, 89, 150, 26, size=16, color=BLUE, bold=True)
    page.image(asset_root / "audio_only_pair.png", 730, 121, 350, 82)
    page.vertex(
        "direction",
        1000,
        88,
        90,
        25,
        (
            f"rounded=1;arcSize=20;html=1;fillColor={BLUE};strokeColor={BLUE};"
            "fontFamily=Helvetica;fontSize=12;fontStyle=1;fontColor=#FFFFFF;"
        ),
    )
    gate_video = page.vertex(
        "",
        710,
        246,
        400,
        145,
        f"rounded=1;arcSize=5;html=1;fillColor=#F4F6F8;strokeColor=#8997A5;strokeWidth=2;",
    )
    page.text("2  Muted video", 728, 255, 160, 26, size=16, color=TEXT, bold=True)
    page.image(asset_root / "muted_video_pair.jpg", 730, 289, 350, 83)
    page.vertex(
        "no shortcut",
        992,
        254,
        98,
        25,
        (
            "rounded=1;arcSize=20;html=1;fillColor=#6D7D8B;strokeColor=#6D7D8B;"
            "fontFamily=Helvetica;fontSize=12;fontStyle=1;fontColor=#FFFFFF;"
        ),
    )
    gate_full = page.vertex(
        "",
        710,
        412,
        400,
        145,
        f"rounded=1;arcSize=5;html=1;fillColor={PALE_GREEN};strokeColor={GREEN};strokeWidth=2;",
    )
    page.text("3  Full AV", 728, 421, 150, 26, size=16, color=GREEN, bold=True)
    page.image(asset_root / "target_filmstrip.jpg", 730, 455, 350, 64)
    page.image(asset_root / "target_waveform.png", 730, 518, 350, 31)
    page.vertex(
        "consistent",
        1000,
        420,
        90,
        25,
        (
            f"rounded=1;arcSize=20;html=1;fillColor={GREEN};strokeColor={GREEN};"
            "fontFamily=Helvetica;fontSize=12;fontStyle=1;fontColor=#FFFFFF;"
        ),
    )
    page.edge(pair, gate_audio, color=NAVY, width=3)
    page.vertex(
        "reject",
        850,
        587,
        110,
        32,
        (
            f"rounded=1;arcSize=30;html=1;fillColor={PALE_CORAL};strokeColor={CORAL};"
            f"fontFamily=Helvetica;fontSize=14;fontStyle=1;fontColor={CORAL};"
        ),
    )
    page.edge(gate_audio, gate_video, color=BLUE, width=2)
    page.edge(gate_video, gate_full, color=GREEN, width=2)

    page.vertex(
        "20%",
        1205,
        95,
        120,
        120,
        (
            f"ellipse;html=1;fillColor={PALE_AMBER};strokeColor={AMBER};strokeWidth=3;"
            f"fontFamily=Helvetica;fontSize=28;fontStyle=1;fontColor={NAVY};"
        ),
    )
    page.text(
        "sampled repeat audit",
        1175,
        222,
        180,
        32,
        size=15,
        color=MUTED,
        bold=True,
        align="center",
    )
    page.vertex(
        "STABILITY AUDIT",
        1185,
        295,
        160,
        42,
        (
            f"rounded=1;arcSize=35;html=1;fillColor={PALE_GREEN};strokeColor={GREEN};"
            f"fontFamily=Helvetica;fontSize=15;fontStyle=1;fontColor={GREEN};"
        ),
    )
    page.line(1210, 366, 110, 4, color=HAIRLINE)
    page.vertex(
        "sample",
        1170,
        400,
        80,
        34,
        (
            f"rounded=1;arcSize=20;html=1;fillColor={PALE_BLUE};strokeColor={BLUE};"
            f"fontFamily=Helvetica;fontSize=13;fontStyle=1;fontColor={NAVY};"
        ),
    )
    page.vertex(
        "pair",
        1260,
        400,
        80,
        34,
        (
            f"rounded=1;arcSize=20;html=1;fillColor={PALE_BLUE};strokeColor={BLUE};"
            f"fontFamily=Helvetica;fontSize=13;fontStyle=1;fontColor={NAVY};"
        ),
    )
    page.vertex(
        "source",
        1215,
        447,
        80,
        34,
        (
            f"rounded=1;arcSize=20;html=1;fillColor={PALE_BLUE};strokeColor={BLUE};"
            f"fontFamily=Helvetica;fontSize=13;fontStyle=1;fontColor={NAVY};"
        ),
    )
    page.text(
        "deduplicate",
        1190,
        490,
        145,
        26,
        size=14,
        color=MUTED,
        bold=True,
        align="center",
    )
    page.vertex(
        "TRAIN",
        1170,
        548,
        72,
        34,
        (
            f"rounded=1;arcSize=20;html=1;fillColor={PALE_BLUE};strokeColor={BLUE};"
            f"fontFamily=Helvetica;fontSize=12;fontStyle=1;fontColor={NAVY};"
        ),
    )
    page.vertex(
        "VAL",
        1250,
        548,
        55,
        34,
        (
            f"rounded=1;arcSize=20;html=1;fillColor={PALE_AMBER};strokeColor={AMBER};"
            f"fontFamily=Helvetica;fontSize=12;fontStyle=1;fontColor={NAVY};"
        ),
    )
    split = page.vertex(
        "TEST",
        1313,
        548,
        72,
        34,
        (
            f"rounded=1;arcSize=20;html=1;fillColor={PALE_GREEN};strokeColor={GREEN};"
            f"fontFamily=Helvetica;fontSize=12;fontStyle=1;fontColor={GREEN};"
        ),
    )
    page.edge(gate_full, split, color=GREEN, width=3)

    gallery = page.image(asset_root / "real_gallery.jpg", 1450, 90, 300, 114)
    full_benchmark = page.vertex(
        "FULL1000",
        1480,
        245,
        240,
        74,
        (
            f"rounded=1;arcSize=8;html=1;fillColor={NAVY};strokeColor={NAVY};"
            "fontFamily=Helvetica;fontSize=30;fontStyle=1;fontColor=#FFFFFF;"
        ),
    )
    page.text(
        "1000 targets + 1000 sources",
        1460,
        326,
        280,
        32,
        size=15,
        color=MUTED,
        bold=True,
        align="center",
    )
    page.vertex(
        "SHA256 LOCKED",
        1490,
        392,
        220,
        42,
        (
            f"rounded=1;arcSize=35;html=1;fillColor={PALE_GREEN};strokeColor={GREEN};"
            f"fontFamily=Helvetica;fontSize=15;fontStyle=1;fontColor={GREEN};"
        ),
    )
    page.text(
        "model-verified",
        1480,
        471,
        240,
        28,
        size=15,
        color=TEXT,
        bold=True,
        align="center",
    )
    page.text(
        "source-disjoint",
        1480,
        505,
        240,
        28,
        size=15,
        color=TEXT,
        bold=True,
        align="center",
    )
    page.text(
        "selection independent of retrieval scores",
        1450,
        570,
        300,
        35,
        size=13,
        color=MUTED,
        align="center",
    )
    page.edge(split, full_benchmark, color=GREEN, width=3, dashed=True)
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
