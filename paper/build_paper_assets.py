from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "results_manifest.json"
TEMPLATE_PATH = ROOT / "main.template.tex"
MAIN_PATH = ROOT / "main.tex"
ABSTRACT_PATH = ROOT / "aaai_audiocvr_abstract.tex"
FIGURE_DIR = ROOT / "figures" / "generated"
GENERATED_DIR = ROOT / "generated"

COLORS = {
    "ink": "#17212B",
    "muted": "#5C6773",
    "line": "#AAB4BE",
    "paper": "#FFFFFF",
    "reference": "#4C78A8",
    "target": "#2A9D78",
    "audio": "#E9A23B",
    "reject": "#C85C5C",
    "soft_blue": "#E8F0F7",
    "soft_green": "#E4F3EE",
    "soft_orange": "#FAEFD9",
    "soft_gray": "#F1F3F5",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def percent(value: Any, decimals: int = 1) -> str:
    if value is None:
        return "TBD"
    return f"{100.0 * float(value):.{decimals}f}"


def latex_scientific(value: float) -> str:
    mantissa, exponent = f"{float(value):.0e}".split("e")
    exponent_value = int(exponent)
    if mantissa == "1":
        return f"10^{{{exponent_value}}}"
    return f"{mantissa}\\times 10^{{{exponent_value}}}"


def percent_mean_std(item: dict[str, Any], mean_key: str, std_key: str) -> str:
    mean = item.get(mean_key)
    std = item.get(std_key)
    if mean is None:
        return "TBD"
    if std is None:
        return percent(mean)
    return f"{percent(mean)} $\\pm$ {percent(std)}"


def p_value(value: Any) -> str:
    if value is None:
        return "TBD"
    if float(value) <= 0.001:
        return "$<0.001$"
    return f"{float(value):.3f}"


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str = "",
    facecolor: str = "#F1F3F5",
    edgecolor: str = "#AAB4BE",
    title_color: str = "#17212B",
    linewidth: float = 1.0,
    title_fontsize: float = 7.0,
    subtitle_fontsize: float = 6.0,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.62,
        title,
        ha="center",
        va="center",
        fontsize=title_fontsize,
        fontweight="bold",
        color=title_color,
    )
    if subtitle:
        ax.text(
            x + width / 2,
            y + height * 0.28,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_fontsize,
            color=COLORS["muted"],
            linespacing=1.15,
        )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#5C6773",
    linewidth: float = 1.2,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=linewidth,
            color=color,
            shrinkA=1,
            shrinkB=1,
            clip_on=False,
        )
    )


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(
        0.0,
        1.02,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.09,
        1.02,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
        color=COLORS["ink"],
    )


def rounded_panel(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    facecolor: str = "#FFFFFF",
    edgecolor: str = "#D4DAE0",
    linewidth: float = 0.8,
    radius: float = 0.018,
    zorder: float = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def draw_waveform(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    color: str,
    variant: int,
    linewidth: float = 0.75,
) -> None:
    samples = np.linspace(0.0, 1.0, 160)
    carrier = np.sin((14 + variant * 3) * np.pi * samples)
    harmonic = 0.42 * np.sin((31 + variant * 5) * np.pi * samples + 0.7)
    envelope = 0.35 + 0.65 * np.sin(np.pi * samples) ** 2
    values = (carrier + harmonic) * envelope
    values /= max(float(np.max(np.abs(values))), 1e-8)
    ax.plot(
        x + samples * width,
        y + height / 2 + values * height * 0.46,
        color=color,
        linewidth=linewidth,
        solid_capstyle="round",
        clip_on=False,
        zorder=4,
    )
    ax.plot(
        [x, x + width],
        [y + height / 2, y + height / 2],
        color=color,
        linewidth=0.25,
        alpha=0.35,
        zorder=3,
    )


def draw_video_strip(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    edgecolor: str,
    label: str,
) -> None:
    rounded_panel(
        ax,
        x,
        y,
        width,
        height,
        facecolor="#FFFFFF",
        edgecolor=edgecolor,
        linewidth=1.0,
        radius=0.014,
        zorder=2,
    )
    gutter = width * 0.035
    frame_width = (width - gutter * 4) / 3
    frame_y = y + height * 0.22
    frame_height = height * 0.62
    for index in range(3):
        frame_x = x + gutter + index * (frame_width + gutter)
        ax.add_patch(
            Rectangle(
                (frame_x, frame_y),
                frame_width,
                frame_height,
                facecolor="#DDE7ED",
                edgecolor="#6B7680",
                linewidth=0.45,
                zorder=3,
            )
        )
        ax.add_patch(
            Rectangle(
                (frame_x, frame_y),
                frame_width,
                frame_height * 0.34,
                facecolor="#BBC9C2",
                edgecolor="none",
                zorder=3,
            )
        )
        person_x = frame_x + frame_width * (0.35 + 0.08 * index)
        ax.add_patch(
            Circle(
                (person_x, frame_y + frame_height * 0.59),
                frame_width * 0.09,
                facecolor="#354A5E",
                edgecolor="none",
                zorder=4,
            )
        )
        ax.add_patch(
            Rectangle(
                (
                    person_x - frame_width * 0.07,
                    frame_y + frame_height * 0.31,
                ),
                frame_width * 0.14,
                frame_height * 0.24,
                facecolor="#486A79",
                edgecolor="none",
                zorder=4,
            )
        )
        ax.add_patch(
            Rectangle(
                (
                    frame_x + frame_width * 0.57,
                    frame_y + frame_height * 0.20,
                ),
                frame_width * 0.31,
                frame_height * 0.16,
                facecolor="#8B6E55",
                edgecolor="#66503D",
                linewidth=0.25,
                zorder=4,
            )
        )
        for key in range(4):
            key_x = frame_x + frame_width * (0.585 + key * 0.067)
            ax.plot(
                [key_x, key_x],
                [
                    frame_y + frame_height * 0.205,
                    frame_y + frame_height * 0.345,
                ],
                color="#EFECE6",
                linewidth=0.5,
                zorder=5,
            )
    ax.text(
        x + width / 2,
        y + height * 0.105,
        label,
        ha="center",
        va="center",
        fontsize=5.4,
        fontweight="bold",
        color=edgecolor,
        zorder=5,
    )


def draw_status_tag(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    width: float = 0.18,
) -> None:
    rounded_panel(
        ax,
        x,
        y,
        width,
        0.065,
        facecolor=mpl.colors.to_rgba(color, 0.10),
        edgecolor=color,
        linewidth=0.7,
        radius=0.012,
        zorder=3,
    )
    ax.text(
        x + width / 2,
        y + 0.032,
        text,
        ha="center",
        va="center",
        fontsize=5.1,
        fontweight="bold",
        color=color,
        zorder=4,
    )


def draw_figure1() -> None:
    configure_matplotlib()
    fig = plt.figure(figsize=(7.15, 2.72))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.08, 1.34], wspace=0.20)

    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "a", "Audio-primary directional query")
    draw_video_strip(
        ax,
        0.02,
        0.49,
        0.38,
        0.30,
        COLORS["reference"],
        "reference video",
    )
    draw_video_strip(
        ax,
        0.60,
        0.49,
        0.38,
        0.30,
        COLORS["target"],
        "target video",
    )
    draw_waveform(ax, 0.04, 0.36, 0.34, 0.075, COLORS["reference"], 0)
    draw_waveform(ax, 0.62, 0.36, 0.34, 0.075, COLORS["target"], 2)
    ax.text(
        0.21,
        0.315,
        "piano",
        ha="center",
        va="center",
        fontsize=5.8,
        fontweight="bold",
        color=COLORS["reference"],
    )
    ax.text(
        0.79,
        0.315,
        "guitar",
        ha="center",
        va="center",
        fontsize=5.8,
        fontweight="bold",
        color=COLORS["target"],
    )
    arrow(ax, (0.41, 0.63), (0.59, 0.63), COLORS["audio"], 1.5)
    rounded_panel(
        ax,
        0.405,
        0.805,
        0.19,
        0.085,
        facecolor=COLORS["soft_orange"],
        edgecolor=COLORS["audio"],
        linewidth=0.8,
        radius=0.014,
        zorder=4,
    )
    ax.text(
        0.50,
        0.847,
        "replace piano\nwith guitar",
        ha="center",
        va="center",
        fontsize=5.2,
        fontweight="bold",
        color="#8A5B13",
        linespacing=1.05,
        zorder=5,
    )
    draw_status_tag(ax, 0.11, 0.18, "edit not satisfied", COLORS["reject"], 0.24)
    draw_status_tag(ax, 0.65, 0.18, "edit satisfied", COLORS["target"], 0.24)
    ax.plot([0.09, 0.91], [0.11, 0.11], color=COLORS["line"], linewidth=0.65)
    ax.plot([0.09, 0.09], [0.10, 0.135], color=COLORS["line"], linewidth=0.65)
    ax.plot([0.91, 0.91], [0.10, 0.135], color=COLORS["line"], linewidth=0.65)
    ax.text(
        0.50,
        0.065,
        "visual context preserved",
        ha="center",
        va="center",
        fontsize=5.3,
        color=COLORS["muted"],
    )

    ax = fig.add_subplot(grid[0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "b", "Exact own-reference masking")
    ax.text(
        0.23,
        0.88,
        "with reference",
        ha="center",
        va="center",
        fontsize=5.6,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.78,
        0.88,
        "masked reference",
        ha="center",
        va="center",
        fontsize=5.6,
        fontweight="bold",
        color=COLORS["ink"],
    )

    def rank_item(
        x: float,
        y: float,
        rank: str,
        label: str,
        fill: str,
        edge: str,
        muted: bool = False,
    ) -> None:
        rounded_panel(
            ax,
            x,
            y,
            0.39,
            0.125,
            facecolor=fill,
            edgecolor=edge,
            linewidth=0.8,
            radius=0.012,
            zorder=2,
        )
        ax.add_patch(
            Circle(
                (x + 0.055, y + 0.062),
                0.026,
                facecolor=edge,
                edgecolor="none",
                zorder=3,
            )
        )
        ax.text(
            x + 0.055,
            y + 0.062,
            rank,
            ha="center",
            va="center",
            fontsize=5.0,
            fontweight="bold",
            color="#FFFFFF",
            zorder=4,
        )
        ax.text(
            x + 0.105,
            y + 0.062,
            label,
            ha="left",
            va="center",
            fontsize=5.3,
            fontweight="bold" if "target" in label or "reference" in label else "normal",
            color=COLORS["muted"] if muted else COLORS["ink"],
            zorder=4,
        )

    rank_item(0.02, 0.69, "1", "own reference", COLORS["soft_blue"], COLORS["reference"])
    rank_item(0.02, 0.53, "2", "target", COLORS["soft_green"], COLORS["target"])
    rank_item(0.02, 0.37, "3", "other candidate", COLORS["soft_gray"], COLORS["line"])
    rank_item(0.59, 0.69, "-", "own reference", "#FBE9E8", COLORS["reject"], True)
    ax.plot([0.64, 0.91], [0.755, 0.755], color=COLORS["reject"], linewidth=1.4, zorder=5)
    rank_item(0.59, 0.53, "1", "target", COLORS["soft_green"], COLORS["target"])
    rank_item(0.59, 0.37, "2", "other candidate", COLORS["soft_gray"], COLORS["line"])

    arrow(ax, (0.43, 0.61), (0.57, 0.61), COLORS["audio"], 1.35)
    ax.text(
        0.50,
        0.68,
        "$s_{i\\rho(i)}\\!\\leftarrow\\!-\\infty$",
        ha="center",
        va="center",
        fontsize=5.5,
        fontweight="bold",
        color="#8A5B13",
    )
    rounded_panel(
        ax,
        0.08,
        0.16,
        0.84,
        0.115,
        facecolor="#F8FAFB",
        edgecolor=COLORS["line"],
        linewidth=0.7,
        radius=0.012,
    )
    ax.text(
        0.50,
        0.218,
        "same score row, same 1,000 queries",
        ha="center",
        va="center",
        fontsize=5.4,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.50,
        0.176,
        "2,000 gallery items $\\rightarrow$ 1,999 effective items",
        ha="center",
        va="center",
        fontsize=5.1,
        color=COLORS["muted"],
    )

    ax = fig.add_subplot(grid[0, 2])
    ax.set_xlim(-42, 104)
    ax.set_ylim(0.0, 4.45)
    ax.axis("off")
    panel_label(ax, "c", "Cross-model exact-reference failure")
    rows = [
        ("E5 adapter  V+A+T", 12.78, 97.26, 3.25),
        ("ImageBind  V+A+T", 2.50, 98.50, 2.15),
        ("OmniEmbed  V+A+T", 0.00, 99.20, 1.05),
    ]
    ax.add_patch(
        Rectangle(
            (0, 0.30),
            16,
            3.55,
            facecolor="#FCEDEA",
            edgecolor="none",
            alpha=0.65,
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (94, 0.30),
            8,
            3.55,
            facecolor="#E6F4EF",
            edgecolor="none",
            alpha=0.85,
            zorder=0,
        )
    )
    for label, before, after, y in rows:
        ax.plot(
            [before, after],
            [y, y],
            color="#9DA9B4",
            linewidth=2.1,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            [before],
            [y],
            s=30,
            color=COLORS["reject"],
            edgecolor="#FFFFFF",
            linewidth=0.55,
            zorder=4,
        )
        ax.scatter(
            [after],
            [y],
            s=34,
            color=COLORS["target"],
            edgecolor="#FFFFFF",
            linewidth=0.55,
            zorder=4,
        )
        ax.text(
            -5,
            y,
            label,
            ha="right",
            va="center",
            fontsize=5.35,
            fontweight="bold",
            color=COLORS["ink"],
        )
        ax.text(
            before + 1.5,
            y + 0.16,
            f"{before:.1f}",
            ha="left",
            va="bottom",
            fontsize=5.0,
            color=COLORS["reject"],
        )
        ax.text(
            after - 1.5,
            y + 0.16,
            f"{after:.1f}",
            ha="right",
            va="bottom",
            fontsize=5.0,
            color=COLORS["target"],
        )
    ax.plot([0, 100], [0.18, 0.18], color=COLORS["line"], linewidth=0.65)
    for tick in [0, 50, 100]:
        ax.plot([tick, tick], [0.14, 0.22], color=COLORS["line"], linewidth=0.65)
        ax.text(
            tick,
            0.01,
            str(tick),
            ha="center",
            va="top",
            fontsize=5.0,
            color=COLORS["muted"],
        )
    ax.text(
        50,
        -0.20,
        "R@1 (%)",
        ha="center",
        va="top",
        fontsize=5.1,
        color=COLORS["muted"],
    )
    ax.scatter([7], [4.05], s=22, color=COLORS["reject"], zorder=4)
    ax.text(12, 4.05, "with reference", ha="left", va="center", fontsize=5.0)
    ax.scatter([59], [4.05], s=22, color=COLORS["target"], zorder=4)
    ax.text(64, 4.05, "mask own reference", ha="left", va="center", fontsize=5.0)
    ax.plot([-38, 102], [2.02, 2.02], color="#D6DCE1", linewidth=0.55)

    fig.savefig(
        FIGURE_DIR / "figure1_reference_confusion.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        FIGURE_DIR / "figure1_reference_confusion.svg",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        FIGURE_DIR / "figure1_reference_confusion.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        FIGURE_DIR / "figure1_reference_confusion.tiff",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def draw_figure2() -> None:
    configure_matplotlib()
    fig = plt.figure(figsize=(7.15, 2.80))
    grid = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.12, 1.68, 1.02], wspace=0.12)

    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "a", "Source-aware clips")
    ax.plot([0.08, 0.92], [0.77, 0.77], color=COLORS["line"], linewidth=1.0)
    for index in range(6):
        x = 0.08 + index * 0.14
        ax.add_patch(
            Rectangle(
                (x, 0.69),
                0.105,
                0.16,
                facecolor="#DDE7ED" if index < 3 else "#E8E0D6",
                edgecolor="#7E8993",
                linewidth=0.45,
            )
        )
        ax.add_patch(
            Circle(
                (x + 0.040 + 0.005 * index, 0.77),
                0.012,
                facecolor="#3F596B",
                edgecolor="none",
                zorder=3,
            )
        )
        ax.add_patch(
            Rectangle(
                (x + 0.032 + 0.005 * index, 0.72),
                0.016,
                0.038,
                facecolor="#4E7582",
                edgecolor="none",
                zorder=3,
            )
        )
    ax.plot([0.075, 0.49], [0.64, 0.64], color=COLORS["reference"], linewidth=1.1)
    ax.plot([0.075, 0.075], [0.62, 0.66], color=COLORS["reference"], linewidth=1.1)
    ax.plot([0.49, 0.49], [0.62, 0.66], color=COLORS["reference"], linewidth=1.1)
    ax.text(0.282, 0.595, "6-9 s", ha="center", va="center", fontsize=5.0, color=COLORS["reference"])
    ax.plot([0.51, 0.925], [0.64, 0.64], color=COLORS["target"], linewidth=1.1)
    ax.plot([0.51, 0.51], [0.62, 0.66], color=COLORS["target"], linewidth=1.1)
    ax.plot([0.925, 0.925], [0.62, 0.66], color=COLORS["target"], linewidth=1.1)
    ax.text(0.717, 0.595, "6-9 s", ha="center", va="center", fontsize=5.0, color=COLORS["target"])
    rounded_panel(ax, 0.08, 0.27, 0.84, 0.25, "#F7F9FA", COLORS["line"], 0.7, 0.012)
    ax.text(0.13, 0.465, "provenance", ha="left", va="center", fontsize=5.3, fontweight="bold")
    metadata = [
        "dataset + raw source",
        "segment index + time",
        "source-disjoint group",
    ]
    for index, text_value in enumerate(metadata):
        y = 0.405 - index * 0.055
        ax.add_patch(Circle((0.145, y), 0.009, facecolor=COLORS["reference"], edgecolor="none"))
        ax.text(0.18, y, text_value, ha="left", va="center", fontsize=5.0, color=COLORS["muted"])
    ax.text(0.50, 0.17, "long videos yield grouped windows", ha="center", va="center", fontsize=5.0, color=COLORS["muted"])
    arrow(ax, (0.94, 0.50), (1.07, 0.50), COLORS["line"], 1.0)

    ax = fig.add_subplot(grid[0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "b", "Audio-first pairing")
    draw_video_strip(ax, 0.08, 0.61, 0.84, 0.23, COLORS["reference"], "candidate A")
    draw_video_strip(ax, 0.08, 0.31, 0.84, 0.23, COLORS["target"], "candidate B")
    draw_waveform(ax, 0.13, 0.545, 0.74, 0.048, COLORS["reference"], 0, 0.55)
    draw_waveform(ax, 0.13, 0.245, 0.74, 0.048, COLORS["target"], 3, 0.55)
    ax.text(0.50, 0.18, "high visual context similarity", ha="center", va="center", fontsize=5.0, color=COLORS["muted"])
    ax.text(0.50, 0.125, "clear speech / music / event delta", ha="center", va="center", fontsize=5.0, fontweight="bold", color="#8A5B13")
    arrow(ax, (0.94, 0.50), (1.07, 0.50), COLORS["line"], 1.0)

    ax = fig.add_subplot(grid[0, 2])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "c", "Modality-isolated verification")
    lanes = [
        (
            0.66,
            "AUDIO ONLY",
            "audible delta; generate edit\nreference fails, target satisfies",
            COLORS["soft_orange"],
            COLORS["audio"],
        ),
        (
            0.39,
            "MUTED VIDEO",
            "same visual context\ntarget not identifiable without sound",
            COLORS["soft_blue"],
            COLORS["reference"],
        ),
        (
            0.12,
            "FULL AUDIOVISUAL",
            "edit remains valid in context\nreject transcript / ASR shortcuts",
            COLORS["soft_green"],
            COLORS["target"],
        ),
    ]
    for y, lane_title, lane_text, fill, edge in lanes:
        rounded_panel(ax, 0.03, y, 0.94, 0.205, fill, edge, 0.8, 0.014)
        ax.add_patch(Circle((0.105, y + 0.102), 0.043, facecolor=edge, edgecolor="none", zorder=3))
        if lane_title == "AUDIO ONLY":
            draw_waveform(ax, 0.075, y + 0.082, 0.060, 0.040, "#FFFFFF", 1, 0.55)
        elif lane_title == "MUTED VIDEO":
            ax.add_patch(Rectangle((0.080, y + 0.075), 0.050, 0.052, facecolor="none", edgecolor="#FFFFFF", linewidth=0.7, zorder=4))
            ax.plot([0.079, 0.132], [y + 0.073, y + 0.130], color="#FFFFFF", linewidth=1.0, zorder=5)
        else:
            ax.add_patch(Polygon([[0.088, y + 0.077], [0.088, y + 0.127], [0.126, y + 0.102]], closed=True, facecolor="#FFFFFF", edgecolor="none", zorder=4))
        ax.text(0.175, y + 0.142, lane_title, ha="left", va="center", fontsize=5.35, fontweight="bold", color=edge)
        ax.text(
            0.175,
            y + 0.073,
            lane_text,
            ha="left",
            va="center",
            fontsize=5.0,
            color=COLORS["ink"],
            linespacing=1.18,
        )
        draw_status_tag(ax, 0.82, y + 0.066, "PASS", edge, 0.11)
    ax.text(
        0.50,
        0.045,
        "A pass at one gate cannot compensate for a failure at another.",
        ha="center",
        va="center",
        fontsize=5.0,
        color=COLORS["muted"],
    )
    arrow(ax, (0.95, 0.50), (1.06, 0.50), COLORS["line"], 1.0)

    ax = fig.add_subplot(grid[0, 3])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "d", "Audit and freeze")
    rounded_panel(ax, 0.08, 0.66, 0.84, 0.20, "#F7F9FA", COLORS["line"], 0.7, 0.012)
    ax.text(0.50, 0.815, "atomic review ledger", ha="center", va="center", fontsize=5.15, fontweight="bold")
    for index, text_value in enumerate(["decision + confidence", "rejection + reviewer", "media + provenance"]):
        y = 0.765 - index * 0.047
        ax.plot([0.16, 0.24], [y, y], color=COLORS["target"], linewidth=1.0)
        ax.text(0.28, y, text_value, ha="left", va="center", fontsize=5.0, color=COLORS["muted"])
    arrow(ax, (0.50, 0.645), (0.50, 0.565), COLORS["line"], 0.9)
    rounded_panel(ax, 0.08, 0.39, 0.84, 0.16, COLORS["soft_blue"], COLORS["reference"], 0.75, 0.012)
    ax.text(0.50, 0.495, "canonical deduplication", ha="center", va="center", fontsize=5.2, fontweight="bold", color=COLORS["reference"])
    ax.text(
        0.50,
        0.435,
        "sample / pair\nsource / inverse",
        ha="center",
        va="center",
        fontsize=5.0,
        color=COLORS["muted"],
        linespacing=1.0,
    )
    arrow(ax, (0.50, 0.375), (0.50, 0.30), COLORS["line"], 0.9)
    rounded_panel(ax, 0.08, 0.12, 0.84, 0.17, COLORS["soft_green"], COLORS["target"], 0.9, 0.012)
    ax.text(0.50, 0.235, "FROZEN FULL1000", ha="center", va="center", fontsize=5.4, fontweight="bold", color=COLORS["target"])
    ax.text(
        0.50,
        0.180,
        "1,000 targets +\n1,000 references",
        ha="center",
        va="center",
        fontsize=5.0,
        color=COLORS["ink"],
        linespacing=1.0,
    )
    ax.text(
        0.50,
        0.060,
        "source-disjoint split\nfrozen SHA256",
        ha="center",
        va="center",
        fontsize=5.0,
        color=COLORS["muted"],
        linespacing=1.1,
    )

    fig.savefig(
        FIGURE_DIR / "figure2_curation_pipeline.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        FIGURE_DIR / "figure2_curation_pipeline.svg",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        FIGURE_DIR / "figure2_curation_pipeline.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        FIGURE_DIR / "figure2_curation_pipeline.tiff",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def table_benchmark(manifest: dict[str, Any]) -> str:
    rows = []
    for label, key in (("Core150", "core150"), ("Full1000", "full1000")):
        item = manifest["benchmarks"][key]
        rows.append(
            f"{label} & {item['query_count']} & {item['sound_event']} & "
            f"{item['music']} & {item['speech']} & {item['logical_gallery_entries']} \\\\"
        )
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{3.2pt}",
            "\\caption{Frozen Audio-CVR evaluation sets. Full1000 is "
            "automatically curated and model-verified; a partial blinded "
            "single-rater audit is reported separately. Neither split is "
            "presented as a multi-annotator gold set.}",
            "\\label{tab:benchmark}",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "Split & Queries & Sound & Music & Speech & Gallery \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )


def table_results(manifest: dict[str, Any]) -> str:
    results = manifest["results"]["full1000"]
    row_specs = [
        ("E5 base", "V+T", "e5_base_vt"),
        ("E5 base", "V+A+T", "e5_base_vat"),
        ("E5 adapter", "V+T", "e5_adapter_vt"),
        ("E5 adapter", "V+A+T", "e5_adapter_vat"),
        ("ImageBind", "V+T", "imagebind_vt"),
        ("ImageBind", "V+A+T", "imagebind_vat"),
        ("OmniEmbed", "V+T", "omniembed_vt"),
        ("OmniEmbed", "V+A+T", "omniembed_vat"),
    ]
    rows = []
    for model, mode, key in row_specs:
        item = results[key]
        rows.append(
            f"{model} & {mode} & "
            f"{percent_mean_std(item, 'with_reference_r1', 'with_reference_r1_std')} & "
            f"{percent_mean_std(item, 'without_reference_r1', 'without_reference_r1_std')} & "
            f"{percent(item['target_over_reference'])} & "
            f"{'TBD' if item['target_reference_margin'] is None else f'{item['target_reference_margin']:.3f}'} \\\\"
        )
    return "\n".join(
        [
            "\\begin{table*}[t]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{5.0pt}",
            "\\caption{Full1000 exact-reference results. R@1 and "
            "target-over-reference (T$>$R) are percentages; E5 adapter R@1 is "
            "mean $\\pm$ std over five seeds. Masked R@1 changes only the "
            "current query's reference score.}",
            "\\label{tab:main-results}",
            "\\begin{tabular}{llrrrr}",
            "\\toprule",
            "Model & Input & With-ref R@1 & Masked R@1 & T$>$R & Margin \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ]
    )


def table_evidence(manifest: dict[str, Any]) -> str:
    ladder = manifest["supplementary_results"]["reference_perturbation_ladder"]
    row_specs = [
        ("E5 adapter", "V+T", "e5_adapter_vt"),
        ("E5 adapter", "V+A+T", "e5_adapter_vat"),
        ("ImageBind", "V+T", "imagebind_vt"),
        ("ImageBind", "V+A+T", "imagebind_vat"),
        ("OmniEmbed", "V+T", "omniembed_vt"),
        ("OmniEmbed", "V+A+T", "omniembed_vat"),
    ]
    tex_rows = []
    for model, mode, key in row_specs:
        exact, transcoded, temporal, spatial = ladder[key]
        tex_rows.append(
            f"{model} & {mode} & {percent(exact)} & {percent(transcoded)} & "
            f"{percent(temporal)} & {percent(spatial)} \\\\"
        )
    return "\n".join(
        [
            "\\begin{table*}[t]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{6.0pt}",
            "\\caption{Reference-identity perturbation ladder on Full1000. "
            "Entries are reference-induced R@1 drops (masked minus with-reference, "
            "percentage points). Transcoding preserves content; temporal removes "
            "0.5\\,s from each end; spatial center-crops 90\\% and resizes.}",
            "\\label{tab:reference-ladder}",
            "\\begin{tabular}{llrrrr}",
            "\\toprule",
            "Model & Input & Exact & Transcoded & Temporal & Spatial \\\\",
            "\\midrule",
            *tex_rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ]
    )


def benchmark_audit_narrative(manifest: dict[str, Any]) -> str:
    frozen = manifest["frozen_test"]
    datasets = frozen["dataset_distribution"]
    audit = frozen["audit"]
    human = audit["human_audit"]
    return (
        "\\paragraph{Composition and audit scope.}\n"
        f"Full1000 contains {datasets['avatar']} Avatar, "
        f"{datasets['vggsound']} VGGSound, {datasets['ave']} AVE, "
        f"{datasets['worldsense']} WorldSense, and "
        f"{datasets['vgg_monoaudio']} VGG-MonoAudio queries. "
        "The final audit found no duplicate sample, source, or pair, no "
        "train/validation leakage, and no missing media. "
        "After freezing, all "
        f"{audit['repeat_review_requested']} requested repeat reviews were "
        f"completed, with {percent(audit['exact_decision_agreement'])}\\% exact "
        f"decision agreement and {percent(audit['field_level_agreement'])}\\% "
        "field-level agreement. This post-freeze audit remained observational: "
        "it neither changed benchmark membership nor informed selection. We "
        "therefore describe Full1000 as automatically curated and "
        "model-verified, not as a multi-annotator gold set. "
        f"A blinded single-rater audit was stopped after {human['unique_sample_count']} "
        f"unique queries ({human['completed_display_count']} displayed items), "
        f"of which {human['valid_count']} were judged valid "
        f"({percent(human['valid_rate'])}\\%). Because only "
        f"{percent(human['completed_display_count'] / human['planned_display_count'])}\\% "
        "of the planned display sequence was completed, this is explicitly a "
        "partial quality audit rather than a prevalence estimate."
    )


def model_config_narrative(manifest: dict[str, Any]) -> str:
    config = manifest["server_reproducibility"]["adapter_training"]
    return (
        "\\paragraph{Models.}\n"
        "We evaluate frozen E5-Omni, E5-Omni with a lightweight low-rank "
        "residual adapter, zero-shot ImageBind, and OmniEmbed-MultiVENT, a "
        "Qwen2.5-Omni checkpoint trained for multimodal video retrieval. "
        f"The adapter has rank {config['rank']} and "
        f"{config['trainable_parameter_count']:,} trainable parameters over "
        f"{config['embedding_dim']}-dimensional frozen embeddings. "
        f"It is trained on {config['forward_pairs']} forward pairs plus "
        f"{config['accepted_inverse_pairs']} independently verified inverse "
        f"pairs ({config['directional_training_instances']} directional "
        f"instances), using {config['steps']} steps, a batch size of "
        f"{config['batch_size']}, and a learning rate of "
        f"${latex_scientific(config['learning_rate'])}$. "
        f"Configuration selection uses only {config['validation_queries']} "
        "validation queries under a one-standard-error rule. "
        "ImageBind uses fixed equal-weight arithmetic over normalized visual, "
        "audio, and text embeddings and receives no benchmark-specific training. "
        "OmniEmbed receives the same fixed retrieval instruction in every "
        "condition and is evaluated zero-shot, without Full1000 tuning. An "
        "Audio-as-Text VLM2Vec reproduction is retained as a supplementary "
        "control rather than treated as the official AudioVLM2Vec system."
    )


def results_narrative(manifest: dict[str, Any]) -> str:
    full = manifest["results"]["full1000"]
    vt = full["e5_adapter_vt"]
    vat = full["e5_adapter_vat"]
    base_vt = full["e5_base_vt"]
    base_vat = full["e5_base_vat"]
    ib_vt = full["imagebind_vt"]
    ib_vat = full["imagebind_vat"]
    oe_vt = full["omniembed_vt"]
    oe_vat = full["omniembed_vat"]
    omni_oe_vt = manifest["results"]["omnicvr1000_omniembed"]["omniembed_vt"]
    omni_oe_vat = manifest["results"]["omnicvr1000_omniembed"]["omniembed_vat"]
    human = manifest["frozen_test"]["audit"]["human_audit"]
    human_valid = manifest["supplementary_results"]["human_valid_subset"]
    ladder = manifest["supplementary_results"]["reference_perturbation_ladder"]
    modes = manifest["supplementary_results"]["e5_adapter_seven_modes"]
    reference_controls = manifest["supplementary_results"]["reference_dominance_controls"]
    full_control = reference_controls["full1000"]
    historical_control = reference_controls["historical_mixed_gallery"]
    return "\n\n".join(
        [
            (
                "\\paragraph{RQ1: Is top-rank failure concentrated on the own reference?}\n"
                f"Masking raises E5-adapter R@1 from "
                f"{percent(vt['with_reference_r1'], 2)}\\% to "
                f"{percent(vt['without_reference_r1'], 2)}\\% for V+T and "
                f"from {percent(vat['with_reference_r1'], 2)}\\% to "
                f"{percent(vat['without_reference_r1'], 2)}\\% for V+A+T. "
                f"Across seeds, in {vat['top1_error_own_reference_count']:,} of "
                f"{vat['top1_error_count_across_seeds']:,} top-1 errors "
                f"({percent(vat['top1_error_own_reference_rate'], 2)}\\%), "
                "the query's own reference ranks first. Frozen E5 exhibits the "
                "same pattern: its V+T and V+A+T R@1 values are "
                f"{percent(base_vt['with_reference_r1'], 1)}\\% and "
                f"{percent(base_vat['with_reference_r1'], 1)}\\% with the "
                f"reference, but both reach "
                f"{percent(base_vat['without_reference_r1'], 1)}\\% after "
                "masking. ImageBind exhibits drops of "
                f"{percent(ib_vt['reference_masking_gain_r1'], 1)} and "
                f"{percent(ib_vat['reference_masking_gain_r1'], 1)} points; "
                "retrieval-trained OmniEmbed exhibits drops of "
                f"{percent(oe_vt['reference_masking_gain_r1'], 1)} and "
                f"{percent(oe_vat['reference_masking_gain_r1'], 1)} points. "
                "The adapted V+A+T target beats each designated visual, audio, "
                "and ASR hard negative in 100\\% of comparisons, but beats its "
                f"own reference in only {percent(full_control['adapter_target_beats_own_reference_mean'], 1)}\\%. "
                "A historical mixed-gallery control places the reference above "
                f"{percent(historical_control['adapter_reference_beats_local_pairwise'], 1)}\\% "
                "of same-source local candidates and "
                f"{percent(historical_control['adapter_reference_beats_random_pairwise'], 2)}\\% "
                "of random candidates. \\textit{Takeaway:} the models generally "
                "enter the correct contextual neighborhood but fail at the final "
                "target--reference direction boundary."
            ),
            (
                "\\paragraph{RQ2: Is the effect only exact self-matching?}\n"
                "We replace only the own-reference item with a deterministic "
                "transcoded, temporally trimmed, or spatially cropped version. "
                f"All {ladder['variant_count']:,} variants were generated without "
                "missing media, and all eight audited temporal/spatial variants "
                "preserved the pre-edit semantics. For adapted E5, the V+A+T "
                "reference-induced R@1 drop decreases from "
                f"{percent(ladder['e5_adapter_vat'][0])} points under exact "
                f"inclusion to {percent(ladder['e5_adapter_vat'][2])} after "
                f"temporal trimming and {percent(ladder['e5_adapter_vat'][3])} "
                "after spatial cropping. Residual drops remain "
                f"{percent(ladder['imagebind_vat'][2])}/"
                f"{percent(ladder['imagebind_vat'][3])} points for ImageBind and "
                f"{percent(ladder['omniembed_vat'][2])}/"
                f"{percent(ladder['omniembed_vat'][3])} for OmniEmbed. Exact "
                "identity is therefore a major component. \\textit{Takeaway:} "
                "identity-reducing transformations leave substantial pre-edit "
                "anchoring, but this residual is not treated as standalone "
                "evidence of semantic edit reasoning."
            ),
            (
                "\\paragraph{RQ3: When does audio help?}\n"
                "Under exact inclusion, V+A+T versus V+T raises adapted-E5 R@1 "
                f"from {percent(vt['with_reference_r1'], 2)} $\\pm$ "
                f"{percent(vt['with_reference_r1_std'], 2)}\\% to "
                f"{percent(vat['with_reference_r1'], 2)} $\\pm$ "
                f"{percent(vat['with_reference_r1_std'], 2)}\\% "
                f"(+{percent(vat['audio_gain_r1'], 2)} points; 95\\% CI "
                f"[{percent(vat['audio_gain_r1_ci95'][0], 2)}, "
                f"{percent(vat['audio_gain_r1_ci95'][1], 2)}]; Holm "
                f"$p={vat['audio_gain_r1_holm_p']:.4f}$). R@5 and R@10 do not "
                f"improve ($\\Delta={percent(vat['audio_gain_r5'], 2)}$ and "
                f"{percent(vat['audio_gain_r10'], 2)} points; Holm "
                f"$p={vat['audio_gain_r5_holm_p']:.3f}$ and "
                f"$p={vat['audio_gain_r10_holm_p']:.1f}$), localizing the gain "
                "to the top ordering boundary. The gain reverses "
                "after temporal and spatial perturbation "
                "($-11.74$ and $-10.62$ points), where the reference audio "
                "remains highly source-identifying. Exact-condition audio also "
                f"hurts ImageBind by {percent(abs(ib_vat['audio_gain_r1']), 1)} "
                f"points and OmniEmbed by {percent(abs(oe_vat['audio_gain_r1']), 1)} "
                "point. Audio-only and A+T reach "
                f"{percent(modes['A_only']['r1_mean'], 2)}\\% and "
                f"{percent(modes['A_T']['r1_mean'], 2)}\\% R@1, neither below "
                f"V+A+T at {percent(modes['V_A_T']['r1_mean'], 2)}\\%. "
                "\\textit{Takeaway:} audio can reinforce either the requested "
                "edit or the pre-edit source; these results do not establish "
                "complete joint audiovisual semantic reasoning."
            ),
            (
                "\\paragraph{RQ4: Does the diagnostic transfer?}\n"
                "Frozen and adapted E5, independently structured ImageBind, and "
                "retrieval-trained OmniEmbed all exhibit own-reference dominance "
                "on Full1000. On 1,000 OmniCVR audio-centered queries, "
                "OmniEmbed V+T rises from "
                f"{percent(omni_oe_vt['with_reference_r1'], 1)}\\% to "
                f"{percent(omni_oe_vt['without_reference_r1'], 1)}\\% after "
                "source masking; V+A+T rises from "
                f"{percent(omni_oe_vat['with_reference_r1'], 1)}\\% to "
                f"{percent(omni_oe_vat['without_reference_r1'], 1)}\\%. "
                f"A partial blinded audit judged {human['valid_count']}/"
                f"{human['unique_sample_count']} completed unique queries valid "
                f"({percent(human['valid_rate'])}\\%); on those "
                f"{human_valid['query_count']} human-valid cases, exact masking "
                "raises V+A+T adapter R@1 from "
                f"{percent(human_valid['e5_adapter_vat']['with_reference_r1'])}\\% "
                f"to {percent(human_valid['e5_adapter_vat']['without_reference_r1'])}\\%. "
                "The completed Core150 portion was "
                f"{human['core150_valid_count']}/{human['core150_count']} "
                f"({percent(human['core150_valid_rate'])}\\%; Wilson 95\\% CI "
                f"[{percent(human['core150_wilson_95_ci'][0])}, "
                f"{percent(human['core150_wilson_95_ci'][1])}]); the audit is "
                "partial and does not estimate Full1000-wide validity. "
                "\\textit{Takeaway:} source anchoring appears across retrievers, "
                "on an external benchmark, and on the audited human-valid subset."
            ),
        ]
    )


def ablation_narrative(manifest: dict[str, Any]) -> str:
    bidir = manifest["supplementary_results"]["verified_bidirectional_ablation"]
    return (
        "\\paragraph{Supplementary training control.}\n"
        "Verified inverse augmentation is not a hidden source of the adapted-E5 "
        "gain: on Core150 it "
        f"reduces R@1 from {percent(bidir['forward_only_r1_mean'])}\\% to "
        f"{percent(bidir['forward_plus_inverse_r1_mean'])}\\% "
        f"($\\Delta={percent(bidir['r1_difference'])}$ points, Holm "
        f"$p={bidir['r1_holm_p']:.3f}$), despite improving the score margin. "
        "We retain this negative result and use the 89-instance model only as "
        "a fixed baseline."
    )


def limitations_audit_narrative(manifest: dict[str, Any]) -> str:
    frozen = manifest["frozen_test"]
    config = manifest["server_reproducibility"]["adapter_training"]
    audit = frozen["audit"]
    human = audit["human_audit"]
    return (
        "\\textit{Annotation scope.} Full1000 is automatically curated and "
        "model-verified rather than a multi-annotator gold set. The deterministic "
        "repeat review was "
        f"completed after freezing ({audit['repeat_review_completed']}/"
        f"{audit['repeat_review_requested']}; "
        f"{percent(audit['exact_decision_agreement'])}\\% exact-decision and "
        f"{percent(audit['field_level_agreement'])}\\% field-level agreement), "
        "but measures same-model stability rather than human consensus. The "
        "single-rater blinded audit is partial "
        f"({human['unique_sample_count']} unique queries; "
        f"{percent(human['valid_rate'])}\\% judged valid) and has no completed "
        "paired repeat for estimating intra-rater agreement. "
        "\\textit{Benchmark scope.} The benchmark excludes speech and remains "
        "source-imbalanced, with 51\\% of queries derived from Avatar; nine "
        "VGG-MonoAudio queries form a separately traceable controlled "
        "supplement. Review profiles are heterogeneous but recorded, strict "
        "same-source local negatives are unavailable at benchmark scale, and "
        "the task is a controlled diagnostic rather than a replacement for "
        "general AV-CVR. \\textit{Interpretation scope.} The E5 adapter is a "
        "few-shot baseline trained on only "
        f"{config['directional_training_instances']} directional instances. "
        "ImageBind uses fixed equal-weight fusion, the Audio-as-Text VLM2Vec "
        "control is not the official AudioVLM2Vec system, and evaluation omits "
        "recent specialized retrievers such as OmniRet. One unrelated OmniCVR "
        "gallery video is excluded identically across OmniEmbed modes, without "
        "affecting a target or own reference. Identity perturbations do not "
        "fully separate identity from direction, so we claim identity-sensitive "
        "pre-edit source anchoring rather than a clean measure of semantic edit "
        "reasoning."
    )


def final_values_missing(manifest: dict[str, Any]) -> list[str]:
    missing = []
    for result_name, values in manifest["results"]["full1000"].items():
        for field_name, value in values.items():
            if value is None:
                missing.append(f"results.full1000.{result_name}.{field_name}")
    return missing


def abstract_results(manifest: dict[str, Any]) -> str:
    missing = final_values_missing(manifest)
    if missing:
        return (
            "Full1000 E5 and ImageBind results are being finalized; their "
            "frozen, paired estimates will replace this sentence before submission."
        )
    e5_vt = manifest["results"]["full1000"]["e5_adapter_vt"]
    e5_vat = manifest["results"]["full1000"]["e5_adapter_vat"]
    imagebind_vat = manifest["results"]["full1000"]["imagebind_vat"]
    omniembed_vat = manifest["results"]["full1000"]["omniembed_vat"]
    omniembed_omni = manifest["results"]["omnicvr1000_omniembed"]["omniembed_vat"]
    human = manifest["frozen_test"]["audit"]["human_audit"]
    human_valid = manifest["supplementary_results"]["human_valid_subset"]
    ladder = manifest["supplementary_results"]["reference_perturbation_ladder"]
    residuals = [
        ladder[key][index]
        for key in (
            "e5_adapter_vt",
            "e5_adapter_vat",
            "imagebind_vt",
            "imagebind_vat",
            "omniembed_vt",
            "omniembed_vat",
        )
        for index in (2, 3)
    ]
    return (
        f"On model-verified Full1000 (1,000 queries; 2,000 target/reference "
        "items), exact masking raises V+A+T R@1 by "
        f"{percent(e5_vat['reference_masking_gain_r1'])}, "
        f"{percent(imagebind_vat['reference_masking_gain_r1'])}, and "
        f"{percent(omniembed_vat['reference_masking_gain_r1'])} points for "
        "adapted E5-Omni, ImageBind, and retrieval-trained OmniEmbed. Temporal "
        "and spatial perturbations reduce exact identity but leave "
        f"{percent(min(residuals))}--{percent(max(residuals))}-point reference "
        "drops. Enabling audio for the E5 adapter raises mean R@1 across five "
        "seeds from "
        f"{percent(e5_vt['with_reference_r1'])}\\% to "
        f"{percent(e5_vat['with_reference_r1'])}\\% "
        f"($\\Delta={percent(e5_vat['audio_gain_r1'])}$ points; 95\\% CI "
        f"[{percent(e5_vat['audio_gain_r1_ci95'][0])}, "
        f"{percent(e5_vat['audio_gain_r1_ci95'][1])}]; Holm-adjusted "
        f"$p={e5_vat['audio_gain_r1_holm_p']:.4f}$), while audio reduces "
        f"ImageBind and OmniEmbed R@1 by "
        f"{percent(abs(imagebind_vat['audio_gain_r1']))} and "
        f"{percent(abs(omniembed_vat['audio_gain_r1']))} points. OmniEmbed "
        "also shows a "
        f"{percent(omniembed_omni['reference_masking_gain_r1'])}-point drop on "
        f"OmniCVR. A partial blinded audit judged {human['valid_count']}/"
        f"{human['unique_sample_count']} queries valid; on those "
        f"{human_valid['query_count']} cases, adapted V+A+T R@1 rises from "
        f"{percent(human_valid['e5_adapter_vat']['with_reference_r1'])}\\% to "
        f"{percent(human_valid['e5_adapter_vat']['without_reference_r1'])}\\% "
        "after exact masking."
    )


def build_manuscript(manifest: dict[str, Any]) -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    replacements = {
        "@@ABSTRACT_RESULTS@@": abstract_results(manifest),
        "@@TABLE_BENCHMARK@@": table_benchmark(manifest),
        "@@BENCHMARK_AUDIT@@": benchmark_audit_narrative(manifest),
        "@@MODEL_CONFIG@@": model_config_narrative(manifest),
        "@@TABLE_RESULTS@@": table_results(manifest),
        "@@RESULTS_NARRATIVE@@": results_narrative(manifest),
        "@@ABLATION_NARRATIVE@@": ablation_narrative(manifest),
        "@@TABLE_EVIDENCE@@": table_evidence(manifest),
        "@@LIMITATIONS_AUDIT@@": limitations_audit_narrative(manifest),
    }
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    unresolved = [token for token in replacements if token in rendered]
    if unresolved:
        raise RuntimeError(f"Unresolved manuscript tokens: {unresolved}")
    MAIN_PATH.write_text(rendered, encoding="utf-8", newline="\n")
    abstract_match = re.search(
        r"\\begin\{abstract\}\s*(.*?)\s*\\end\{abstract\}",
        rendered,
        flags=re.DOTALL,
    )
    if abstract_match is None:
        raise RuntimeError("Generated manuscript does not contain an abstract.")
    ABSTRACT_PATH.write_text(
        "\\begin{abstract}\n"
        + abstract_match.group(1).strip()
        + "\n\\end{abstract}\n",
        encoding="utf-8",
        newline="\n",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_build_summary(manifest: dict[str, Any]) -> None:
    outputs = [
        MAIN_PATH,
        ABSTRACT_PATH,
        FIGURE_DIR / "figure1_reference_confusion.pdf",
        FIGURE_DIR / "figure1_reference_confusion.svg",
        FIGURE_DIR / "figure1_reference_confusion.png",
        FIGURE_DIR / "figure1_reference_confusion.tiff",
        FIGURE_DIR / "figure2_curation_pipeline.pdf",
        FIGURE_DIR / "figure2_curation_pipeline.svg",
        FIGURE_DIR / "figure2_curation_pipeline.png",
        FIGURE_DIR / "figure2_curation_pipeline.tiff",
    ]
    summary = {
        "schema_version": 1,
        "frozen_test_sha256": manifest["frozen_test"]["sha256"],
        "missing_final_fields": final_values_missing(manifest),
        "outputs": {
            str(path.relative_to(ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in outputs
        },
    }
    (GENERATED_DIR / "asset_build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (GENERATED_DIR / "results_snapshot.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Audio-CVR figures, tables, and the single-source AAAI manuscript."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any required Full1000 result remains unset.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Keep the existing Draw.io figure exports instead of rebuilding the legacy Matplotlib figures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    missing = final_values_missing(manifest)
    if args.strict and missing:
        joined = "\n".join(f"- {field}" for field in missing)
        raise SystemExit(f"Strict build blocked by missing final fields:\n{joined}")
    if not args.skip_figures:
        draw_figure1()
        draw_figure2()
    build_manuscript(manifest)
    write_build_summary(manifest)
    print(
        json.dumps(
            {
                "state": "COMPLETE_WITH_PLACEHOLDERS" if missing else "COMPLETE",
                "missing_final_field_count": len(missing),
                "main_tex": str(MAIN_PATH),
                "figure_dir": str(FIGURE_DIR),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
