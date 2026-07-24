#!/usr/bin/env python3
"""Convert the latest Draw.io PNG previews into matching 600-dpi TIFFs."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated"
FIGURES = (
    "figure1_reference_confusion",
    "figure2_curation_pipeline",
)


def main() -> int:
    for basename in FIGURES:
        png_path = GENERATED / f"{basename}.png"
        tiff_path = GENERATED / f"{basename}.tiff"
        with Image.open(png_path) as image:
            image.convert("RGB").save(
                tiff_path,
                format="TIFF",
                compression="tiff_lzw",
                dpi=(600, 600),
            )
        print(f"{tiff_path.name}: {tiff_path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
