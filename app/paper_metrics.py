from __future__ import annotations

from typing import Any


MSRVTT_REFERENCE_RESULTS: dict[str, dict[str, Any]] = {
    "avigate_paper": {
        "name": "AVIGATE (Paper)",
        "dataset": "MSR-VTT 9k split",
        "source": "AVIGATE CVPR 2025 Table 1",
        "t2v": {"R@1": 50.2, "R@5": 74.3, "R@10": 83.2},
        "v2t": {"R@1": 49.7, "R@5": 75.3, "R@10": 83.7},
        "rsum": 416.4,
    },
    "avigate_reproduction": {
        "name": "AVIGATE (Local Reproduction)",
        "dataset": "MSR-VTT 9k split",
        "source": "Local reproduction report",
        "t2v": {"R@1": 48.1, "R@5": 74.5, "R@10": 84.3},
        "v2t": {"R@1": 46.8, "R@5": 75.4, "R@10": 84.1},
        "rsum": 413.2,
    },
}


def get_reference_result(name: str) -> dict[str, Any]:
    try:
        return MSRVTT_REFERENCE_RESULTS[name]
    except KeyError as exc:
        available = ", ".join(sorted(MSRVTT_REFERENCE_RESULTS))
        raise KeyError(f"unknown reference {name!r}; available: {available}") from exc
