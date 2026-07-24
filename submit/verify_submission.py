from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAIN_PDF = ROOT / "01_main_paper" / "Audio-CVR_AAAI27_anonymous.pdf"
CHECKLIST_PDF = (
    ROOT / "03_reproducibility_checklist" / "ReproducibilityChecklist.pdf"
)
CODE_ZIP = (
    ROOT
    / "06_code_and_data_supplement"
    / "Audio-CVR_code_and_data_anonymous.zip"
)
EXPECTED_TITLE = (
    "Audio-CVR: Automatic Curation and Reference-Aware Evaluation for "
    "Directional Audio Composed Video Retrieval"
)
EXPECTED_TEST_SHA = (
    "70bd998c33bd4c2168ac18afb26ec6fbe928b234c61241f53412be387d52ec9e"
)
FORBIDDEN = re.compile(
    r"(wangqihao|10\.1\.4\.86|/data02/usr/|[A-Za-z]:\\Users\\)",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout


def resolve_tool(explicit: str | None, name: str) -> str | None:
    if explicit:
        return explicit
    return shutil.which(name)


def verify_checksums() -> dict[str, str]:
    checksum_path = ROOT / "SHA256SUMS.txt"
    expected: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value, relative = line.split(maxsplit=1)
        expected[relative.strip()] = value.lower()
    for relative, value in expected.items():
        path = ROOT / relative
        assert path.is_file(), relative
        assert sha256(path) == value, relative
    assert MAIN_PDF.relative_to(ROOT).as_posix() in expected
    assert CHECKLIST_PDF.relative_to(ROOT).as_posix() in expected
    assert CODE_ZIP.relative_to(ROOT).as_posix() in expected
    return expected


def verify_openreview_text() -> dict[str, int]:
    text_root = ROOT / "02_openreview_text"
    title = (text_root / "title.txt").read_text(encoding="utf-8").strip()
    tldr = (text_root / "tldr.txt").read_text(encoding="utf-8").strip()
    abstract = (text_root / "abstract.txt").read_text(encoding="utf-8").strip()
    assert title == EXPECTED_TITLE
    assert 1 <= len(tldr) <= 250
    assert 500 <= len(abstract) <= 5000
    assert EXPECTED_TEST_SHA not in abstract
    assert "Full1000" in abstract and "34/50" in abstract
    assert "17.1%" in abstract and "93.5%" in abstract
    assert "human-validated" not in abstract.lower()
    assert "http://" not in abstract and "https://" not in abstract
    assert not FORBIDDEN.search(title + "\n" + tldr + "\n" + abstract)
    return {
        "title_characters": len(title),
        "tldr_characters": len(tldr),
        "abstract_characters": len(abstract),
    }


def verify_pdf(
    path: Path,
    *,
    expected_pages: int,
    pdfinfo: str | None,
    pdffonts: str | None,
    pdftotext: str | None,
) -> dict[str, object]:
    assert path.is_file() and path.read_bytes().startswith(b"%PDF")
    result: dict[str, object] = {"bytes": path.stat().st_size}
    if pdfinfo:
        info = run([pdfinfo, str(path)])
        pages = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
        size = re.search(r"^Page size:\s+(.+)$", info, re.MULTILINE)
        encrypted = re.search(r"^Encrypted:\s+(.+)$", info, re.MULTILINE)
        assert pages and int(pages.group(1)) == expected_pages
        assert size and "612 x 792" in size.group(1) and "letter" in size.group(1).lower()
        assert encrypted and encrypted.group(1).strip().lower() == "no"
        assert not re.search(r"^(Author|Title|Subject):\s+\S", info, re.MULTILINE)
        result["pages"] = expected_pages
        result["letter"] = True
        result["encrypted"] = False
    if pdffonts:
        fonts = run([pdffonts, str(path)])
        lines = [
            line.split()
            for line in fonts.splitlines()[2:]
            if line.strip() and not line.startswith("-")
        ]
        assert lines
        assert all("Type 3" not in " ".join(line) for line in lines)
        assert all("yes" in line for line in lines)
        result["font_count"] = len(lines)
        result["type3_font_count"] = 0
    if pdftotext:
        with tempfile.TemporaryDirectory(prefix="audiocvr_pdf_text_") as temp_dir:
            text_path = Path(temp_dir) / "paper.txt"
            run([pdftotext, str(path), str(text_path)])
            text = text_path.read_text(encoding="utf-8", errors="replace")
        assert not FORBIDDEN.search(text)
        assert "github.com" not in text.lower()
        result["text_characters"] = len(text)
    return result


def verify_code_zip(*, full_code_tests: bool) -> dict[str, object]:
    assert CODE_ZIP.stat().st_size <= 50 * 1024 * 1024
    with zipfile.ZipFile(CODE_ZIP) as archive:
        names = archive.namelist()
        assert names
        assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in names)
        assert all("__pycache__" not in name and not name.endswith(".pyc") for name in names)
        text_suffixes = {".json", ".jsonl", ".md", ".py", ".sh", ".txt", ".yaml", ".yml"}
        violations: list[str] = []
        for info in archive.infolist():
            if Path(info.filename).suffix.lower() not in text_suffixes:
                continue
            if info.filename == "verify_package.py":
                # This file necessarily contains the forbidden-pattern regex
                # used by the inner anonymity audit.
                continue
            text = archive.read(info).decode("utf-8", errors="replace")
            if FORBIDDEN.search(text):
                violations.append(info.filename)
        assert not violations, violations
        with tempfile.TemporaryDirectory(prefix="audiocvr_submission_cold_") as temp_dir:
            target = Path(temp_dir)
            archive.extractall(target)
            run([sys.executable, "verify_package.py", "--skip-imports"], cwd=target)
            run(
                [
                    sys.executable,
                    "-m",
                    "compileall",
                    "-q",
                    "app",
                    "tests",
                    "tools",
                ],
                cwd=target,
            )
            if full_code_tests:
                run(
                    [
                        sys.executable,
                        "-m",
                        "unittest",
                        "discover",
                        "-s",
                        "tests",
                        "-v",
                    ],
                    cwd=target,
                )
    return {
        "bytes": CODE_ZIP.stat().st_size,
        "entry_count": len(names),
        "full_code_tests": full_code_tests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Audio-CVR upload bundle")
    parser.add_argument("--pdfinfo")
    parser.add_argument("--pdffonts")
    parser.add_argument("--pdftotext")
    parser.add_argument("--full-code-tests", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "QA_REPORT.json")
    args = parser.parse_args()

    pdfinfo = resolve_tool(args.pdfinfo, "pdfinfo")
    pdffonts = resolve_tool(args.pdffonts, "pdffonts")
    pdftotext = resolve_tool(args.pdftotext, "pdftotext")
    report = {
        "state": "COMPLETE",
        "test1000_sha256": EXPECTED_TEST_SHA,
        "checksums": verify_checksums(),
        "openreview_text": verify_openreview_text(),
        "main_pdf": verify_pdf(
            MAIN_PDF,
            expected_pages=7,
            pdfinfo=pdfinfo,
            pdffonts=pdffonts,
            pdftotext=pdftotext,
        ),
        "checklist_pdf": verify_pdf(
            CHECKLIST_PDF,
            expected_pages=2,
            pdfinfo=pdfinfo,
            pdffonts=pdffonts,
            pdftotext=pdftotext,
        ),
        "code_zip": verify_code_zip(full_code_tests=args.full_code_tests),
        "tooling": {
            "pdfinfo": Path(pdfinfo).name if pdfinfo else None,
            "pdffonts": Path(pdffonts).name if pdffonts else None,
            "pdftotext": Path(pdftotext).name if pdftotext else None,
        },
    }
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
