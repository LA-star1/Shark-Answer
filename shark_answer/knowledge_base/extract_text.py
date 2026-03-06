#!/usr/bin/env python3
"""Step 1: Extract text from all PDFs in the knowledge base.

Run once (or re-run to pick up newly added files — already-extracted PDFs are skipped).

Usage:
    python -m shark_answer.knowledge_base.extract_text
    python -m shark_answer.knowledge_base.extract_text --dir ~/Desktop/BTC/AI/shark_answer_kb
    python -m shark_answer.knowledge_base.extract_text --force   # re-extract all

Progress:
    Extracting 450/2600 — physics_9702/mark_schemes/2024_june_ms_21.pdf

Summary:
    Total extracted: 2540
    Total skipped:   55  (already had .txt companion)
    Total failed:    5   (likely scanned images — flag for manual OCR)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DEFAULT_KB_DIR = Path.home() / "Desktop/BTC/AI/shark_answer_kb"


def _extract_pdf(pdf_path: Path) -> str | None:
    """Extract text from a single PDF using PyMuPDF.

    Returns the full text string on success, None if the PDF is scanned/image-only
    or if extraction fails.
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        pages: list[str] = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text.strip():
                pages.append(f"[Page {page_num + 1}]\n{text.rstrip()}")
        doc.close()

        if not pages:
            return None  # All pages are blank / image-only
        return "\n\n".join(pages)

    except Exception as exc:  # noqa: BLE001
        return None


def main(kb_dir: Path, force: bool = False) -> int:
    """Extract text from all PDFs under kb_dir.

    Returns exit code (0 = OK, 1 = some failures).
    """
    all_pdfs = sorted(kb_dir.rglob("*.pdf"))
    total = len(all_pdfs)

    if total == 0:
        print(f"No PDFs found in {kb_dir}", file=sys.stderr)
        return 1

    extracted = 0
    skipped = 0
    failed: list[Path] = []

    for i, pdf_path in enumerate(all_pdfs, start=1):
        txt_path = pdf_path.with_suffix(".txt")
        rel = pdf_path.relative_to(kb_dir)

        # Progress line (overwrite in place)
        print(f"\rExtracting {i}/{total} — {rel}    ", end="", flush=True)

        # Skip if already extracted and not forcing
        if txt_path.exists() and not force:
            skipped += 1
            continue

        text = _extract_pdf(pdf_path)
        if text is None:
            failed.append(pdf_path)
            # Print failure on its own line so it doesn't get overwritten
            print(f"\n  [FAILED/SCANNED] {rel}", file=sys.stderr, flush=True)
        else:
            txt_path.write_text(text, encoding="utf-8")
            extracted += 1

    print()  # Final newline after progress

    # Summary
    print("\n=== Summary ===")
    print(f"  Total PDFs:   {total}")
    print(f"  Extracted:    {extracted}")
    print(f"  Skipped:      {skipped}  (already had .txt companion)")
    print(f"  Failed:       {len(failed)}  (likely scanned images — flag for manual OCR)")

    if failed:
        print("\nFailed PDFs (need manual review or OCR tool):")
        for p in failed:
            print(f"  {p.relative_to(kb_dir)}")

    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract text from all PDFs in the CIE knowledge base directory."
    )
    parser.add_argument(
        "--dir",
        default=str(_DEFAULT_KB_DIR),
        help="Path to the knowledge base root directory (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if a .txt companion already exists",
    )
    args = parser.parse_args()

    kb_dir = Path(args.dir).expanduser()
    if not kb_dir.exists():
        print(f"ERROR: Directory not found: {kb_dir}", file=sys.stderr)
        sys.exit(1)

    sys.exit(main(kb_dir, force=args.force))
