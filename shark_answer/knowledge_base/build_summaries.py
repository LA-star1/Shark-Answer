#!/usr/bin/env python3
"""Step 4 (optional, one-time): Build subject-level examiner report summaries.

Sends all examiner reports for each subject to Claude Haiku and produces a
concise, actionable summary (~1000 words) saved as:
    {kb_dir}/{subject_key}/subject_summary.txt

Run AFTER extract_text.py has completed.

Cost estimate: ~$0.50 total for all 7 subjects (~70 examiner reports).

Usage:
    python -m shark_answer.knowledge_base.build_summaries
    python -m shark_answer.knowledge_base.build_summaries --subject economics_9708
    python -m shark_answer.knowledge_base.build_summaries --force   # regenerate existing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Cap each ER at this many chars before sending to Haiku.
# 15 ERs × 8 000 chars ≈ 30 000 tokens — well under the 200 k context limit
# and the 50 k tokens/minute rate limit.
_MAX_ER_CHARS_SUMMARY: int = 8_000

_DEFAULT_KB_DIR = Path.home() / "Desktop/BTC/AI/shark_answer_kb"

_SUBJECT_DISPLAY: dict[str, str] = {
    "physics_9702":      "Physics (9702)",
    "chemistry_9701":    "Chemistry (9701)",
    "math_9709":         "Mathematics (9709)",
    "further_math_9231": "Further Mathematics (9231)",
    "economics_9708":    "Economics (9708)",
    "cs_9618":           "Computer Science (9618)",
    "chinese_8238":      "Chinese (8238)",
}

_SUMMARY_PROMPT = """\
You are analyzing ALL available CIE A-Level {subject_name} examiner reports.

Below are {count} examiner report(s) covering years {years}.

=== EXAMINER REPORTS ===
{er_texts}
=== END OF REPORTS ===

Produce a concise subject summary (max 1000 words) structured as follows:

## 1. Mark-Earning Patterns
Bullet-point list of specific language, structure, and content that examiners \
consistently reward across all years. Include exact wording where possible.

## 2. Common Student Mistakes
Bullet-point list of errors, omissions, and misconceptions that appear \
repeatedly. These are things answers must NEVER do.

## 3. Evolution in Marking Standards
How has marking changed over the years covered? Any topic area gaining or \
losing emphasis? Any shift in strictness?

## 4. Examiner Preferences
Specific phrases examiners explicitly mention rewarding. Formatting rules. \
Whether working/units/sig figs must be shown for specific question types.

## 5. Paper-Specific Notes
Any patterns or expectations specific to individual paper components \
(Paper 1, Paper 2, Paper 3, etc.) if mentioned in the reports.

Be concrete and actionable. A student writing an exam answer right now \
should be able to read this and immediately apply it.\
"""


def _load_er_texts(
    subject_key: str,
    manifest_data: dict,
    kb_dir: Path,
) -> tuple[list[str], list[str]]:
    """Load all examiner report .txt files for a subject.

    Returns (texts_list, years_list).
    """
    er_entries = sorted(
        manifest_data.get("examiner_reports", []),
        key=lambda e: e.get("year", 0),
    )

    texts: list[str] = []
    years: list[str] = []

    for entry in er_entries:
        year    = entry.get("year", "?")
        session = (entry.get("session") or "unknown").title()
        fname   = entry.get("renamed") or entry.get("original", "")
        txt_path = (
            kb_dir / subject_key / "examiner_reports" / Path(fname).with_suffix(".txt")
        )
        if txt_path.exists():
            text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                # Truncate very large ERs so the combined prompt stays under limits
                if len(text) > _MAX_ER_CHARS_SUMMARY:
                    text = text[:_MAX_ER_CHARS_SUMMARY] + "\n[...truncated]"
                texts.append(f"[{year} {session} Examiner Report]\n{text}")
                years.append(str(year))
        else:
            print(
                f"    WARNING: No .txt for {fname} — run extract_text.py first",
                file=sys.stderr,
            )

    return texts, years


def build_summary(
    subject_key: str,
    manifest_data: dict,
    kb_dir: Path,
    anthropic_client,
    force: bool = False,
) -> bool:
    """Build and save the subject summary for one subject.

    Returns True on success, False if skipped or failed.
    """
    subject_name = _SUBJECT_DISPLAY.get(subject_key, subject_key)
    summary_path = kb_dir / subject_key / "subject_summary.txt"

    if summary_path.exists() and not force:
        print(f"  [{subject_key}] Already exists — skipping (use --force to regenerate)")
        return True

    er_texts, years = _load_er_texts(subject_key, manifest_data, kb_dir)

    if not er_texts:
        print(f"  [{subject_key}] No extracted examiner reports found — skipping")
        return False

    prompt = _SUMMARY_PROMPT.format(
        subject_name=subject_name,
        count=len(er_texts),
        years=", ".join(years),
        er_texts="\n\n".join(er_texts),
    )

    est_tokens = len("\n\n".join(er_texts)) // 4
    print(
        f"  [{subject_key}] Sending {len(er_texts)} ER(s) to Claude Haiku "
        f"(years: {', '.join(years)}, ~{est_tokens:,} tokens)...",
        flush=True,
    )

    # Retry up to 3 times on rate-limit (429) errors, waiting 70 s each time.
    response = None
    for attempt in range(3):
        try:
            response = anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            break  # success
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            if "529" in err or ("rate_limit" in err and attempt < 2):
                wait = 70 * (attempt + 1)
                print(f"  [{subject_key}] Rate limit — waiting {wait}s (attempt {attempt+1}/3)...",
                      flush=True)
                time.sleep(wait)
            else:
                print(f"  [{subject_key}] API error: {exc}", file=sys.stderr)
                return False

    if response is None:
        print(f"  [{subject_key}] Failed after 3 attempts", file=sys.stderr)
        return False

    summary_text = response.content[0].text.strip()
    summary_path.write_text(summary_text, encoding="utf-8")

    in_tok  = response.usage.input_tokens
    out_tok = response.usage.output_tokens
    # Haiku pricing (as of 2025): $0.25/MTok input, $1.25/MTok output
    cost_usd = (in_tok * 0.00025 + out_tok * 0.00125) / 1000
    print(
        f"  [{subject_key}] Done — {len(summary_text):,} chars saved, "
        f"{in_tok:,} in / {out_tok:,} out tokens, ${cost_usd:.4f}"
    )
    return True


def _resolve_api_key() -> str:
    """Find ANTHROPIC_API_KEY from environment or .env file."""
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if key:
        return key

    # Walk up from this file to find .env
    search = Path(__file__).parent
    for _ in range(5):
        env_file = search / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")
        parent = search.parent
        if parent == search:
            break
        search = parent

    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build subject-level examiner report summaries using Claude Haiku."
    )
    parser.add_argument(
        "--dir",
        default=str(_DEFAULT_KB_DIR),
        help="Path to KB root directory (default: %(default)s)",
    )
    parser.add_argument(
        "--subject",
        help="Only process this subject key, e.g. economics_9708",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate summaries even if they already exist",
    )
    args = parser.parse_args()

    kb_dir = Path(args.dir).expanduser()
    if not kb_dir.exists():
        print(f"ERROR: KB directory not found: {kb_dir}", file=sys.stderr)
        sys.exit(1)

    mpath = kb_dir / "manifest.json"
    if not mpath.exists():
        print(f"ERROR: manifest.json not found at {mpath}", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(mpath.read_text(encoding="utf-8"))

    api_key = _resolve_api_key()
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in environment or .env file", file=sys.stderr)
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed — run: pip install anthropic", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    subjects = (
        [args.subject]
        if args.subject
        else [k for k in manifest if k != "summary"]
    )

    print(f"Building examiner report summaries for {len(subjects)} subject(s)...\n")

    success = 0
    failed  = 0
    for subject_key in subjects:
        if subject_key not in manifest:
            print(f"  [{subject_key}] Not found in manifest — skipping")
            continue
        ok = build_summary(
            subject_key=subject_key,
            manifest_data=manifest[subject_key],
            kb_dir=kb_dir,
            anthropic_client=client,
            force=args.force,
        )
        if ok:
            success += 1
        else:
            failed += 1
        # Pause between subjects to stay under the 50 k tokens/min rate limit
        if subject_key != subjects[-1]:
            print(f"  Waiting 70 s before next subject (rate limit)...", flush=True)
            time.sleep(70)

    print(f"\n=== Summary ===")
    print(f"  Completed: {success}")
    print(f"  Failed:    {failed}")
    print(
        "\nSummary files saved as {kb_dir}/{subject}/subject_summary.txt\n"
        "They will be automatically included in all future prompts for each subject."
    )


if __name__ == "__main__":
    main()
