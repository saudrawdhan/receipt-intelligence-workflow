"""Command-line entry point for the receipt VLM workflow.

Usage:
    python -m app.cli samples/receipt1.jpg
    python -m app.cli samples/receipt1.jpg --mock          # no API key needed
    python -m app.cli samples/receipt1.jpg --json out.json  # also save JSON
"""
import argparse
import json
import sys
from pathlib import Path

# Allow running as `python app/cli.py ...` as well as `python -m app.cli`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run
from src.report import render_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Receipt VLM mini-workflow")
    parser.add_argument("image", help="Path to the receipt image")
    parser.add_argument("--mock", action="store_true",
                        help="Use a canned result instead of calling the model")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="Optional path to also write the full result as JSON")
    args = parser.parse_args()

    try:
        result = run(args.image, mock=args.mock)
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the user
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(render_report(result))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nSaved JSON -> {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
