"""Pulls a small set of REAL receipts from the CORD-v2 dataset.

CORD (Consolidated Receipt Dataset, naver-clova-ix/cord-v2 on Hugging Face) holds
real Indonesian receipts, each with a human-annotated ground-truth JSON. We stream
a handful, save the images locally, and extract the fields we care about (total,
subtotal, tax, item count) so the evaluation harness can score our pipeline
against real ground truth.
"""
import json
from pathlib import Path
from typing import Optional

from .amounts import parse_cord_amount

DATASET_ID = "naver-clova-ix/cord-v2"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "cord_samples"
MANIFEST = SAMPLES_DIR / "ground_truth.json"


def _menu_items(gt_parse: dict) -> list:
    menu = gt_parse.get("menu", [])
    if isinstance(menu, dict):  # a single item is stored as a dict, not a list
        return [menu]
    return menu if isinstance(menu, list) else []


def _parse_ground_truth(raw: str) -> dict:
    """Reduce CORD's verbose annotation to the fields we evaluate."""
    gt_parse = json.loads(raw).get("gt_parse", {})
    sub_total = gt_parse.get("sub_total", {}) or {}
    total = gt_parse.get("total", {}) or {}
    return {
        "total": parse_cord_amount(total.get("total_price")),
        "subtotal": parse_cord_amount(sub_total.get("subtotal_price")),
        "tax": parse_cord_amount(sub_total.get("tax_price")),
        "item_count": len(_menu_items(gt_parse)),
    }


def prepare_samples(n: int = 15) -> list[dict]:
    """Download n receipts from CORD's test split and save them locally.

    Returns a manifest list of {file, total, subtotal, tax, item_count}.
    """
    from datasets import load_dataset

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    stream = load_dataset(DATASET_ID, split="test", streaming=True)

    manifest = []
    for i, example in enumerate(stream):
        if i >= n:
            break
        file_name = f"receipt_{i:02d}.png"
        example["image"].convert("RGB").save(SAMPLES_DIR / file_name)
        gt = _parse_ground_truth(example["ground_truth"])
        gt["file"] = file_name
        manifest.append(gt)

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest() -> Optional[list[dict]]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download real CORD receipts")
    parser.add_argument("-n", type=int, default=15, help="how many receipts")
    args = parser.parse_args()
    rows = prepare_samples(args.n)
    print(f"Saved {len(rows)} receipts to {SAMPLES_DIR}")
