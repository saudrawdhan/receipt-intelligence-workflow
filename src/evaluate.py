"""Evaluation harness: scores the workflow against real CORD ground truth.

Two modes:
  live    - run the full pipeline on each saved CORD receipt (uses API quota)
  offline - re-score predictions already saved in the report against the current
            ground truth, with no API calls (useful when quota is exhausted or to
            re-score after fixing the parser)

It compares the extracted total / subtotal / tax / item-count to the human
annotation and reports per-field accuracy, so we can state real numbers.
"""
import json
import time
from pathlib import Path

from .amounts import parse_cord_amount, amounts_match
from .dataset import SAMPLES_DIR, load_manifest, prepare_samples
from .pipeline import run
from .prompt import current_variant

# Spacing between calls to stay under the free-tier per-minute limit.
THROTTLE_SECONDS = 6
REPORT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "evaluation_report.json"
_FIELDS = ("total", "subtotal", "tax", "item_count")


def _pct(matches: int, counted: int) -> float:
    return round(100 * matches / counted, 1) if counted else 0.0


def _score_row(pred: dict, gt: dict) -> dict:
    """Compare one prediction row against ground truth. Returns checks per field."""
    checks = {
        "total": amounts_match(parse_cord_amount(pred.get("total_pred")), gt["total"]),
        "subtotal": amounts_match(parse_cord_amount(pred.get("subtotal_pred")), gt["subtotal"]),
        "tax": amounts_match(parse_cord_amount(pred.get("tax_pred")), gt["tax"]),
        "item_count": pred.get("items_pred") == gt["item_count"],
    }
    return checks


# Which row key holds the prediction for each field.
_PRED_KEY = {"total": "total_pred", "subtotal": "subtotal_pred",
             "tax": "tax_pred", "item_count": "items_pred"}


def _summarise(rows: list[dict], manifest: dict) -> dict:
    score = {k: [0, 0] for k in _FIELDS}  # [matches, counted]
    for row in rows:
        gt = manifest[row["file"]]
        # Refresh the displayed ground truth so it always matches what was scored.
        row["total_gt"] = gt["total"]
        row["items_gt"] = gt["item_count"]
        checks = _score_row(row, gt)
        for field in _FIELDS:
            truth = gt["item_count"] if field == "item_count" else gt[field]
            has_truth = field == "item_count" or truth is not None
            has_pred = _PRED_KEY[field] in row  # skip fields never captured
            if has_truth and has_pred:
                score[field][1] += 1
                score[field][0] += int(checks[field])
                row[f"{field}_ok"] = checks[field]
    return score


def _print_summary(score: dict, n: int, model: str):
    print("\n" + "=" * 46)
    print("  EVALUATION ON REAL CORD RECEIPTS")
    print("=" * 46)
    print(f"Receipts tested : {n}")
    print(f"Model           : {model}")
    for field in _FIELDS:
        m, c = score[field]
        print(f"{field:11}: {_pct(m, c):5.1f}%   ({m}/{c} correct)")
    print("=" * 46)


def _write_report(model: str, manifest_list: list, rows: list, score: dict):
    report = {
        "dataset": "naver-clova-ix/cord-v2 (test split)",
        "model": model,
        "prompt_variant": current_variant(),
        "n_receipts": len(manifest_list),
        "accuracy_percent": {f: _pct(*score[f]) for f in _FIELDS},
        "per_receipt": rows,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report -> {REPORT_PATH}")


def evaluate(n: int = 15) -> dict:
    """Live evaluation: calls the model on each receipt."""
    manifest_list = load_manifest() or prepare_samples(n)
    manifest = {g["file"]: g for g in manifest_list}

    rows = []
    model = None
    for i, gt in enumerate(manifest_list, 1):
        result = run(str(SAMPLES_DIR / gt["file"]), mock=False)
        model = result.model
        e = result.extraction
        rows.append({
            "file": gt["file"],
            "total_pred": e.total,
            "subtotal_pred": e.subtotal,
            "tax_pred": e.tax,
            "items_pred": len(e.line_items),
            "items_gt": gt["item_count"],
            "decision": result.decision.status,
            "confidence": e.confidence,
        })
        ok = amounts_match(parse_cord_amount(e.total), gt["total"])
        print(f"[{i:2}/{len(manifest_list)}] {'OK' if ok else 'XX'} {gt['file']}  "
              f"total pred={e.total} gt={gt['total']}  items {len(e.line_items)}/{gt['item_count']}")
        if i < len(manifest_list):
            time.sleep(THROTTLE_SECONDS)

    score = _summarise(rows, manifest)
    _print_summary(score, len(manifest_list), model)
    _write_report(model, manifest_list, rows, score)
    return {"accuracy_percent": {f: _pct(*score[f]) for f in _FIELDS}}


def rescore_offline() -> dict:
    """Re-score predictions already in the report against the current manifest."""
    if not REPORT_PATH.exists():
        raise RuntimeError("No report to re-score. Run a live evaluation first.")
    old = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    manifest_list = load_manifest()
    manifest = {g["file"]: g for g in manifest_list}
    rows = old["per_receipt"]

    score = _summarise(rows, manifest)
    _print_summary(score, len(rows), old.get("model", "unknown") + " (offline re-score)")
    _write_report(old.get("model", "unknown"), manifest_list, rows, score)
    return {"accuracy_percent": {f: _pct(*score[f]) for f in _FIELDS}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the workflow on CORD")
    parser.add_argument("-n", type=int, default=15, help="how many receipts (live mode)")
    parser.add_argument("--offline", action="store_true",
                        help="re-score saved predictions without calling the API")
    args = parser.parse_args()
    rescore_offline() if args.offline else evaluate(args.n)
