"""Stage 5 rendering: turn the WorkflowResult into a clean human-readable report."""
from .pipeline import WorkflowResult

_STATUS_ICON = {"AUTO_APPROVE": "[OK]", "NEEDS_REVIEW": "[REVIEW]", "REJECT": "[REJECT]"}


def render_report(result: WorkflowResult) -> str:
    e = result.extraction
    d = result.decision
    lines = []
    lines.append("=" * 52)
    lines.append("  RECEIPT VLM WORKFLOW - FINAL RESULT")
    lines.append("=" * 52)
    lines.append(f"Image      : {result.image_path}")
    lines.append(f"Model      : {result.model}")
    lines.append("")
    lines.append("-- Structured Output (from the VLM) --")
    lines.append(f"Image type : {e.image_type}")
    lines.append(f"Merchant   : {e.merchant_name or '-'}")
    lines.append(f"Date       : {e.transaction_date or '-'}  {e.transaction_time or ''}".rstrip())
    lines.append(f"Currency   : {e.currency or '-'}")
    if e.line_items:
        lines.append("Items      :")
        for it in e.line_items:
            qty = f" x{it.quantity:g}" if it.quantity else ""
            amt = f"{it.amount:.2f}" if it.amount is not None else "-"
            lines.append(f"             - {it.description}{qty}  =>  {amt}")
    lines.append(f"Subtotal   : {e.subtotal if e.subtotal is not None else '-'}")
    lines.append(f"Tax        : {e.tax if e.tax is not None else '-'}")
    lines.append(f"Total      : {e.total if e.total is not None else '-'} {e.currency or ''}".rstrip())
    lines.append(f"Payment    : {e.payment_method or '-'}")
    lines.append(f"Finding    : {e.main_finding}")
    lines.append(f"Confidence : {e.confidence:.2f}")
    if e.uncertainty_notes:
        lines.append(f"Uncertainty: {e.uncertainty_notes}")
    lines.append("")
    lines.append("-- Decision / Action --")
    lines.append(f"Status     : {_STATUS_ICON.get(d.status, '')} {d.status}")
    lines.append(f"Math check : {'passed' if d.math_check_passed else 'not verified'}"
                 + (f" (computed {d.computed_total})" if d.computed_total is not None else ""))
    lines.append(f"Category   : {d.expense_category}")
    lines.append("Reasons    :")
    for r in d.reasons:
        lines.append(f"             - {r}")
    lines.append(f"Next step  : {d.recommended_action}")
    lines.append("=" * 52)
    return "\n".join(lines)
