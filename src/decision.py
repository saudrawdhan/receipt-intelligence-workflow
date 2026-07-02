"""Stage 4: Decision / Action.

This is where the workflow turns raw model output into a decision. The rules
live in plain Python (not in the model) so they are deterministic, auditable,
and easy to change.

The total is verified two ways: first `subtotal + tax == total`, and if those
fields are missing, we fall back to checking that the line items sum to the
total. The auto-approve amount cap is currency-aware, because a "large" amount
in a cents currency (500 SAR) is tiny in a no-cents currency (500 IDR).
"""
from typing import Optional, List, Literal
from pydantic import BaseModel

from .schema import ReceiptExtraction

CONFIDENCE_THRESHOLD = 0.70
MATH_TOLERANCE_ABS = 0.05
MATH_TOLERANCE_REL = 0.02  # 2%, for large totals

# The auto-approve cap only applies to currencies that use cents. High-value
# amounts there genuinely warrant a manager's approval.
CENTS_CURRENCIES = {"SAR", "USD", "EUR", "GBP", "AED", "KWD", "BHD", "QAR"}
AUTO_APPROVE_CAP = 500.0


class Decision(BaseModel):
    status: Literal["AUTO_APPROVE", "NEEDS_REVIEW", "REJECT"]
    math_check_passed: bool
    computed_total: Optional[float]
    expense_category: str
    reasons: List[str]
    recommended_action: str


def _verify_total(data: ReceiptExtraction) -> tuple[bool, str, Optional[float]]:
    """Check the total. Returns (verified, method, computed_value)."""
    if data.total is None:
        return False, "no total", None
    tol = max(MATH_TOLERANCE_ABS, abs(data.total) * MATH_TOLERANCE_REL)

    if data.subtotal is not None and data.tax is not None:
        computed = round(data.subtotal + data.tax, 2)
        if abs(computed - data.total) <= tol:
            return True, "subtotal + tax", computed

    item_sum = round(sum(i.amount for i in data.line_items if i.amount is not None), 2)
    if item_sum > 0 and abs(item_sum - data.total) <= tol:
        return True, "sum of items", item_sum

    return False, "could not verify", None


def decide(data: ReceiptExtraction) -> Decision:
    if not data.is_receipt:
        return Decision(
            status="REJECT",
            math_check_passed=False,
            computed_total=None,
            expense_category="n/a",
            reasons=[f"Image is not a receipt (detected: {data.image_type})."],
            recommended_action="Reject and ask the user to upload a valid receipt.",
        )

    verified, method, computed = _verify_total(data)
    category = data.suggested_category or "other"
    currency = (data.currency or "").upper()

    reasons: List[str] = []
    if data.total is None:
        reasons.append("No total amount detected.")
    elif not verified:
        reasons.append(f"Total could not be verified ({method}).")
    if data.confidence < CONFIDENCE_THRESHOLD:
        reasons.append(f"Low model confidence ({data.confidence:.2f}).")
    if currency in CENTS_CURRENCIES and data.total is not None and data.total > AUTO_APPROVE_CAP:
        reasons.append(f"Total {data.total} {currency} exceeds the auto-approve limit "
                       f"of {AUTO_APPROVE_CAP} {currency}.")
    if data.uncertainty_notes:
        reasons.append(f"Model flagged: {data.uncertainty_notes}")

    if not reasons:
        return Decision(
            status="AUTO_APPROVE",
            math_check_passed=verified,
            computed_total=computed,
            expense_category=category,
            reasons=[f"All checks passed (total verified by {method})."],
            recommended_action=f"Auto-approve and file under '{category}'.",
        )

    return Decision(
        status="NEEDS_REVIEW",
        math_check_passed=verified,
        computed_total=computed,
        expense_category=category,
        reasons=reasons,
        recommended_action="Send to a human reviewer before approval.",
    )
