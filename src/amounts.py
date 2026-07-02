"""Money parsing helpers used by the evaluation harness.

The CORD receipts are Indonesian rupiah, which has no cents. The annotations are
inconsistent about separators - the same dataset uses both "60.000" and "28,000"
to mean sixty/twenty-eight thousand. Because there are no real decimals, the safe
and correct normalisation is to drop every separator and keep the digits.
"""
import re
from typing import Optional


def parse_cord_amount(value) -> Optional[float]:
    """Normalise a rupiah amount (str like '28,000' or a number) to a float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    digits = re.sub(r"[^\d]", "", str(value))
    return float(digits) if digits else None


def amounts_match(predicted: Optional[float], truth: Optional[float],
                  rel_tol: float = 0.01) -> bool:
    """True if two amounts agree within 1% (or 1 unit for small values)."""
    if predicted is None or truth is None:
        return False
    return abs(predicted - truth) <= max(1.0, abs(truth) * rel_tol)
