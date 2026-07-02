"""Stage 2: the Vision-Language Model call.

Sends the image + prompt to Gemini and gets back a validated ReceiptExtraction.
A mock mode returns a canned result so the rest of the pipeline can be tested
without an API key or network call.
"""
import time
from pathlib import Path
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from PIL import Image

from . import config
from .prompt import get_prompt
from .schema import ReceiptExtraction, LineItem

# Transient server-side errors worth retrying (overload / rate limit).
_RETRY_CODES = {429, 500, 503}
_MAX_ATTEMPTS = 6
_MAX_BACKOFF = 30


def _mock_green_market() -> ReceiptExtraction:
    """Canned result matching the real Gemini reading of samples/receipt_green_market.png
    (same numbers as examples/sample_output.json), so the mock demo is not misleading."""
    return ReceiptExtraction(
        image_type="receipt",
        is_receipt=True,
        merchant_name="GREEN MARKET",
        merchant_address="Prince Sultan Rd, Al Khobar",
        transaction_date="2026-06-29",
        transaction_time="19:24",
        currency="SAR",
        line_items=[
            LineItem(description="Fresh Milk 2L", quantity=2, amount=15.0),
            LineItem(description="Whole Wheat Bread", quantity=1, amount=6.5),
            LineItem(description="Free-Range Eggs 30", quantity=1, amount=23.0),
            LineItem(description="Bananas 1.2kg", quantity=1, amount=8.4),
            LineItem(description="Tomatoes 1kg", quantity=1, amount=5.75),
            LineItem(description="Olive Oil 1L", quantity=1, amount=34.9),
            LineItem(description="Chicken Breast 1kg", quantity=1, amount=28.5),
        ],
        subtotal=122.05,
        tax=18.31,
        total=140.36,
        payment_method="Card",
        suggested_category="groceries",
        main_finding="Grocery receipt from Green Market, total 140.36 SAR.",
        confidence=0.99,
        uncertainty_notes=None,
    )


def _mock_non_receipt() -> ReceiptExtraction:
    """Canned result for an image that is not a receipt (the reject branch)."""
    return ReceiptExtraction(
        image_type="drawing",
        is_receipt=False,
        main_finding="The image is a simple drawing of a house, not a receipt.",
        confidence=0.98,
        uncertainty_notes=None,
    )


def _mock_generic() -> ReceiptExtraction:
    """Canned result for any other sample (e.g. the CORD receipts) that has no
    specific fixture of its own."""
    return ReceiptExtraction(
        image_type="receipt",
        is_receipt=True,
        merchant_name="Lulu Hypermarket",
        merchant_address="King Fahd Rd, Dammam",
        transaction_date="2026-06-28",
        transaction_time="18:42",
        currency="SAR",
        line_items=[
            LineItem(description="Milk 2L", quantity=2, unit_price=7.50, amount=15.00),
            LineItem(description="Brown Bread", quantity=1, unit_price=5.00, amount=5.00),
            LineItem(description="Eggs 30pc", quantity=1, unit_price=22.00, amount=22.00),
        ],
        subtotal=42.00,
        tax=6.30,
        total=48.30,
        payment_method="card",
        suggested_category="groceries",
        main_finding="Grocery receipt from Lulu Hypermarket, total 48.30 SAR.",
        confidence=0.93,
        uncertainty_notes=None,
    )


def _mock_from_cord(image_path: str):
    """For a CORD sample, build an offline result from its dataset ground truth,
    so each CORD receipt shows its own real totals instead of one shared fixture.
    Returns None if the image is not a known CORD sample."""
    from .dataset import load_manifest

    manifest = load_manifest() or []
    name = Path(image_path).name
    row = next((r for r in manifest if r["file"] == name), None)
    if row is None:
        return None
    count = int(row.get("item_count") or 0)
    return ReceiptExtraction(
        image_type="receipt",
        is_receipt=True,
        currency="IDR",
        line_items=[LineItem(description=f"Item {i + 1}", quantity=1) for i in range(count)],
        subtotal=row.get("subtotal"),
        tax=row.get("tax"),
        total=row.get("total"),
        suggested_category="other",
        main_finding=f"Indonesian receipt, total {row.get('total')} IDR "
                     f"(offline preview built from the dataset ground truth).",
        confidence=0.90,
        uncertainty_notes="Offline mock: line-item details are not reconstructed, "
                          "only the known totals and item count.",
    )


def _mock_extraction(image_path: str = "") -> ReceiptExtraction:
    """A fixed example used by --mock, chosen by the image so different demo
    images give different (offline) results."""
    if "not_a_receipt" in image_path:
        return _mock_non_receipt()
    if "green_market" in image_path:
        return _mock_green_market()
    cord = _mock_from_cord(image_path)
    if cord is not None:
        return cord
    return _mock_generic()


def analyze_image(image_path: str, mock: bool = False) -> ReceiptExtraction:
    """Run stage 2. Returns a validated ReceiptExtraction."""
    if mock:
        return _mock_extraction(image_path)

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    client = genai.Client(api_key=config.require_api_key())
    image = Image.open(path)
    gen_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ReceiptExtraction,
        temperature=0.0,
    )

    response = _generate_with_retry(client, image, gen_config)

    # When response_schema is a Pydantic model, the SDK parses it for us.
    if response.parsed is not None:
        return response.parsed
    # Fallback: validate the raw JSON text ourselves.
    return ReceiptExtraction.model_validate_json(response.text)


def _generate_with_retry(client, image, gen_config):
    """Call Gemini, retrying transient overload/rate-limit errors with backoff."""
    last_error = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=[get_prompt(), image],
                config=gen_config,
            )
        except genai_errors.APIError as exc:
            if exc.code not in _RETRY_CODES or attempt == _MAX_ATTEMPTS:
                raise
            last_error = exc
            wait = min(2 ** attempt, _MAX_BACKOFF)  # 2,4,8,16,30,30s
            print(f"  Gemini busy ({exc.code}), retrying in {wait}s "
                  f"(attempt {attempt}/{_MAX_ATTEMPTS - 1})...")
            time.sleep(wait)
    raise last_error
