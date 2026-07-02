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


def _mock_receipt() -> ReceiptExtraction:
    """Canned result for a normal receipt image."""
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


def _mock_non_receipt() -> ReceiptExtraction:
    """Canned result for an image that is not a receipt (the reject branch)."""
    return ReceiptExtraction(
        image_type="drawing",
        is_receipt=False,
        main_finding="The image is a simple drawing of a house, not a receipt.",
        confidence=0.98,
        uncertainty_notes=None,
    )


def _mock_extraction(image_path: str = "") -> ReceiptExtraction:
    """A fixed example used by --mock, chosen by filename so different demo
    images give different (but still fake, offline) results."""
    if "not_a_receipt" in image_path:
        return _mock_non_receipt()
    return _mock_receipt()


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
