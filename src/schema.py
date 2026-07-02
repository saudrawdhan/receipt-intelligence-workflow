"""Structured-output schema.

This Pydantic model is handed directly to Gemini as the response schema, so the
model is forced to return JSON in exactly this shape. It is the contract for
stage 3 of the workflow (Structured Output).
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str = Field(description="Name of the purchased item")
    quantity: Optional[float] = Field(default=None, description="Units bought, if shown")
    unit_price: Optional[float] = Field(default=None, description="Price per unit, number only")
    amount: Optional[float] = Field(default=None, description="Line total, number only")


class ReceiptExtraction(BaseModel):
    """Everything the Vision-Language Model reads from the image."""

    # --- What kind of image is this ---
    image_type: str = Field(description="Short label: receipt, invoice, or what it actually is")
    is_receipt: bool = Field(description="True only if this is a receipt or invoice")

    # --- Main information detected ---
    merchant_name: Optional[str] = None
    merchant_address: Optional[str] = None
    transaction_date: Optional[str] = Field(default=None, description="ISO YYYY-MM-DD if possible")
    transaction_time: Optional[str] = None
    currency: Optional[str] = Field(default=None, description="ISO code or symbol")
    line_items: List[LineItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    payment_method: Optional[str] = Field(default=None, description="cash, card, etc.")
    suggested_category: Optional[str] = Field(
        default=None,
        description="groceries, dining, fuel, electronics, pharmacy, transport, utilities, other",
    )

    # --- Finding + confidence ---
    main_finding: str = Field(description="One short sentence summarizing the receipt")
    confidence: float = Field(description="Model self-rated confidence 0.0-1.0", ge=0.0, le=1.0)
    uncertainty_notes: Optional[str] = Field(
        default=None, description="Anything blurry, cut off, or ambiguous"
    )
