"""The workflow orchestrator: wires the five stages together.

  Image Input  ->  Vision-Language Model  ->  Structured Output
       ->  Decision / Action  ->  Final Result
"""
from typing import Optional
from pydantic import BaseModel

from .schema import ReceiptExtraction
from .decision import Decision, decide
from .vlm_client import analyze_image


class WorkflowResult(BaseModel):
    """Stage 5: the final bundled result (extraction + decision)."""
    image_path: str
    model: str
    extraction: ReceiptExtraction
    decision: Decision


def run(image_path: str, mock: bool = False, model: Optional[str] = None) -> WorkflowResult:
    from . import config

    # Stage 1 + 2 + 3: image in -> VLM -> structured output
    extraction = analyze_image(image_path, mock=mock)

    # Stage 4: decision / action
    decision = decide(extraction)

    # Stage 5: final result
    return WorkflowResult(
        image_path=image_path,
        model="mock" if mock else (model or config.GEMINI_MODEL),
        extraction=extraction,
        decision=decision,
    )
