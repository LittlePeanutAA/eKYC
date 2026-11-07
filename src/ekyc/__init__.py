"""eKYC: information extraction from Vietnamese citizen ID cards (CCCD)."""
from .pipeline import EKYC, CardNotFoundError, ExtractionResult, FieldResult, draw

__version__ = "2.0.0"
__all__ = ["EKYC", "CardNotFoundError", "ExtractionResult", "FieldResult", "draw", "__version__"]
