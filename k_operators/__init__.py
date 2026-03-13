"""K-Operators language model package."""

from .train_app import main as train_main
from .infer_app import main as infer_main

__all__ = ["train_main", "infer_main"]
