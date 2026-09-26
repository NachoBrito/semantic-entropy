"""Semantic Entropy for Hallucination Detection in LLMs.

Demonstrates the black-box discrete semantic entropy approach introduced in:
- Farquhar et al., "Detecting hallucinations in large language models using semantic entropy",
  Nature 630, 625–630 (2024). https://doi.org/10.1038/s41586-024-07421-0
- Kuhn et al., "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation
  in Natural Language Generation", ICLR 2023.
"""

from .llm import ask_llm
from .nli import (
    calculate_semantic_entropy,
    cluster_strings,
    get_semantic_ids,
)
from .semantic_entropy import main

__all__ = [
    "ask_llm",
    "calculate_semantic_entropy",
    "cluster_strings",
    "get_semantic_ids",
    "main",
]
