"""Semantic Entropy for Hallucination Detection in LLMs.

Demonstrates the black-box discrete semantic entropy approach introduced in:
- Farquhar et al., "Detecting hallucinations in large language models using semantic entropy",
  Nature 630, 625–630 (2024). https://doi.org/10.1038/s41586-024-07421-0
- Kuhn et al., "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation
  in Natural Language Generation", ICLR 2023.

In the black-box setting where model sequence log-probabilities are unavailable,
we sample M responses across varied temperatures, group semantically equivalent
responses via bidirectional Natural Language Inference (NLI) using DeBERTa, and
compute the entropy of the resulting empirical cluster distribution:
    P(C_i | x) = |C_i| / M
    SE(x) = - sum_i P(C_i | x) * log(P(C_i | x))
"""

from collections import Counter
import sys
from typing import Sequence

from .llm import ask_llm
from .nli import (
    calculate_semantic_entropy,
    cluster_strings,
    get_semantic_ids,
)

__all__ = [
    "main",
]


def main(
    question: str | None = None,
    temperatures: Sequence[float] = (0.2, 0.5, 0.7, 0.9, 1.0),
) -> tuple[str, float]:
    """Ask a question, sample at different temperatures, cluster via NLI, and compute semantic entropy.

    Demonstrates black-box hallucination detection by measuring the semantic
    dispersion of answers generated under different temperatures.

    Args:
        question: The question to ask the LLM. If None, uses command-line
            arguments or a default factual question.
        temperatures: Sequence of temperatures for repeated sampling.

    Returns:
        tuple[str, float]: (response, entropy_value) where `response` is the
            primary LLM answer and `entropy_value` is the discrete semantic
            entropy in nats.
    """
    if question is None:
        if len(sys.argv) > 1:
            question = " ".join(sys.argv[1:])
        else:
            question = "Who was the second person to walk on the moon?"

    print("=" * 70)
    print("DEMONSTRATION: SEMANTIC ENTROPY FOR HALLUCINATION DETECTION")
    print("=" * 70)
    print(f"\n[Step 1] Input Question:\n  \"{question}\"")

    # 1. Ask initial question (at low temperature to obtain the primary response)
    print("\n[Step 2] Asking primary question (temperature=0.1)...")
    primary_response = ask_llm(question, temperature=0.1)
    print(f"  Primary Response: \"{primary_response}\"")

    # 2. Repeat question across different temperature values
    print(f"\n[Step 3] Repeating question with {len(temperatures)} different temperature values:")
    sampled_answers: list[str] = [primary_response]
    print(f"  - Temp 0.1 (primary): \"{primary_response}\"")

    for temp in temperatures:
        ans = ask_llm(question, temperature=temp)
        sampled_answers.append(ans)
        print(f"  - Temp {temp:.1f}:          \"{ans}\"")

    # 3. Cluster answers using NLI (bidirectional entailment)
    print(f"\n[Step 4] Clustering {len(sampled_answers)} answers by semantic equivalence using NLI...")
    semantic_ids = get_semantic_ids(sampled_answers)

    clusters: dict[int, list[str]] = {}
    for ans, cluster_id in zip(sampled_answers, semantic_ids):
        clusters.setdefault(cluster_id, []).append(ans)

    for cluster_id, texts in sorted(clusters.items()):
        print(f"  Cluster {cluster_id} ({len(texts)}/{len(sampled_answers)} answers):")
        for text in texts:
            print(f"    * \"{text}\"")

    # 4. Calculate semantic entropy
    print("\n[Step 5] Calculating Semantic Entropy (discrete / black-box):")
    total_answers = len(sampled_answers)
    counts = Counter(semantic_ids)
    for cluster_id, count in sorted(counts.items()):
        prob = count / total_answers
        print(f"  P(Cluster {cluster_id}) = {count}/{total_answers} = {prob:.3f}")

    entropy = calculate_semantic_entropy(semantic_ids)
    print(f"\n  Calculated Semantic Entropy: {entropy:.4f} nats")

    if entropy < 0.3:
        print("  Confidence Interpretation: LOW ENTROPY.")
        print("  -> High semantic consensus across temperatures (likely factual / correct).")
    else:
        print("  Confidence Interpretation: HIGH ENTROPY.")
        print("  -> Semantic divergence across temperatures (likely hallucination / confabulation).")

    print("=" * 70)
    return primary_response, entropy


if __name__ == "__main__":
    main()
