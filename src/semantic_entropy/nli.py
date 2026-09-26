"""Natural Language Inference (NLI) clustering for Semantic Entropy.

Demonstrates semantic equivalence clustering using DeBERTa, as introduced
in Kuhn et al. ("Semantic Uncertainty: Linguistic Invariances for Uncertainty
Estimation in Natural Language Generation") and Farquhar et al. ("Detecting
hallucinations in large language models using semantic entropy", Nature 2024).
"""

from collections import Counter
import math
from typing import Sequence
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "microsoft/deberta-large-mnli"
ENTAILMENT_LABEL_ID = 2  # In MNLI, label index 2 corresponds to 'ENTAILMENT'

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()


def check_entailment(premise: str, hypothesis: str) -> bool:
    """Return True if premise entails hypothesis."""
    if premise.strip() == hypothesis.strip():
        return True

    inputs = tokenizer(
        premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    return predicted_class == ENTAILMENT_LABEL_ID


def are_equivalent(text_a: str, text_b: str) -> bool:
    """Return True if both texts bidirectionally entail each other."""
    return check_entailment(text_a, text_b) and check_entailment(text_b, text_a)


def get_semantic_ids(texts: Sequence[str]) -> list[int]:
    """Assign a cluster ID to each text based on bidirectional entailment."""
    cluster_representatives: list[str] = []
    semantic_ids: list[int] = []

    for text in texts:
        # Check if text matches an existing cluster's representative
        assigned_id = None
        for cluster_id, representative in enumerate(cluster_representatives):
            if are_equivalent(text, representative):
                assigned_id = cluster_id
                break

        # If it doesn't match any existing cluster, start a new one
        if assigned_id is None:
            assigned_id = len(cluster_representatives)
            cluster_representatives.append(text)

        semantic_ids.append(assigned_id)

    return semantic_ids


def cluster_strings(texts: Sequence[str]) -> list[list[str]]:
    """Group texts into clusters of semantically equivalent meanings."""
    ids = get_semantic_ids(texts)
    clusters: dict[int, list[str]] = {}
    for text, cluster_id in zip(texts, ids):
        clusters.setdefault(cluster_id, []).append(text)
    return list(clusters.values())


def calculate_semantic_entropy(semantic_ids: Sequence[int]) -> float:
    """Calculate discrete semantic entropy (in nats) from cluster assignments.

    Following Farquhar et al. (Nature 2024), the black-box discrete semantic entropy
    approximates cluster probabilities P(C_i | x) directly from the proportion of
    sampled answers in each cluster:
        P(C_i | x) = count(C_i) / M
        SE(x) = - sum_{i=1}^{|C|} P(C_i | x) * log(P(C_i | x))

    Args:
        semantic_ids: Cluster assignment integer for each sampled answer.

    Returns:
        float: The semantic entropy value (>= 0.0 nats).
    """
    if not semantic_ids:
        return 0.0

    total = len(semantic_ids)
    counts = Counter(semantic_ids)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0.0:
            entropy -= p * math.log(p)

    return float(entropy)