This is the companion code for my [article about the Semantic Entropy concept](https://www.nachobrito.es/artificial-intelligence/semantic-entropy/). 

# Semantic Entropy

A minimal implementation of the black-box discrete **Semantic Entropy** method for detecting LLM hallucinations, based on:
- [*Detecting hallucinations in large language models using semantic entropy*](https://www.nature.com/articles/s41586-024-07421-0) (Farquhar et al., Nature 2024).
- [*Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation*](https://arxiv.org/abs/2302.09664) (Kuhn et al., ICLR 2023).

---

## How It Works

1. **Ask**: Queries a local language model (`Qwen3.5-0.8B` via `llama-cpp-python`) to generate a primary response.
2. **Sample**: Repeats the question across multiple temperature values to assess generation variance.
3. **Cluster**: Groups answers into semantic equivalence classes using bidirectional Natural Language Inference (`microsoft/deberta-large-mnli`).
4. **Compute Entropy**: Calculates discrete semantic entropy across the clusters.
   - **Low Entropy ($\approx 0$)**: Semantic consensus across temperatures $\rightarrow$ Low likelihood of hallucination.
   - **High Entropy ($> 0$)**: Semantic divergence across temperatures $\rightarrow$ Likely hallucination/confabulation.

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/NachoBrito/semantic-entropy.git
cd semantic-entropy
```

### 2. Install dependencies

Using [`uv`](https://github.com/astral-sh/uv) (recommended):

```bash
uv sync
```

---

## Running the Code

### CLI Execution

Run with the default question:
```bash
uv run semantic-entropy
```

Pass a custom question:
```bash
uv run semantic-entropy "What is the capital of France?"
```

*(Alternatively, run as a module: `uv run python -m semantic_entropy "Your question"`)*

### Python API

```python
from semantic_entropy import main

# Run with a question and custom sampling temperatures
response, entropy = main(
    question="Who was the second person to walk on the moon?",
    temperatures=[0.2, 0.5, 0.7, 0.9, 1.0],
)

print(f"Primary Answer: {response}")
print(f"Semantic Entropy: {entropy:.4f} nats")
```

---

## Project Structure

```text
src/semantic_entropy/
├── llm.py               # Local GGUF LLM wrapper (llama-cpp-python)
├── nli.py               # DeBERTa-based NLI clustering & entropy computation
├── semantic_entropy.py  # End-to-end semantic entropy pipeline
├── __init__.py          # Public package API exports
└── __main__.py          # CLI entry point
```
