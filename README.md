# Transformers and Attention: Does A appear before B?

A tiny transformer encoder classifier built in PyTorch, following the Chapter 15
live build guide. It reads a 6-token sequence and predicts whether A comes before B.
All the work and explanations are in [`transformers_attention.ipynb`](transformers_attention.ipynb).

## Results

| | Validation accuracy |
|---|---|
| Baseline (majority class) | 0.50 |
| Without positions | 0.50 |
| With positions | 1.00 |

**Final test accuracy (with positions): 1.00**, checked once at the end.

- **What attention adds:** each token's vector becomes a weighted mix of every
  token in the sequence, so it carries context and not just its own identity.
- **Why position matters:** without position embeddings, a sequence and its
  A/B-swapped partner get the exact same scores, so the model is stuck at 50%.
- **Limitation:** it only knows 8 tokens and length-6 sequences. It can't handle
  longer sequences or real text.

## How to run it

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
```

Open the notebook in VS Code, pick the `.venv` kernel, and click **Run All**.
CPU only, and no downloads are needed. The seeds are fixed, so you get the same numbers.
Python 3.14.7, torch 2.15.0.

## Reference

Géron, A. (2025). *Hands-On Machine Learning with Scikit-Learn and PyTorch.* O'Reilly Media. Chapter 15.
