# Scale-Agnostic Image Quality Assessment via Magnitude-Aware Ranking

**Presented at QoMEX 2026**  
Adam Zizien, Karel Fliegel  
<!-- [Paper](#) · [Pretrained Models (Release)](#) · [Dataset Scores](Evaluation_results/) -->

**Try a model of your choice in your browser (no install):** https://zizieada.github.io/wphl/

---

## Overview

This repository accompanies the paper *Scale-Agnostic Image Quality Assessment via Magnitude-Aware Ranking*. The core contribution is the **Weighted Pairwise Hinge Loss (WPHL)**, which is a loss function that incorporates label magnitude into both the margin and the sample weighting. This enables the model to be sensitive not just to the *direction* of quality differences, but to their *magnitude*, without requiring score normalization across training sets.

The framework is evaluated on top of [CKDN](https://github.com/researchmm/CKDN) (the reimplementation provided in [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch/blob/main/pyiqa/archs/ckdn_arch.py)), a knowledge-distillation-based IQA model, and the retrained weights are provided.

The weights are released in three formats: PyTorch checkpoints, TorchScript traces, and ONNX exports. The two ONNX models run directly in the browser through the hosted scorer at https://zizieada.github.io/wphl/, with no Python or install, and no image leaves your machine. See [Pretrained Models](#pretrained-models) below.

---

## Repository Structure

```
├── WeightedPairwiseHinge.py      # Proposed loss function (PyTorch)
├── train_example.py              # Minimal example: pairwise training with WPHL
├── index.html                    # Browser-based ONNX scorer (also hosted, link above)
├── Evaluation_results/
│   ├── README.md                 # Dataset disclosure and citation guidance
│   └── *.csv                     # Subjective scores + objective metric
```

---

## Loss Function

The **Weighted Pairwise Hinge Loss** is implemented in `WeightedPairwiseHinge.py`.

### Quick Start

```python
import torch
from WeightedPairwiseHinge import WeightedPairwiseHinge

criterion = WeightedPairwiseHinge(margin=0.5, p=0.5)

# Predicted scores for a batch of image pairs
o1 = torch.tensor([0.8, 0.3, 0.6])
o2 = torch.tensor([0.5, 0.7, 0.4])

# Ground-truth quality labels (any scale — e.g. MOS)
y1 = torch.tensor([4.2, 2.1, 3.8])
y2 = torch.tensor([2.8, 3.9, 2.5])

# Ranking target: +1 if image 1 is better, -1 otherwise
target = torch.sign(y1 - y2)

loss, acc = criterion(o1, o2, y1, y2, target)
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `margin` | `0.1` | Base margin scale factor |
| `p` | `1.0` | Exponent for importance weighting |
| `eps` | `1e-6` | Numerical stability term in the weighted mean |
| `initial_max_loss_coeff` | `0.0` | Blend coefficient λ for the max-loss term (0 = mean only) |

---

## Pretrained Models

Retrained CKDN weights are available in the latest release, in three formats:

| File | Format | Description |
|---|---|---|
| `CKDN_ImageNet.pth` | PyTorch checkpoint | Best-performing model. Requires the CKDN model class to load (the IQA-PyTorch variant to be precise). |
| `*.pt` | TorchScript (traced) | Other retrained variants. Loadable with `torch.jit.load`, no model definition required. |
| `CKDN_ImageNet.onnx` | ONNX | Best-performing model, exported for use without PyTorch. |
| `CKDN_WPHL+.onnx` | ONNX | Variant trained with WPHL on an expanded, multi-dataset pool with a longer schedule. Included to show that the WPHL training signal continues to hold with more data, without any manual MOS alignment. |

### Run in the browser (ONNX, no Python)

Both `.onnx` files run client-side in any modern browser, either through the hosted page at **https://zizieada.github.io/wphl/** or by opening `iqa_scorer.html` from this repository locally. No Python, PyTorch, or install is required, and no image leaves your machine.

1. **Load model** — select `CKDN_ImageNet.onnx` or `CKDN_WPHL+.onnx`.
2. **Choose reference image** — the original / undistorted image.
3. **Choose distorted image(s)** — one or many versions to evaluate.
4. **Compute scores**, then **Export CSV**.

Both models take two inputs `(dist, ref)` in that order and emit a single scalar score per pair. Whether a higher or lower score means "better" depends on the model's training direction. The raw outputs are not calibrated to a fixed scale, so interpret them against your own validation set.

> Note: each distorted image should depict the same content as the reference, or the score is meaningless. The first browser run fetches ONNX Runtime Web from a CDN; see the comments at the bottom of `iqa_scorer.html` for fully offline use.

### Loading a Traced Model

```python
import torch

model = torch.jit.load("CKDN_ImageNet_traced.pt", map_location="cpu")
model.eval()
```

---

## Training Example

`train_example.py` demonstrates how to integrate WPHL into a pairwise training loop with a generic IQA backbone. See the file for the full annotated example.

---

## Citation

If you use the loss function, pretrained models, or evaluation data from this repository, please cite:

```bibtex
@inproceedings{zizien2026wphl,
  title     = {Scale-Agnostic Image Quality Assessment via Magnitude-Aware Ranking},
  author    = {Zizien, Adam and Fliegel, Karel},
  booktitle = {Proceedings of the International Conference on Quality of Multimedia Experience (QoMEX)},
  year      = {2026}
}
```

If you use the evaluation scores in `Evaluation_results/`, please also cite the original dataset authors as listed in [Evaluation_results/README.md](Evaluation_results/README.md).

---

## License

This repository (code, pretrained models, and evaluation data) is released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt the material for any purpose, including commercial use, provided appropriate credit is given, a link to the license is included, and any changes are indicated.

Note that the **underlying datasets** referenced in `Evaluation_results/` are the work of their respective authors and may carry separate license terms. Consult the original dataset papers before redistributing that data.
