# DCDSM Cross-Attention (Approach D + BCE/Tversky Hybrid Loss)

A dual-branch CNN-BiLSTM multimodal binary classification model that combines **language model features + handcrafted features**. The fusion layer uses **bidirectional cross-attention + gated residual connections** (Approach D), and the loss function uses a **BCE + Tversky hybrid loss** to balance Sensitivity/Specificity and AUC/AUPRC.

This configuration was selected from multiple approaches evaluated during the exploratory stage, including different fusion layers (token self-attention / GMU / SE channel recalibration / direct concatenation / cross-attention) and different loss functions (pure BCE / pure Tversky / BCE + Tversky hybrid / hard-negative mining).

The final selected configuration has been organized into a minimal, reproducible implementation suitable for publication on GitHub, with unused experimental code branches removed.

## Directory Structure

```text
dcdsm-cross-attn/
├── README.md                This file
├── requirements.txt         Python dependencies (see comments in the file;
│                            using pip freeze to lock exact versions is strongly recommended)
├── .gitignore
├── data/                    Training/test CSV files
│                            (not committed to Git; see data/README.md)
│   └── README.md
├── outputs/                 Training/evaluation outputs: weights, plots,
│                            OOF predictions, prediction results
│                            (not committed to Git)
├── src/                     Core code, organized as a Python package
│   ├── __init__.py
│   ├── config.py            Global configuration: paths, random seed,
│   │                        model architecture, and training hyperparameters
│   ├── fusion.py            Fusion module: Approach D, bidirectional
│   │                        cross-attention + gated residual connections
│   ├── losses.py            Loss function: BCE + Tversky hybrid loss
│   ├── model.py             Full model assembly
│   │                        (CNN-BiLSTM encoders + fusion layer + classification head)
│   ├── data_utils.py        Data loading, normalization, and dimension adjustment
│   ├── train.py             Main training logic
│   │                        (10-fold cross-validation + test-set evaluation)
│   ├── evaluate.py          Load weights and evaluate on the bundled
│   │                        or a completely new external test set
│   └── compat.py            Compatibility patch for different h5py versions
│                            (only needed in certain environments; see comments)
└── scripts/                 Command-line entry points
                            (responsible for adding the repository root to sys.path)
    ├── run_train.py
    └── run_evaluate.py
```

## Environment Setup

```bash
conda create -n dcdsm python=3.9   # Use the Python version used during training;
                                   # 3.9 is only an example
conda activate dcdsm
pip install -r requirements.txt
```

> `tf.keras.layers.MultiHeadAttention` requires TensorFlow >= 2.4.  
> The fusion layer (Approach D) uses this layer. If the TensorFlow version in your environment is too old, model construction will fail with an error such as:
>
> `AttributeError: module '...keras...layers' has no attribute 'MultiHeadAttention'`
>
> This is an environment issue rather than a problem with the code. Upgrade TensorFlow or switch to an environment that supports the model.

## Data Preparation

See [`data/README.md`](data/README.md).

Place the four CSV files in the `data/` directory:

- one handcrafted-feature file for training,
- one language-model-feature file for training,
- one handcrafted-feature file for testing,
- one language-model-feature file for testing.

Make sure the filenames match those specified in `data/README.md`.

## Model Training

```bash
python scripts/run_train.py
```

The script performs the following procedure:

10-fold stratified cross-validation (`StratifiedKFold`, with `random_state` fixed to `RANDOM_SEED=42` in `src/config.py`) → train a model for each fold → record Acc/AUC/AUPRC/Sn/Sp/MCC and save the weights and loss curves for each fold → select the best fold based on validation accuracy → evaluate the selected model on the test set → generate ROC/PR curves and save per-sample prediction results.

All outputs are written to `outputs/` by default. The filename prefix is determined by:

```python
MODEL_TAG = "crossattn_bce_tversky"
```

in `config.py`.

| File | Description |
|---|---|
| `crossattn_bce_tversky_fold{N}.weights.h5` | Model weights for each fold |
| `crossattn_bce_tversky_loss_fold{N}.png` | Training/validation loss curves for each fold |
| `crossattn_bce_tversky_oof.npy` | Out-of-fold prediction probabilities from 10-fold cross-validation |
| `crossattn_bce_tversky_best.weights.h5` | Weights from the fold with the highest validation accuracy (final model) |
| `crossattn_bce_tversky_roc_test.png` / `_pr_test.png` | ROC / PR curves on the test set |
| `crossattn_bce_tversky_test_predictions.csv` | Per-sample prediction results on the test set |

## Evaluation on a New External Test Set

```bash
python scripts/run_evaluate.py \
    --test-manual data/new_test_manual_features.csv \
    --test-bert data/new_test_language_model_features.csv \
    --weights outputs/crossattn_bce_tversky_best.weights.h5
```

If no arguments are provided, the script uses the bundled test set described in `data/README.md` together with the best weights generated during training.

**There are two common pitfalls here. The script already handles both of them, but understanding the underlying principles will make it easier to reuse the code in the future:**

1. **The normalization reference must come from the training set. Do not fit the scaler again on a new test set.**

   The script reloads `train_*.csv` to reconstruct the `StandardScaler`. This procedure is deterministic: as long as the contents of the training CSV files remain unchanged, the resulting scaler will be identical to the one used during the original training process.

   The reconstructed scaler is then used to `transform` the new test set.

   If `fit_transform` is applied directly to the new data, the normalization reference will be incorrect and all evaluation metrics may become distorted.

2. **The weight file contains only the model weights; it does not contain the network architecture.**

   Therefore, the evaluation script first reconstructs exactly the same architecture used during training by calling `create_cnn_model()` from `src/model.py`, and then calls `load_weights`.

   This ensures that the network architecture matches the saved weights.

## Reproducibility

- All sources of randomness (`numpy` / Python `random` / `tensorflow` / `StratifiedKFold` / training-set shuffling) use the same `RANDOM_SEED=42` defined in `src/config.py`.

- The decision threshold is fixed at `0.5`. No threshold search is performed, ensuring that the same threshold is used throughout training and evaluation.

- **The checkpoint files contain weights only and do not include the network architecture.** Therefore, reproduction requires using exactly the same architecture defined in `src/model.py` before calling `load_weights`.

- It is strongly recommended to use exactly the same TensorFlow version as the original training environment. See the notes in `requirements.txt` and the recommendation to use `pip freeze`.

- Major TensorFlow version changes (for example, moving from TensorFlow 2.0 to 2.4+) may introduce incompatibilities in the way HDF5 weight files are read.

  If an error such as

  ```text
  AttributeError: 'str' object has no attribute 'decode'
  ```

  occurs, refer to the explanations and compatibility fixes in `src/compat.py`.

## Methodology

### Fusion Layer: Approach D — Bidirectional Cross-Attention + Gated Residual Connections

The language model features and handcrafted features are first independently encoded by CNN-BiLSTM encoders.

Bidirectional cross-attention is then performed between the two modalities:

- language model → handcrafted features
- handcrafted features → language model

Each direction includes a residual connection followed by Layer Normalization.

Finally, a learnable gating vector is used to perform a weighted combination of the outputs from the two attention directions.

References:

- Tsai et al., ACL 2019 (MulT)
- Nagrani et al., NeurIPS 2021

The original baseline code directly applied `tf.keras.layers.Attention()` to 2D features. However, this layer expects 3D inputs.

When applied to 2D inputs, the operation can degenerate into attention **between samples**, potentially causing information leakage within a batch, making the results dependent on batch size and sample order, and preventing proper single-sample inference. In addition, this approach does not provide learnable Q/K/V projections.

Approach D addresses these issues while preserving the original design intent of cross-attention.

### Loss Function: BCE + Tversky Hybrid Loss

Under the pure BCE baseline, Sensitivity was high while Specificity was low.

Replacing BCE with pure Tversky Loss,

```text
TP / (TP + α·FP + β·FN)
```

with:

```text
α = 0.7
β = 0.3
```

places a greater penalty on false positives and improves the balance between Sensitivity and Specificity.

However, AUC/AUPRC decreased substantially.

One possible explanation is that pure Tversky loss primarily focuses on the overlap represented by TP/FP/FN around a fixed decision threshold and does not directly constrain the overall ranking quality of predicted probabilities.

Therefore, the two losses are combined as:

```text
0.3 · BCE + 0.7 · Tversky
```

to balance these objectives. This follows the general idea of combining complementary loss functions, as commonly seen in medical image segmentation frameworks such as nnU-Net.

## References

- Tsai, Y.-H. H., et al.  
  "Multimodal Transformer for Unaligned Multimodal Language Sequences."  
  ACL 2019. (MulT, cross-attention)

- Nagrani, A., et al.  
  "Attention Bottlenecks for Multimodal Fusion."  
  NeurIPS 2021.

- Salehi, S. S. M., et al.  
  "Tversky loss function for image segmentation using 3D fully convolutional deep networks."  
  MICCAI-DLMIA 2017.
