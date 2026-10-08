# DCDSM

> **DCDSM: An effective method for driver synonymous mutation prediction with a dual-channel fusion network**

DCDSM is a deep learning model designed for **driver synonymous mutation prediction**.  
In this study, we hypothesize that if a synonymous mutation has been validated as pathogenic in the germline and also appears as a somatic mutation in tumors, then it may be a cancer driver synonymous mutation.

To enhance the representation ability of DNA sequences, DCDSM integrates features from three large-scale DNA sequence pre-trained language models:

- **DNABERT**
- **HyenaDNA**
- **ChemicalBERT**

In addition, the model incorporates eight types of handcrafted features and processes them using a **dual-channel fusion network**. Based on this design, DCDSM applies **CNN-BiLSTM** to extract both local and global features, and then employs a **cross-attention mechanism and gated units** for feature fusion, thereby improving the prediction performance of driver synonymous mutations.

---


---

## 📁 Project Structure


```text
dcdsm-cross-attn/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── docs/
│   └── DCDSM_framework.png
├── outputs/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── fusion.py
│   ├── losses.py
│   ├── model.py
│   ├── data_utils.py
│   ├── train.py
│   ├── evaluate.py
│   └── compat.py
└── scripts/
    ├── run_train.py
    └── run_evaluate.py
```

---

## 📊 Data

The `data/` directory contains the feature files required for model training and evaluation.

Please refer to:

```text
data/README.md
```

The model uses four feature files:

- Handcrafted features for the training set;
- Language-model features for the training set;
- Handcrafted features for the test set;
- Language-model features for the test set.

Make sure that the filenames are consistent with those specified in `data/README.md`.

---

## 🧠 Model

The `src/` and `scripts/` directories contain the core implementation of DCDSM:

| File | Description |
|---|---|
| `src/config.py` | Model configuration, paths, random seed, and training hyperparameters |
| `src/fusion.py` | Cross-attention and gated feature-fusion module |
| `src/losses.py` | BCE + Tversky hybrid loss |
| `src/model.py` | CNN-BiLSTM encoders, fusion module, and classification head |
| `src/data_utils.py` | Data loading, normalization, and feature-dimension processing |
| `src/train.py` | Model training and 10-fold cross-validation |
| `src/evaluate.py` | Evaluation on the test set or an external dataset |
| `src/compat.py` | Compatibility utilities for different h5py versions |
| `scripts/run_train.py` | Entry point for model training |
| `scripts/run_evaluate.py` | Entry point for model evaluation |

---

## ⚙️ Environment Setup

Create and activate a conda environment:

```bash
conda create -n dcdsm python=3.9
conda activate dcdsm
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

> The fusion module uses `tf.keras.layers.MultiHeadAttention`, which requires a compatible TensorFlow version.

---

## 🚀 Usage

### Model Training

Run:

```bash
python scripts/run_train.py
```

The training script performs 10-fold stratified cross-validation, saves fold-specific model weights and prediction results, and evaluates the selected model on the test set.

The default output files are saved in the `outputs/` directory.

### Evaluation on an External Test Set

Run:

```bash
python scripts/run_evaluate.py \
    --test-manual data/new_test_manual_features.csv \
    --test-bert data/new_test_language_model_features.csv \
    --weights outputs/crossattn_bce_tversky_best.weights.h5
```

If no arguments are provided, the script uses the test set configured in `data/README.md`.

---

## 📝 Notes

- The random seed is fixed at `42` for reproducibility.
- The classification threshold is fixed at `0.5`.
- The normalization parameters should be obtained from the training set and should not be refitted on the test set.
- The saved `.h5` files contain model weights only; the model architecture must be reconstructed before loading the weights.
- It is recommended to use the same TensorFlow and h5py versions as those used during model training.

---

## Citation

If you use DCDSM in your research, please cite the corresponding paper or repository.
