# Data Description

Place the following four CSV files in this `data/` directory. The filenames must exactly match those listed below.

(The default paths are configured in `src/config.py`. If you want to use different filenames or paths, modify that file or set the `DCDSM_DATA_DIR` environment variable to point to another directory.)

| Filename | Description |
|---|---|
| `train_lightGBM_threshold-1.csv` | Training set — handcrafted features (after LightGBM feature selection) |
| `train_bert_lightGBM_processed.csv` | Training set — language model features  |
| `test_lightGBM_threshold-1.csv` | Test set — handcrafted features |
| `test_bert_lightGBM_processed.csv` | Test set — language model features |

## Format Requirements

- Each file must contain the `LABEL` column. All other columns are treated as features.
- Due to HGMD data protection and licensing requirements, the `sequence_id` column has been removed from all released data files.
- Therefore, the handcrafted-feature file and language-model-feature file for the same split must contain exactly the same samples in exactly the same row order. The code assumes that corresponding rows in the two files refer to the same sample, so the row order must not be changed independently.
- The number of columns in the handcrafted features and language model features does not need to exactly match the examples shown here, since it depends on your feature-selection threshold. However, the set of feature column names must be identical between the training and test sets.

## Evaluating a Completely New External Test Set

If you later want to evaluate a trained model on a new test set that is not one of the four files above (for example, another batch of clinical samples), you do not need to overwrite the existing `test_*.csv` files. Simply run:

```bash
python scripts/run_evaluate.py \
    --test-manual data/your_new_test_manual_features.csv \
    --test-bert data/your_new_test_language_model_features.csv \
    --weights outputs/crossattn_bce_tversky_best.weights.h5
```

The script will automatically reload the `train_*.csv` files to reconstruct the scaler used for normalization. It will **not** fit a new scaler on the new test set, which is important. See the notes at the top of `src/evaluate.py` for details.

