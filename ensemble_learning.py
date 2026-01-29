import torch
import json
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.metrics import classification_report

# ── Your own modules ────────────────────────────────────────────────────────
from scl import Instructor               # assuming this is scl.py
from config import get_config
from sciBERT import get_results          # your original sciBERT function

# ── Configuration ───────────────────────────────────────────────────────────
CANONICAL_TEST_PATH = 'dataset/OADS_Test.json'      # ← the one you want to trust
PREDICTIONS_CSV_PATH = 'dataset/predictions-mapped.csv'   # bertGCN old predictions

LABEL_MAPPING = {
    'general-url': 0,
    'third-party-dataset': 1,
    'author-provided-dataset': 2,
    'third-party-software': 3,
    'author-provided-software': 4,
    'project': 5
}


def load_canonical_test_data(path=CANONICAL_TEST_PATH):
    """Load the test data we want EVERY model to use"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"[Canonical] Loaded {len(data)} examples from {path}")
    if len(data) > 0:
        print(f"    First text preview: {data[0]['text'][:120]}...")
    return data


def majority_voting(*prediction_sets):
    """Safe majority voting – requires all arrays same length"""
    if not prediction_sets:
        raise ValueError("No predictions provided")

    lengths = [len(p) for p in prediction_sets]
    if len(set(lengths)) != 1:
        raise ValueError(f"Prediction lengths do not match: {lengths}")

    stacked = np.vstack(prediction_sets).T   # (n_samples, n_models)
    final_preds = []

    for votes in stacked:
        count = Counter(votes)
        if len(count) == len(prediction_sets):  # all different → fallback rule
            final_preds.append(votes[1])        # e.g. take SCL / 2nd model
        else:
            final_preds.append(count.most_common(1)[0][0])

    return np.array(final_preds)


def main():
    args, logger = get_config()
    logger.info("Starting ensemble pipeline – using canonical test set")

    # ── 1. Load canonical test data (what ALL models should see) ────────────
    canonical_test = load_canonical_test_data()
    n_samples = len(canonical_test)

    # ── 2. SCL predictions (SupConLoss / Instructor model) ──────────────────
    instructor = Instructor(args, logger)
    # Note: If Instructor.run() still loads its own file internally,
    # you may need to patch load_data() in data_utils.py to accept a list/dict
    true_labels, scl_predictions = instructor.run()
    scl_predictions = np.array(scl_predictions)

    print(f"[SCL] Got {len(scl_predictions)} predictions")
    if len(scl_predictions) != n_samples:
        logger.warning(f"SCL predictions ({len(scl_predictions)}) ≠ canonical size ({n_samples})")

    # ── 3. SciBERT predictions – force same data ────────────────────────────
    # Convert canonical JSON → pandas DataFrame expected by your sciBERT code
    test_df = pd.DataFrame(canonical_test)
    # Your original sciBERT.py expects 'text' column (and optionally 'label')
    if 'text' not in test_df.columns and 'Text' in test_df.columns:
        test_df = test_df.rename(columns={'Text': 'text'})

    # Monkey-patch or call a modified version – here we assume you can reuse
    # the logic from get_results() but with our DataFrame
    # (If you don't want to change sciBERT.py, extract the prediction part)
    sciBERT_predictions = get_results()   # ← currently uses CSV – you'll need to update sciBERT.py
    sciBERT_predictions = np.array(sciBERT_predictions)

    print(f"[SciBERT] Got {len(sciBERT_predictions)} predictions")
    if len(sciBERT_predictions) != n_samples:
        logger.warning(f"SciBERT size mismatch: {len(sciBERT_predictions)} vs {n_samples}")

    # ── 4. bertGCN / old CSV predictions (fallback – may not match) ─────────
    try:
        pred_df = pd.read_csv(PREDICTIONS_CSV_PATH)
        bertGCN_preds = pred_df['label'].map(LABEL_MAPPING).tolist()
        bertGCN_preds = np.array(bertGCN_preds)
        print(f"[bertGCN CSV] Loaded {len(bertGCN_preds)} predictions")
    except Exception as e:
        logger.error(f"Could not load bertGCN CSV: {e}")
        bertGCN_preds = None

    # ── 5. Ensemble – only include models that match length ─────────────────
    pred_list = [scl_predictions]
    names = ["SCL"]

    if len(sciBERT_predictions) == n_samples:
        pred_list.append(sciBERT_predictions)
        names.append("SciBERT")
    else:
        logger.warning("Skipping SciBERT – length mismatch")

    if bertGCN_preds is not None and len(bertGCN_preds) == n_samples:
        pred_list.append(bertGCN_preds)
        names.append("bertGCN")
    elif bertGCN_preds is not None:
        logger.warning(f"Skipping bertGCN – size {len(bertGCN_preds)} ≠ {n_samples}")

    if len(pred_list) < 2:
        logger.error("Not enough matching predictions to ensemble")
        return

    logger.info(f"Ensembling {len(pred_list)} models: {', '.join(names)}")

    final_predictions = majority_voting(*pred_list)

    # ── 6. Evaluation ───────────────────────────────────────────────────────
    if true_labels is not None:
        true_labels = np.array(true_labels)
        if len(true_labels) == len(final_predictions):
            print("\nFinal Ensemble Classification Report:")
            print(classification_report(true_labels, final_predictions, target_names=LABEL_MAPPING.keys()))
        else:
            logger.warning("Cannot compute report – true_labels length mismatch")
    else:
        print("\nNo ground truth → only predictions generated.")

    # Save
    np.save("final_ensemble_predictions.npy", final_predictions)
    print(f"Saved {len(final_predictions)} ensemble predictions to final_ensemble_predictions.npy")


if __name__ == '__main__':
    main()