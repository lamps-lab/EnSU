import torch
import json
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.metrics import classification_report

# Your own modules
from scl import Instructor
from config import get_config
from sciBERT import get_results

# ── Configuration ───────────────────────────────────────────────────────────
CANONICAL_TEST_PATH = 'dataset/OADS_Test.json'
PREDICTIONS_CSV_PATH = 'dataset/predictions-mapped.csv'

LABEL_MAPPING = {
    'general-url': 0,
    'third-party-dataset': 1,
    'author-provided-dataset': 2,
    'third-party-software': 3,
    'author-provided-software': 4,
    'project': 5
}


def load_canonical_test_data(path=CANONICAL_TEST_PATH):
    """Load test data – handles different text key names like 'Text' / 'text'"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"[Canonical] Loaded {len(data)} examples from {path}")
    
    if not data:
        raise ValueError("JSON file is empty")
    
    # Detect text key
    first_item = data[0]
    possible_text_keys = ['text', 'Text', 'sentence', 'content', 'abstract']
    text_key = None
    for key in possible_text_keys:
        if key in first_item and isinstance(first_item[key], str) and len(first_item[key].strip()) > 10:
            text_key = key
            break
    
    if text_key is None:
        raise KeyError(
            f"Could not find text field. Keys in first item: {list(first_item.keys())}"
        )
    
    print(f"[Canonical] Using text field: '{text_key}'")
    
    # Preview
    preview = first_item[text_key][:120].replace('\n', ' ').strip()
    print(f"    First text preview: {preview}...")
    
    # Normalize to 'text' for consistency
    for item in data:
        if text_key != 'text' and text_key in item:
            item['text'] = item.pop(text_key)
    
    return data


def majority_voting(*prediction_sets):
    if not prediction_sets:
        raise ValueError("No predictions provided")

    lengths = [len(p) for p in prediction_sets]
    if len(set(lengths)) != 1:
        raise ValueError(f"Prediction lengths do not match: {lengths}")

    stacked = np.vstack(prediction_sets).T
    final_preds = []

    for votes in stacked:
        count = Counter(votes)
        if len(count) == len(prediction_sets):  # all different → fallback to second (SCL)
            final_preds.append(votes[1])
        else:
            final_preds.append(count.most_common(1)[0][0])

    return np.array(final_preds)


def main():
    args, logger = get_config()
    logger.info("Starting ensemble – using canonical test set")

    # 1. Load canonical test data
    canonical_test = load_canonical_test_data()
    n_samples = len(canonical_test)

    # 2. SCL predictions
    instructor = Instructor(args, logger)
    true_labels, scl_predictions = instructor.run()
    scl_predictions = np.array(scl_predictions)
    print(f"[SCL] Got {len(scl_predictions)} predictions")

    if len(scl_predictions) != n_samples:
        logger.warning(f"SCL size mismatch: {len(scl_predictions)} vs canonical {n_samples}")

    # 3. SciBERT predictions
    sciBERT_predictions = get_results()
    sciBERT_predictions = np.array(sciBERT_predictions)
    print(f"[SciBERT] Got {len(sciBERT_predictions)} predictions")

    if len(sciBERT_predictions) != n_samples:
        logger.warning(f"SciBERT size mismatch: {len(sciBERT_predictions)} vs canonical {n_samples}")

    # 4. bertGCN (old CSV) – likely smaller, will be skipped unless regenerated
    bertGCN_preds = None
    try:
        pred_df = pd.read_csv(PREDICTIONS_CSV_PATH)
        bertGCN_preds = pred_df['label'].map(LABEL_MAPPING).tolist()
        bertGCN_preds = np.array(bertGCN_preds)
        print(f"[bertGCN CSV] Loaded {len(bertGCN_preds)} predictions")
        if len(bertGCN_preds) != n_samples:
            logger.warning(f"bertGCN size {len(bertGCN_preds)} ≠ canonical {n_samples} → skipping")
            bertGCN_preds = None
    except Exception as e:
        logger.error(f"Could not load bertGCN CSV: {e}")

    # 5. Build list of predictions that match length
    pred_list = [scl_predictions]
    names = ["SCL"]

    if len(sciBERT_predictions) == n_samples:
        pred_list.append(sciBERT_predictions)
        names.append("SciBERT")

    if bertGCN_preds is not None and len(bertGCN_preds) == n_samples:
        pred_list.append(bertGCN_preds)
        names.append("bertGCN")

    if len(pred_list) < 2:
        logger.warning("Not enough matching predictions to ensemble. Using SCL only.")
        final_predictions = scl_predictions
    else:
        logger.info(f"Ensembling {len(pred_list)} models: {', '.join(names)}")
        final_predictions = majority_voting(*pred_list)

    # 6. Evaluation
    if true_labels is not None:
        true_labels = np.array(true_labels)
        if len(true_labels) == len(final_predictions):
            print("\nEnsemble Classification Report:")
            print(classification_report(true_labels, final_predictions, target_names=LABEL_MAPPING.keys()))
        else:
            print("True labels length mismatch – cannot compute full report")

    # Save
    np.save("final_ensemble_predictions.npy", final_predictions)
    print(f"Saved {len(final_predictions)} predictions to final_ensemble_predictions.npy")


if __name__ == '__main__':
    main()