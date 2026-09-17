"""
Slice-type classification proxy for evaluating SSL embedding quality (Sriranjana
C's task — see docs/team/sriranjana_c/design.md, "Evaluation proxy").

Scores embeddings via Micro-F1 on slice-type classification, using labels from
the already-in-repo Kaggle slice-labeled sets (deepslice_secure5g/,
network_slicing_puspakmeher/, network_slicing_5g_amohankumar/) as a stand-in
downstream task — this lets the SSL stage be validated before the TGNN exists to
test against for real.

Design skeleton — the scoring loop is real and runnable (sklearn), but the
`embed_row` hook is a placeholder until masked_reconstruction.py's encoder is
trained.
"""

from __future__ import annotations

import csv
import os

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

LABELED_DATASETS = {
    "network_slicing_puspakmeher": {
        "path": "data/raw/network_slicing_puspakmeher/deepslice_data.csv",
        "label_col": "slice Type",
    },
    "network_slicing_5g_amohankumar_train": {
        "path": "data/raw/network_slicing_5g_amohankumar/train_dataset.csv",
        "label_col": "slice Type",
    },
}


def embed_row(row: dict) -> list[float]:
    """
    Placeholder — should call FlowFeatureEncoder.encode() from
    masked_reconstruction.py once it's trained. Returns a dummy embedding
    (a hash-based deterministic vector) purely so the scoring pipeline below is
    exercisable end-to-end before the real encoder exists.
    """
    h = hash(tuple(sorted(row.items())))
    return [((h >> (i * 4)) & 0xF) / 15.0 for i in range(8)]  # 8-d dummy, not the real D=64


def load_labeled_rows(dataset_key: str, limit: int = 5000) -> tuple[list[list[float]], list[str]]:
    spec = LABELED_DATASETS[dataset_key]
    path = os.path.join(REPO_ROOT, spec["path"])
    embeddings, labels = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            label = row.get(spec["label_col"])
            if label is None:
                continue
            embeddings.append(embed_row(row))
            labels.append(label)
    return embeddings, labels


def evaluate_micro_f1(dataset_key: str) -> float:
    """Trains a quick classifier on top of the embeddings and reports Micro-F1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    X, y = load_labeled_rows(dataset_key)
    if len(set(y)) < 2:
        raise ValueError(f"Only one class found in {dataset_key} — check label_col mapping.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = LogisticRegression(max_iter=500).fit(X_train, y_train)
    preds = clf.predict(X_test)
    score = f1_score(y_test, preds, average="micro")
    print(f"{dataset_key}: Micro-F1 = {score:.4f} (embed_row is still a placeholder — "
          f"re-run once the real SSL encoder from masked_reconstruction.py is trained)")
    return score


if __name__ == "__main__":
    for key in LABELED_DATASETS:
        path = os.path.join(REPO_ROOT, LABELED_DATASETS[key]["path"])
        if os.path.exists(path):
            try:
                evaluate_micro_f1(key)
            except Exception as e:
                print(f"{key}: skipped — {e}")
        else:
            print(f"{key}: not found at {path}")
