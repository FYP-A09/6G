"""Evaluate frozen 5G-NIDD SSL embeddings with downstream classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

try:
    from .masked_reconstruction import FlowFeatureEncoder, MaskedReconstructionConfig
    from .preprocessing import NIDDPreprocessor, load_nidd_csv, split_by_sequence
except ImportError:
    from masked_reconstruction import FlowFeatureEncoder, MaskedReconstructionConfig
    from preprocessing import NIDDPreprocessor, load_nidd_csv, split_by_sequence


def load_trained_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = FlowFeatureEncoder(
        config=MaskedReconstructionConfig(**checkpoint["model_config"]),
        category_sizes=checkpoint["category_sizes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    preprocessor = NIDDPreprocessor.from_metadata(checkpoint["preprocessing"])
    return model, preprocessor, checkpoint


def generate_representations(
    frame: pd.DataFrame,
    model: FlowFeatureEncoder,
    preprocessor: NIDDPreprocessor,
    mask_ratio: float,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return SSL embeddings, processed raw features, and labels."""
    batch = preprocessor.transform_with_mask(frame, mask_ratio=mask_ratio, seed=seed)
    ssl_embeddings = []
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            output = model(
                batch.context[start : start + batch_size].to(device),
                batch.numeric[start : start + batch_size].to(device),
            )
            ssl_embeddings.append(output.embedding.cpu().numpy())
    raw_features = torch.cat([batch.context.float(), batch.targets], dim=1).numpy()
    labels = frame["Label"].astype(str).to_numpy()
    return np.concatenate(ssl_embeddings), raw_features, labels


def score_predictions(y_true: np.ndarray, predictions: np.ndarray) -> dict:
    classes = sorted(set(y_true))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=classes, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "micro_f1": float(f1_score(y_true, predictions, average="micro")),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(classes)
        },
    }


def evaluate_representation(
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
) -> dict:
    classifier = LogisticRegression(max_iter=1000, random_state=seed)
    classifier.fit(features[train_indices], labels[train_indices])
    predictions = classifier.predict(features[test_indices])
    return {"representation": name, **score_predictions(labels[test_indices], predictions)}


def evaluate_nidd(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    n_rows: int | None = None,
    batch_size: int = 256,
    seed: int = 0,
    device_name: str = "cpu",
) -> dict:
    """Evaluate representations from checkpoint training rows to validation rows."""
    device = torch.device(device_name)
    model, preprocessor, checkpoint = load_trained_model(checkpoint_path, device)
    frame = load_nidd_csv(dataset_path, n_rows=n_rows)
    training_frame, validation_frame = split_by_sequence(
        frame,
        validation_fraction=checkpoint["training_config"]["validation_fraction"],
    )
    training_frame = training_frame.loc[training_frame["Label"].notna()].reset_index(drop=True)
    validation_frame = validation_frame.loc[validation_frame["Label"].notna()].reset_index(drop=True)
    frame = pd.concat([training_frame, validation_frame], ignore_index=True)
    ssl_features, raw_features, labels = generate_representations(
        frame,
        model,
        preprocessor,
        mask_ratio=checkpoint["model_config"]["mask_ratio"],
        seed=seed,
        batch_size=batch_size,
        device=device,
    )
    random_features = np.random.default_rng(seed).standard_normal(ssl_features.shape)
    train_indices = np.arange(len(training_frame))
    test_indices = np.arange(len(training_frame), len(frame))
    results = [
        evaluate_representation("ssl_embedding_64d", ssl_features, labels, train_indices, test_indices, seed),
        evaluate_representation("processed_raw_features", raw_features, labels, train_indices, test_indices, seed),
        evaluate_representation("random_embedding_64d", random_features, labels, train_indices, test_indices, seed),
    ]
    report = {
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(checkpoint_path),
        "rows_evaluated": len(labels),
        "evaluation_split": "checkpoint_sequence_training_to_validation",
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "label_counts": {label: int((labels == label).sum()) for label in sorted(set(labels))},
        "seed": seed,
        "results": results,
    }
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path")
    parser.add_argument("checkpoint_path")
    parser.add_argument("output_path")
    parser.add_argument("--n-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evaluate_nidd(
        args.dataset_path,
        args.checkpoint_path,
        args.output_path,
        n_rows=args.n_rows,
        batch_size=args.batch_size,
        seed=args.seed,
        device_name=args.device,
    )
