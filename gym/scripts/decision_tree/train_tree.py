# gym/scripts/decision_tree/train_tree.py
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from gym.scripts.utils.naming import utc_stamp, safe_name


def main() -> None:
    ap = argparse.ArgumentParser(description="Train scikit-learn DecisionTreeClassifier on generated dataset.")
    ap.add_argument("--dataset", type=str, required=True, help="Path to dataset folder (contains samples.npz)")
    ap.add_argument("--out", type=str, default="", help="Optional output model filename")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--max-depth", type=int, default=10)
    ap.add_argument("--min-samples-leaf", type=int, default=5)
    ap.add_argument("--criterion", type=str, default="gini", choices=["gini", "entropy", "log_loss"])
    args = ap.parse_args()

    dataset_dir = Path(args.dataset)
    npz_path = dataset_dir / "samples.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing samples.npz: {npz_path}")

    data = np.load(npz_path)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=float(args.test_size),
        random_state=int(args.seed),
        stratify=stratify,
    )

    clf = DecisionTreeClassifier(
        criterion=args.criterion,
        max_depth=int(args.max_depth),
        min_samples_leaf=int(args.min_samples_leaf),
        random_state=int(args.seed),
    )

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)

    models_dir = Path("gym/data/models/sklearn")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.out.strip() or f"dt__{safe_name(dataset_dir.name)}__depth{args.max_depth}__seed{args.seed}.joblib"
    model_path = models_dir / model_name
    joblib.dump(clf, model_path)
    model_size = os.path.getsize(model_path)

    meta = {
        "model_name": model_name,
        "created_utc": utc_stamp(),
        "dataset_dir": str(dataset_dir),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "seed": int(args.seed),
        "test_size": float(args.test_size),
        "hyperparams": {
            "criterion": args.criterion,
            "max_depth": int(args.max_depth),
            "min_samples_leaf": int(args.min_samples_leaf),
        },
        "metrics": {
            "accuracy": acc,
            "confusion_matrix_labels": [0, 1, 2],
            "confusion_matrix": cm,
            "classification_report": report_dict,
        },
        "feature_extractor": "gym.policies.utils.features.extract_features",
        "notes": "DecisionTreeClassifier trained via behavior cloning dataset.",
        # --- Métricas nuevas ---
        "wall_time_seconds": round(train_time, 3),
        "model_size_bytes": model_size,
        "tree_depth": int(clf.get_depth()),
        "n_leaves": int(clf.get_n_leaves()),
        "feature_importances": clf.feature_importances_.tolist(),
    }
    model_path.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Model saved: {model_path} ({model_size:,} bytes)")
    print(f"Accuracy: {acc:.4f} | Depth: {clf.get_depth()} | Leaves: {clf.get_n_leaves()} | Time: {train_time:.3f}s")
    print("\nClassification report:\n")
    print(report_text)


if __name__ == "__main__":
    main()