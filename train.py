"""
train.py
--------
Trains a Random Forest Classifier on the food ordering dataset and
serialises the fitted model to models/rf_model.pkl.

Run directly:
    python backend/train.py
"""

import pathlib
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Ensure local backend imports work when run from project root
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from preprocess import load_and_preprocess

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "rf_model.pkl"


def train(n_estimators: int = 100, random_state: int = 42) -> dict:
    """
    Train Random Forest, evaluate, save model.

    Returns
    -------
    dict with keys: model, feature_names, train_acc, test_acc,
                    report, conf_matrix, X_test, y_test
    """
    # 1. Load preprocessed data
    print("[train] Loading and preprocessing data ...")
    data = load_and_preprocess()
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    feature_names = data["feature_names"]

    # 2. Train
    print(f"[train] Training RandomForestClassifier "
          f"(n_estimators={n_estimators}, random_state={random_state}) ...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 3. Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))
    report    = classification_report(y_test, model.predict(X_test),
                                      target_names=["No Order", "Order"])
    conf_mat  = confusion_matrix(y_test, model.predict(X_test))

    print(f"\n[train] Train accuracy : {train_acc:.4f}")
    print(f"[train] Test  accuracy : {test_acc:.4f}")
    print(f"\n[train] Classification Report:\n{report}")
    print(f"[train] Confusion Matrix:\n{conf_mat}")

    # 4. Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n[train] Model saved -> {MODEL_PATH}")

    # 5. Top 10 feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    print("\n[train] Top-10 Feature Importances:")
    for rank, idx in enumerate(indices, 1):
        print(f"  {rank:2d}. {feature_names[idx]:<45s} {importances[idx]:.4f}")

    return {
        "model":         model,
        "feature_names": feature_names,
        "train_acc":     train_acc,
        "test_acc":      test_acc,
        "report":        report,
        "conf_matrix":   conf_mat,
        "X_test":        X_test,
        "y_test":        y_test,
    }


if __name__ == "__main__":
    train()
