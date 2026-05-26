"""
BigMart Sales — Evaluate saved model & generate submission
===========================================================
Usage:
    python evaluate.py
"""

import os
import glob
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from utils.data_loader  import load_data
from utils.preprocessor import Preprocessor
from utils.visualizer   import (plot_actual_vs_predicted,
                                 plot_residuals,
                                 plot_feature_importance)

warnings.filterwarnings("ignore")

MODEL_DIR  = "models"
TRAIN_PATH = "data/raw/Train.csv"
TEST_PATH  = "data/raw/Test.csv"
OUTPUT_DIR = "outputs"
LOG_DIR    = "logs"


def load_latest_model(model_dir: str):
    pkls = glob.glob(os.path.join(model_dir, "*.pkl"))
    if not pkls:
        raise FileNotFoundError(
            f"No .pkl model found in '{model_dir}'. Run train.py first."
        )
    path = max(pkls, key=os.path.getmtime)
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"  Loaded model : {path}")
    return model, os.path.splitext(os.path.basename(path))[0].replace("_", " ")


def main():
    print("=" * 55)
    print("   BigMart Sales — Evaluation")
    print("=" * 55)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Load & preprocess
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)
    preprocessor = Preprocessor()
    X_train, y_train, X_test = preprocessor.fit_transform(train_df, test_df)

    # Load model
    model, name = load_latest_model(MODEL_DIR)

    # Metrics on training set
    y_pred = model.predict(X_train)
    rmse   = np.sqrt(mean_squared_error(y_train, y_pred))
    mae    = mean_absolute_error(y_train, y_pred)
    r2     = r2_score(y_train, y_pred)

    print(f"\n  ── Metrics on Train Set ──────────────────────")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  R²    : {r2:.4f}")

    # Test predictions
    test_preds = np.clip(model.predict(X_test), 0, None)
    submission = pd.DataFrame({
        "Item_Identifier"  : test_df["Item_Identifier"],
        "Outlet_Identifier": test_df["Outlet_Identifier"],
        "Item_Outlet_Sales" : test_preds,
    })
    sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"\n  Submission saved → {sub_path}")
    print(f"  Predicted sales range : {test_preds.min():.2f} – {test_preds.max():.2f}")

    # Plots
    plot_actual_vs_predicted(y_train, y_pred, name,
                             save_path=os.path.join(LOG_DIR, "eval_actual_vs_predicted.png"))
    plot_residuals(y_train, y_pred,
                   save_path=os.path.join(LOG_DIR, "eval_residuals.png"))
    plot_feature_importance(model, X_train.columns.tolist(),
                            save_path=os.path.join(LOG_DIR, "eval_feature_importance.png"))

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
