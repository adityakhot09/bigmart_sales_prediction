"""
BigMart Sales Prediction — Training Pipeline
=============================================
Runs full pipeline: load → preprocess → train → evaluate → save model.
"""

import os
import warnings
import json
from datetime import datetime

import pandas as pd
import numpy as np

from utils.data_loader   import load_data
from utils.preprocessor  import Preprocessor
from utils.trainer       import ModelTrainer
from utils.visualizer    import (plot_feature_importance,
                                  plot_actual_vs_predicted,
                                  plot_residuals)

warnings.filterwarnings("ignore")


# ─── Config ────────────────────────────────────────────────────────────────────

CONFIG = {
    "train_path"  : "data/raw/Train.csv",
    "test_path"   : "data/raw/Test.csv",
    "output_dir"  : "outputs",
    "model_dir"   : "models",
    "log_dir"     : "logs",
    "target"      : "Item_Outlet_Sales",
    "random_state": 42,
    "cv_folds"    : 5,
}


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("   BigMart Sales Prediction Pipeline")
    print("=" * 60)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["model_dir"],  exist_ok=True)
    os.makedirs(CONFIG["log_dir"],    exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading data …")
    train_df, test_df = load_data(CONFIG["train_path"], CONFIG["test_path"])
    print(f"      Train : {train_df.shape}   Test : {test_df.shape}")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("\n[2/5] Preprocessing …")
    preprocessor = Preprocessor(target=CONFIG["target"])
    X_train, y_train, X_test = preprocessor.fit_transform(train_df, test_df)
    print(f"      Features : {X_train.shape[1]}   Train rows : {X_train.shape[0]}")

    # ── 3. Train models ───────────────────────────────────────────────────────
    print("\n[3/5] Training models …")
    trainer = ModelTrainer(random_state=CONFIG["random_state"],
                           cv_folds=CONFIG["cv_folds"])
    results = trainer.train_all(X_train, y_train)

    # ── 4. Evaluate & select best ─────────────────────────────────────────────
    print("\n[4/5] Evaluation results:")
    print(f"  {'Model':<28}  {'RMSE':>10}  {'R²':>8}")
    print("  " + "-" * 50)
    for name, res in results.items():
        print(f"  {name:<28}  {res['rmse']:>10.4f}  {res['r2']:>8.4f}")

    best_name  = min(results, key=lambda k: results[k]["rmse"])
    best_model = trainer.models[best_name]
    best_res   = results[best_name]
    print(f"\n  ✓ Best model : {best_name}  (RMSE={best_res['rmse']:.4f}  R²={best_res['r2']:.4f})")

    # ── 5. Save artefacts ─────────────────────────────────────────────────────
    print("\n[5/5] Saving artefacts …")

    # Predictions
    y_pred_train = best_model.predict(X_train)
    preds_df = pd.DataFrame({"Actual": y_train, "Predicted": y_pred_train})
    preds_path = os.path.join(CONFIG["output_dir"], "train_predictions.csv")
    preds_df.to_csv(preds_path, index=False)

    # Test submission
    test_preds = best_model.predict(X_test)
    submission = pd.DataFrame({
        "Item_Identifier"  : test_df["Item_Identifier"],
        "Outlet_Identifier": test_df["Outlet_Identifier"],
        "Item_Outlet_Sales" : np.clip(test_preds, 0, None),
    })
    submission_path = os.path.join(CONFIG["output_dir"], "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"  Submission → {submission_path}")

    # Save model
    trainer.save_model(best_name, CONFIG["model_dir"])

    # Log results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {"timestamp": timestamp, "best_model": best_name, "results": {
        k: {m: float(v) for m, v in r.items()} for k, r in results.items()
    }}
    with open(os.path.join(CONFIG["log_dir"], f"run_{timestamp}.json"), "w") as f:
        json.dump(log, f, indent=2)

    # Plots
    print("\n  Generating plots …")
    plot_feature_importance(best_model, X_train.columns.tolist(),
                            save_path=os.path.join(CONFIG["log_dir"], "feature_importance.png"))
    plot_actual_vs_predicted(y_train, y_pred_train, best_name,
                             save_path=os.path.join(CONFIG["log_dir"], "actual_vs_predicted.png"))
    plot_residuals(y_train, y_pred_train,
                   save_path=os.path.join(CONFIG["log_dir"], "residuals.png"))

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Best model  : {best_name}")
    print(f"  RMSE        : {best_res['rmse']:.4f}")
    print(f"  R²          : {best_res['r2']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
