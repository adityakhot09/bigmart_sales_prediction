"""
BigMart Sales — Hyperparameter Tuning
=======================================
Tunes the best model using GridSearchCV / RandomizedSearchCV.
Edit TUNE_MODEL below to choose which model to tune.

Usage:
    python tune.py
"""

import warnings
import pickle
import os
import numpy as np

from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics         import mean_squared_error

from utils.data_loader  import load_data
from utils.preprocessor import Preprocessor

warnings.filterwarnings("ignore")

TUNE_MODEL = "random_forest"   # "random_forest" | "gradient_boosting"
N_ITER     = 30
CV_FOLDS   = 3
RANDOM_STATE = 42
MODEL_DIR  = "models"

PARAM_GRIDS = {
    "random_forest": {
        "n_estimators"  : [200, 300, 500],
        "max_depth"     : [6, 8, 10, None],
        "min_samples_leaf": [2, 4, 8],
        "max_features"  : ["sqrt", "log2", 0.5],
    },
    "gradient_boosting": {
        "n_estimators"  : [200, 300, 500],
        "learning_rate" : [0.02, 0.05, 0.1],
        "max_depth"     : [3, 5, 7],
        "subsample"     : [0.6, 0.8, 1.0],
    },
}

BASE_MODELS = {
    "random_forest"    : RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}


def main():
    print("=" * 55)
    print(f"   Tuning : {TUNE_MODEL}")
    print("=" * 55)

    train_df, test_df = load_data("data/raw/Train.csv", "data/raw/Test.csv")
    preprocessor = Preprocessor()
    X_train, y_train, _ = preprocessor.fit_transform(train_df, test_df)

    model  = BASE_MODELS[TUNE_MODEL]
    params = PARAM_GRIDS[TUNE_MODEL]

    search = RandomizedSearchCV(
        model, params,
        n_iter=N_ITER,
        cv=CV_FOLDS,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=2,
    )

    print(f"\n  Running {N_ITER} iterations × {CV_FOLDS}-fold CV …\n")
    search.fit(X_train, y_train)

    best = search.best_estimator_
    cv_rmse = np.sqrt(-search.best_score_)
    best.fit(X_train, y_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, best.predict(X_train)))

    print(f"\n  Best params : {search.best_params_}")
    print(f"  CV RMSE     : {cv_rmse:.4f}")
    print(f"  Train RMSE  : {train_rmse:.4f}")

    # Save tuned model
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{TUNE_MODEL}_tuned.pkl")
    with open(path, "wb") as f:
        pickle.dump(best, f)
    print(f"\n  Tuned model saved → {path}")


if __name__ == "__main__":
    main()
