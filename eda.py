"""
BigMart Sales — Exploratory Data Analysis
==========================================
Run this before training to understand the dataset.
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_loader  import load_data
from utils.visualizer   import plot_eda

warnings.filterwarnings("ignore")

TRAIN_PATH = "data/raw/Train.csv"
TEST_PATH  = "data/raw/Test.csv"
SAVE_DIR   = "logs"


def main():
    print("=" * 55)
    print("   BigMart Sales — Exploratory Data Analysis")
    print("=" * 55)

    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

    # ── Basic info ────────────────────────────────────────────────────────────
    print("\n── Shape ────────────────────────────────────────────────")
    print(f"  Train : {train_df.shape}")
    print(f"  Test  : {test_df.shape}")

    print("\n── Train dtypes ─────────────────────────────────────────")
    print(train_df.dtypes.to_string())

    print("\n── Missing values (Train) ───────────────────────────────")
    missing = train_df.isnull().sum()
    missing = missing[missing > 0]
    pct = (missing / len(train_df) * 100).round(2)
    print(pd.DataFrame({"Missing": missing, "Pct (%)": pct}).to_string())

    print("\n── Target summary ───────────────────────────────────────")
    print(train_df["Item_Outlet_Sales"].describe().round(2).to_string())

    print("\n── Unique values per categorical column ─────────────────")
    cat_cols = train_df.select_dtypes("object").columns
    for col in cat_cols:
        vals = train_df[col].value_counts()
        print(f"\n  {col} ({len(vals)} unique):")
        print(vals.to_string(header=False))

    print("\n── Item_Fat_Content label inconsistencies ───────────────")
    print(train_df["Item_Fat_Content"].value_counts().to_string())

    print("\n── Outlet_Establishment_Year counts ─────────────────────")
    print(train_df["Outlet_Establishment_Year"].value_counts().sort_index().to_string())

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n── Generating EDA plots → logs/ …")
    plot_eda(train_df, save_dir=SAVE_DIR)

    print("\nEDA complete!")


if __name__ == "__main__":
    main()
