# BigMart Sales Prediction 🛒

End-to-end machine learning project to predict item sales across BigMart outlets using regression models.

---

## Project Structure

```
bigmart_sales_prediction/
│
├── eda.py                ← Step 1 : Exploratory Data Analysis
├── train.py              ← Step 2 : Train & compare all models
├── evaluate.py           ← Step 3 : Evaluate best model + generate submission
├── tune.py               ← Step 4 : Hyperparameter tuning (optional)
├── requirements.txt
│
├── utils/
│   ├── data_loader.py    ← Load & validate CSVs
│   ├── preprocessor.py   ← Full cleaning + feature engineering pipeline
│   ├── trainer.py        ← Multi-model training with cross-validation
│   └── visualizer.py     ← All plots (EDA, metrics, feature importance)
│
├── data/
│   └── raw/              ← ⬅ Place Train.csv & Test.csv here
│
├── models/               ← Saved .pkl model files (auto-created)
├── outputs/              ← submission.csv (auto-created)
├── logs/                 ← Plots + training logs (auto-created)
│
└── .vscode/
    ├── launch.json       ← 4 run configs (F5 to launch)
    └── settings.json
```

---

## Dataset

Download from **Kaggle**:
> https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data

Or **Analytics Vidhya**:
> https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/

Place `Train.csv` and `Test.csv` inside `data/raw/`.

| Column                     | Description                        |
|----------------------------|------------------------------------|
| Item_Identifier            | Unique product ID                  |
| Item_Weight                | Weight of product (has nulls)      |
| Item_Fat_Content           | Low Fat / Regular (inconsistent)   |
| Item_Visibility            | Display area fraction              |
| Item_Type                  | Product category (16 types)        |
| Item_MRP                   | Maximum Retail Price               |
| Outlet_Identifier          | Store ID                           |
| Outlet_Establishment_Year  | Year store opened                  |
| Outlet_Size                | Small / Medium / High (has nulls)  |
| Outlet_Location_Type       | Tier 1 / 2 / 3                    |
| Outlet_Type                | Grocery / Supermarket type         |
| **Item_Outlet_Sales**      | **Target variable** (train only)   |

---

## Quick Start

### 1. Open in VS Code
```bash
cd bigmart_sales_prediction
code .
```

### 2. Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run in order

```bash
# Step 1 — Understand the data
python eda.py

# Step 2 — Train all models
python train.py

# Step 3 — Evaluate & generate submission
python evaluate.py

# Step 4 — Tune the best model (optional, takes longer)
python tune.py
```

Or use **Run & Debug** (Ctrl+Shift+D) in VS Code and pick a config.

---

## Preprocessing Steps

| Step | Action |
|------|--------|
| Label fixing | Standardises `Item_Fat_Content` (LF → Low Fat, etc.) |
| Missing imputation | `Item_Weight` by item mean; `Outlet_Size` by outlet-type mode |
| Feature engineering | Outlet age, visibility ratio, MRP bins, item category, supermarket flag |
| Encoding | Ordinal for size; label-encode low-cardinality; one-hot the rest |
| Scaling | StandardScaler on all numeric columns |

---

## Models Trained

| Model               | Notes                               |
|---------------------|-------------------------------------|
| Linear Regression   | Baseline                            |
| Ridge Regression    | L2 regularised                      |
| Lasso Regression    | L1 regularised, auto feature select |
| Random Forest       | 300 trees, depth 8                  |
| Gradient Boosting   | 300 trees, lr=0.05                  |
| XGBoost *(optional)*| Install via requirements.txt        |
| LightGBM *(optional)*| Install via requirements.txt       |

**Expected best CV-RMSE ≈ 1050–1100** (Random Forest / XGBoost / LightGBM)

---

## Outputs

| File | Description |
|------|-------------|
| `models/*.pkl` | Best serialised model |
| `outputs/submission.csv` | Test set predictions for submission |
| `logs/feature_importance.png` | Top 20 features |
| `logs/actual_vs_predicted.png` | Scatter plot on train set |
| `logs/residuals.png` | Residual distribution |
| `logs/eda_*.png` | EDA plots |

---

## Requirements

- Python 3.9+
- scikit-learn, pandas, numpy, matplotlib, seaborn
- xgboost, lightgbm *(optional but recommended)*
