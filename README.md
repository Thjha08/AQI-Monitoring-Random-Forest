# Chennai AQI — Random Forest Regression (VS Code)

This project uses **RandomForestRegressor**, not time-series forecasting.

## Setup
Keep `main.py`, `requirements.txt`, and `Processed Data.xlsx` in the same folder.

Open the folder in VS Code and run in the terminal:

```bash
source .venv/bin/activate
python -m pip install pandas openpyxl scikit-learn matplotlib joblib
python main.py
```

If the virtual environment has no pip, use:
```bash
uv pip install --index-url https://pypi.org/simple pandas openpyxl scikit-learn matplotlib joblib
uv run --python .venv/bin/python main.py
```

## What it does
- Filters Chennai rows
- Removes exact duplicates and rows with missing/non-numeric predictors or target
- Excludes AQI <= 0 (adjust if zero is valid in your source)
- Trains Random Forest regression
- Reports MAE, RMSE and R²
- Saves test predictions, actual-vs-predicted graph, feature-importance graph, and model

Metrics can be unstable with a small dataset. If AQI is calculated from the same pollutant inputs, the model learns that relationship rather than predicting future AQI.
