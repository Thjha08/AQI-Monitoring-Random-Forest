from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Processed Data.xlsx"

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError("Put 'Processed Data.xlsx' in the same folder as main.py.")

    df = pd.read_excel(DATA_FILE)
    df.columns = df.columns.astype(str).str.strip()

    required = {"Location", "AQI"}
    if not required.issubset(df.columns):
        raise ValueError(f"Excel must contain columns: {sorted(required)}")

    # Keep Chennai only.
    df["Location_clean"] = df["Location"].astype(str).str.strip().str.lower()
    df = df[df["Location_clean"] == "chennai"].copy()
    if df.empty:
        raise ValueError("No Chennai rows were found in the Location column.")

    before = len(df)
    df = df.drop_duplicates()
    print("Exact duplicate rows removed:", before - len(df))

    candidate_features = ["PM2.5", "CO", "O3", "Temp", "Humidity"]
    features = [col for col in candidate_features if col in df.columns]
    if not features:
        raise ValueError(f"No expected predictors found. Available columns: {df.columns.tolist()}")

    for col in features + ["AQI"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=features + ["AQI"])
    print("Rows removed for missing/non-numeric predictors or AQI:", before - len(df))

    # Zero AQI is treated as invalid for this dataset; change this if zero is valid in your source.
    before = len(df)
    df = df[df["AQI"] > 0].copy()
    print("Rows removed for AQI <= 0:", before - len(df))

    if len(df) < 10:
        raise ValueError(f"Only {len(df)} valid Chennai rows remain. More data is needed for reliable evaluation.")

    X, y = df[features], df["AQI"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_features=1.0,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("\n--- Chennai AQI: Random Forest Regression ---")
    print("Features:", features)
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")

    results = pd.DataFrame({
        "Actual_AQI": y_test.to_numpy(),
        "Predicted_AQI": y_pred
    })
    results.to_csv(BASE_DIR / "random_forest_test_predictions.csv", index=False)

    # Actual versus predicted plot
    low = min(results.min())
    high = max(results.max())
    plt.figure(figsize=(7, 6))
    plt.scatter(results["Actual_AQI"], results["Predicted_AQI"])
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title("Chennai AQI: Random Forest Actual vs Predicted")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "random_forest_actual_vs_predicted.png", dpi=150)
    plt.show()

    # Feature importance plot
    importance = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values()

    plt.figure(figsize=(8, 4))
    importance.plot(kind="barh")
    plt.xlabel("Feature importance")
    plt.title("Random Forest: Feature Importance")
    plt.tight_layout()
    plt.savefig(
    BASE_DIR / "random_forest_feature_importance.png",
    dpi=150
)
    plt.show()

    # Demonstration prediction using median values (not a real new reading).
    example = pd.DataFrame([{col: float(df[col].median()) for col in features}])
    print("\nExample input using median values:")
    print(example.to_string(index=False))
    print(f"Example predicted AQI: {model.predict(example)[0]:.2f}")

    joblib.dump({"model": model, "features": features},
                BASE_DIR / "chennai_aqi_random_forest.joblib")
    print("\nSaved:")
    print("- random_forest_test_predictions.csv")
    print("- random_forest_actual_vs_predicted.png")
    print("- random_forest_feature_importance.png")
    print("- chennai_aqi_random_forest.joblib")
    print("\nThis is Random Forest regression, not time-series forecasting.")

if __name__ == "__main__":
    main()
