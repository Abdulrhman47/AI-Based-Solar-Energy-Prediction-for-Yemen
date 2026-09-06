import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. LOAD CLEAN DATASET
# ==========================================

file_name = "yemen_solar_cleaned.csv"

df = pd.read_csv(file_name)

print("=" * 55)
print("YEMEN SOLAR RADIATION - MACHINE LEARNING")
print("=" * 55)

print("\nDataset loaded successfully")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# ==========================================
# 2. TARGET
# ==========================================

target = "Solar_Radiation_kWh_m2_day"

drop_columns = [
    target,
    "Date"
]

X = df.drop(columns=drop_columns, errors="ignore")
y = df[target]

valid_rows = X.notna().all(axis=1) & y.notna()

X = X.loc[valid_rows]
y = y.loc[valid_rows]

print("\nTraining samples:", len(X))

# ==========================================
# 3. DETECT CATEGORICAL FEATURES
# ==========================================

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=[np.number]
).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

# ==========================================
# 4. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining set:", len(X_train))
print("Testing set :", len(X_test))

# ==========================================
# 6. MODELS
# ==========================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=150,
            random_state=42
        )
}

# ==========================================
# 7. TRAIN AND EVALUATE
# ==========================================

results = []
predictions = {}
trained_models = {}

for name, model in models.items():

    print("\n" + "=" * 55)
    print("Training:", name)
    print("=" * 55)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    trained_models[name] = pipeline

    y_pred = pipeline.predict(X_test)

    predictions[name] = y_pred

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2  :", round(r2, 4))

# ==========================================
# 8. MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)

print("\n")
print("=" * 55)
print("MODEL COMPARISON")
print("=" * 55)

print(results_df.to_string(index=False))

# ==========================================
# 9. BEST MODEL
# ==========================================

best_model = results_df.iloc[0]

print("\n")
print("=" * 55)
print("BEST MODEL")
print("=" * 55)

print("Model :", best_model["Model"])
print("MAE   :", round(best_model["MAE"], 4))
print("RMSE  :", round(best_model["RMSE"], 4))
print("R2    :", round(best_model["R2"], 4))

# ==========================================
# 10. SAVE BEST MODEL
# ==========================================

best_model_name = best_model["Model"]

final_model = trained_models[best_model_name]

joblib.dump(
    final_model,
    "best_solar_model.pkl"
)

print("\nBest model saved successfully")
print("Saved model:", best_model_name)
print("File: best_solar_model.pkl")

# ==========================================
# 11. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)

# ==========================================
# 12. MODEL COMPARISON GRAPH
# ==========================================

plt.figure(figsize=(9, 6))

plt.bar(
    results_df["Model"],
    results_df["R2"]
)

plt.title("Machine Learning Model Comparison")
plt.xlabel("Model")
plt.ylabel("R² Score")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "model_comparison.png",
    dpi=300
)

plt.close()

print("\nCreated files:")
print("1. best_solar_model.pkl")
print("2. model_comparison_results.csv")
print("3. model_comparison.png")

print("\n" + "=" * 55)
print("MACHINE LEARNING COMPLETED SUCCESSFULLY")
print("=" * 55)