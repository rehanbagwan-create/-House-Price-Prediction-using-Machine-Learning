# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 06:54:03 2026

@author: Samir
"""

# =========================================================
# END-TO-END MACHINE LEARNING PROJECT – REAL ESTATE PRICES
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("RealEstate_Housing_Dataset.csv")
print(df)

print("-------------------------------------------------------------")

print(df.head())

print("-------------------------------------------------------------")

print(df.info())

print("-------------------------------------------------------------")

print(df.describe())

# ---------------------------------------------------------
# 3. DATA VISUALIZATION
# ---------------------------------------------------------

df.hist(figsize=(12,8))
plt.suptitle("Feature Distributions")
plt.show()

print("-------------------------------------------------------------")

plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# # ---------------------------------------------------------
# # 4. PREPARE DATA
# # ---------------------------------------------------------
print(df.columns)

print("-------------------------------------------------------------")

x = df.drop("MedHouseVal", axis=1)
print(x)

print("-------------------------------------------------------------")
y = df["MedHouseVal"]
print(y)

print("-------------------------------------------------------------")

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("\nData Preprocessing Completed.")

# ---------------------------------------------------------
# 5. TRAIN MULTIPLE MODELS
# ---------------------------------------------------------

# ---- Linear Regression ----
lr = LinearRegression()
lr.fit(x_train_scaled, y_train)
lr_pred = lr.predict(x_test_scaled)

print("-------------------------------------------------------------")

# ---- Decision Tree ----
dt = DecisionTreeRegressor(random_state=42)
dt.fit(x_train_scaled, y_train)
dt_pred = dt.predict(x_test_scaled)

print("-------------------------------------------------------------")

# ---- Random Forest ----
rf = RandomForestRegressor(random_state=42)
rf.fit(x_train_scaled, y_train)
rf_pred = rf.predict(x_test_scaled)
print("\nModel Training Completed.")

print("-------------------------------------------------------------")

# ---------------------------------------------------------
# 6. MODEL EVALUATION FUNCTION
# ---------------------------------------------------------
def evaluate_model(name, y_test, predictions):
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print(f"\n{name} Performance:")
    print("RMSE:", rmse)
    print("R2 Score:", r2)

evaluate_model("Linear Regression", y_test, lr_pred)

print("-------------------------------------------------------------")

evaluate_model("Decision Tree", y_test, dt_pred)

print("-------------------------------------------------------------")

evaluate_model("Random Forest", y_test, rf_pred)

print("-------------------------------------------------------------")

# ---------------------------------------------------------
# 7. CROSS-VALIDATION (Random Forest)
# ---------------------------------------------------------
scores = cross_val_score(
    rf, x_train_scaled, y_train,
    scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)
print("\nRandom Forest Cross-Validation RMSE:")
print("Scores:", rmse_scores)
print("Mean:", rmse_scores.mean())

# ---------------------------------------------------------
# 8. HYPERPARAMETER TUNING (GRID SEARCH)
# ---------------------------------------------------------
param_grid = {
    'n_estimators': [50, 100],
    'max_features': [4, 6, 8],
    'max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(x_train_scaled, y_train)
best_model = grid_search.best_estimator_

print("\nBest Parameters:", grid_search.best_params_)

# ---------------------------------------------------------
# 9. FINAL EVALUATION
# ---------------------------------------------------------
final_pred = best_model.predict(x_test_scaled)
evaluate_model("Tuned Random Forest", y_test, final_pred)

# ---------------------------------------------------------
# 10. VISUALIZE PREDICTIONS
# ---------------------------------------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, final_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()],
          [y_test.min(), y_test.max()], 'r--')
plt.show()

# ---------------------------------------------------------
# 11. FEATURE IMPORTANCE
# ---------------------------------------------------------
importances = best_model.feature_importances_
features = x.columns

feat_imp = pd.Series(importances, index=features)
feat_imp.sort_values().plot(kind='barh', figsize=(8,6))
plt.title("Feature Importance")
plt.show()

print("\nProject Completed Successfully!")


# Generate production level model using pikl library.

