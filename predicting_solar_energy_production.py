import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# ------------------------- Load Data -------------------------
df = pd.read_csv("Solar Energy.csv")

df.head()

print("Shape:", df.shape)

print(df.info())
# ------------------------- Preprocessing -------------------------
# Convert dates
df['Data Through Date'] = pd.to_datetime(df['Data Through Date'], errors='coerce')
df['Interconnection Date'] = pd.to_datetime(df['Interconnection Date'], errors='coerce')

# Drop mostly-missing column
df.drop(columns=['Energy Storage System Size (kWac)'], inplace=True, errors='ignore')

# Fill missing numeric values
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

# Fill missing categorical values
for col in df.select_dtypes(include='object').columns:
    df[col].fillna('Unknown', inplace=True)

#------------ EDA --------------
# Distribution of target
plt.figure(figsize=(8, 5))
sns.histplot(df['Estimated Annual PV Energy Production (kWh)'], bins=40, kde=True)
plt.title(f"Distribution of Estimated Annual PV Energy Production (kWh)")
plt.show()

# Correlation heatmap for numeric features
plt.figure(figsize=(8, 4))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm")
plt.title("Numeric Correlation Heatmap")
plt.show()

# Drop identifier/non-informative columns
drop_cols = ['Data Through Date', 'Project ID', 'Division', 'Substation', 'Circuit ID']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)



#-------- Feature Engineering --------

df['Interconnect_Year'] = df['Interconnection Date'].dt.year
df['Interconnect_Month'] = df['Interconnection Date'].dt.month
df['County_Avg_Size'] = df.groupby('County')['Estimated PV System Size (kWdc)'].transform('mean')

# ------------------------- Features & Target -------------------------
# Drop leaky features
leaky_features = ["Estimated PV System Size (kWdc)", "PV System Size (kWac)"]
X = df.drop(columns=leaky_features + ["Estimated Annual PV Energy Production (kWh)"], errors='ignore')
y = df["Estimated Annual PV Energy Production (kWh)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Identify categorical and numeric columns
categorical_cols = [col for col in ['Utility','City/Town','County','Developer','Metering Method'] if col in X_train.columns]
numeric_cols = [col for col in ['Zip', 'Number of Projects', 'Interconnect_Year', 'Interconnect_Month', 'County_Avg_Size'] if col in X_train.columns]


# ------------------------- Handle Skewness -------------------------
# Identify skewed numeric features
skewed_features = X_train[numeric_cols].skew()
skewed_features = skewed_features[abs(skewed_features) > 0.5].index.tolist()
print("Skewed Features:", skewed_features)

# Use log1p transform
log_transformer = FunctionTransformer(np.log1p, validate=True)

# Preprocessor
preprocessor = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

# ----------- Random Forest model -----------

# Random Forest pipeline
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        n_jobs=-1,
        random_state=42
    ))
])

# Train
rf_pipeline.fit(X_train, y_train)

# Predict & Evaluate
y_pred_rf = rf_pipeline.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Results:")
print(f"MAE  : {mae_rf:,.2f}")
print(f"RMSE : {rmse_rf:,.2f}")
print(f"R²   : {r2_rf:.4f}")

# ------------------------- XGBoost Model -------------------------
from xgboost import XGBRegressor

# XGBoost pipeline
xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    ))
])

# Train
xgb_pipeline.fit(X_train, y_train)

# Predict & Evaluate
y_pred_xgb = xgb_pipeline.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print("XGBoost Results:")
print(f"MAE  : {mae_xgb:,.2f}")
print(f"RMSE : {rmse_xgb:,.2f}")
print(f"R²   : {r2_xgb:.4f}")

from sklearn.model_selection import RandomizedSearchCV

# ------------------------- Random Forest Hyperparameter Tuning -------------------------
rf_param_grid = {
    'regressor__n_estimators': [200, 300, 400],
    'regressor__max_depth': [10, 15, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__max_features': ['sqrt', 'log2']
}

rf_search = RandomizedSearchCV(
    rf_pipeline,                 # existing Random Forest pipeline
    rf_param_grid,
    n_iter=10,                   # number of random combinations
    cv=3,                        # 3-fold cross-validation
    scoring='r2',                # metric to optimize
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# Fit hyperparameter search
rf_search.fit(X_train, y_train)

# Evaluate
y_pred_rf = rf_search.predict(X_test)
print("Random Forest with Hyperparameter Tuning:")
print("Best Parameters:", rf_search.best_params_)
print(f"MAE  : {mean_absolute_error(y_test, y_pred_rf):,.2f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred_rf)):,.2f}")
print(f"R²   : {r2_score(y_test, y_pred_rf):.4f}")


# ------------------------- XGBoost Hyperparameter Tuning -------------------------
xgb_param_grid = {
    'regressor__n_estimators': [300, 400, 500],
    'regressor__max_depth': [4, 6, 8],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__subsample': [0.7, 0.8, 0.9],
    'regressor__colsample_bytree': [0.7, 0.8, 0.9]
}

xgb_search = RandomizedSearchCV(
    xgb_pipeline,                # existing XGBoost pipeline
    xgb_param_grid,
    n_iter=10,                    # number of random combinations
    cv=3,
    scoring='r2',
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# Fit hyperparameter search
xgb_search.fit(X_train, y_train)

# Evaluate
y_pred_xgb = xgb_search.predict(X_test)
print("XGBoost with Hyperparameter Tuning:")
print("Best Parameters:", xgb_search.best_params_)
print(f"MAE  : {mean_absolute_error(y_test, y_pred_xgb):,.2f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred_xgb)):,.2f}")
print(f"R²   : {r2_score(y_test, y_pred_xgb):.4f}")


# ------------------------- LightGBM Regressor -------------------------
from lightgbm import LGBMRegressor

lgbm_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ))
])

# Train
lgbm_pipeline.fit(X_train, y_train)

# Predict
y_pred_lgbm = lgbm_pipeline.predict(X_test)

# Evaluate
mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
r2_lgbm = r2_score(y_test, y_pred_lgbm)

print("\n🌿 LightGBM Model Results:")
print(f"MAE  : {mae_lgbm:,.2f}")
print(f"RMSE : {rmse_lgbm:,.2f}")
print(f"R²   : {r2_lgbm:.4f}")

# ------------------------- LightGBM (Hyperparameter Tuning) -------------------------
lgbm_param_grid = {
    'regressor__n_estimators': [300, 400, 500],
    'regressor__max_depth': [-1, 10, 20, 30],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__num_leaves': [31, 50, 70],
    'regressor__subsample': [0.7, 0.8, 0.9],
    'regressor__colsample_bytree': [0.7, 0.8, 0.9]
}

lgbm_search = RandomizedSearchCV(
    lgbm_pipeline,
    lgbm_param_grid,
    n_iter=10,
    cv=3,
    scoring='r2',
    verbose=2,
    n_jobs=-1,
    random_state=42
)

lgbm_search.fit(X_train, y_train)

# Predict
y_pred_lgbm = lgbm_search.predict(X_test)

# Evaluate
mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
r2_lgbm = r2_score(y_test, y_pred_lgbm)

print("\n🌿 LightGBM with Hyperparameter Tuning Results:")
print("Best Parameters:", lgbm_search.best_params_)
print(f"MAE  : {mae_lgbm:,.2f}")
print(f"RMSE : {rmse_lgbm:,.2f}")
print(f"R²   : {r2_lgbm:.4f}")

# ------------------------- Feature Importance for Random Forest-------------------------
best_rf = rf_search.best_estimator_.named_steps['regressor']
preprocessor = rf_search.best_estimator_.named_steps['preprocessor']

# Get feature importances
feature_names = preprocessor.get_feature_names_out()
importances = best_rf.feature_importances_

# Create DataFrame
feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_imp = feat_imp.sort_values(by="Importance", ascending=False).head(15)

# Plot
plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=feat_imp)
plt.title("Top 15 Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()

# ------------------------- Feature Importance for XGBoost-------------------------
# Get the trained XGBoost regressor from the pipeline
best_xgb = xgb_search.best_estimator_.named_steps['regressor']  # or xgb_pipeline if no tuning
preprocessor = xgb_search.best_estimator_.named_steps['preprocessor']

# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Get feature importances
importances = best_xgb.feature_importances_

# Create a DataFrame for easier plotting
feat_imp_xgb = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(15)

# Plot top 15 features
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_imp_xgb)
plt.title("Top 15 Feature Importances (XGBoost)")
plt.tight_layout()
plt.show()

# ------------------------- Feature Importance for LightGBM-------------------------
# Get the trained LightGBM regressor from the pipeline
best_lgbm = lgbm_search.best_estimator_.named_steps['regressor']  # or lgbm_pipeline if no tuning
preprocessor = lgbm_search.best_estimator_.named_steps['preprocessor']

# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Get feature importances
importances = best_lgbm.feature_importances_

# Create a DataFrame for plotting
feat_imp_lgbm = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(15)

# Plot top 15 features
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_imp_lgbm)
plt.title("Top 15 Feature Importances (LightGBM)")
plt.tight_layout()
plt.show()

# Save XGBoost pipeline
joblib.dump(xgb_pipeline, "xgboost_model.joblib")
