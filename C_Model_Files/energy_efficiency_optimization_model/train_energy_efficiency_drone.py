# file name: train_energy_efficiency_drone.py
# file path: /home/ec2-user/energy_optimization_project/C_Model_Files/energy_efficiency_optimization_model/train_energy_efficiency_drone.py

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

def main():
    # ---------------------------------------------------------------------
    # 1. Paths & Configuration
    # ---------------------------------------------------------------------
    CSV_PATH = "/home/ec2-user/energy_optimization_project/D_Dataset_Files/energy_efficiency_optimization_dataset/gen_energy_efficiency_data_drone.csv"
    MODEL_OUT = "/home/ec2-user/energy_optimization_project/C_Model_Files/energy_efficiency_optimization_model/energy_efficiency_drone.joblib"
    SCALER_OUT = "/home/ec2-user/energy_optimization_project/C_Model_Files/energy_efficiency_optimization_model/scaler_energy_efficiency_drone.joblib"

    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV not found at {CSV_PATH}")
        sys.exit(1)

    print(f"[INFO] Loading dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"[INFO] Data shape: {df.shape}")

    # Expected columns for the drone dataset:
    # Features: dof_1..dof_4, vel_1..vel_4, avg_acceleration, var_acceleration, avg_force, var_force, avg_velocity, var_velocity
    # Target: energy_usage
    expected_cols = (
        [f"dof_{i}" for i in range(1, 5)] +
        [f"vel_{i}" for i in range(1, 5)] +
        ["avg_acceleration", "var_acceleration", "avg_force", "var_force", "avg_velocity", "var_velocity", "energy_usage"]
    )
    for c in expected_cols:
        if c not in df.columns:
            print(f"[ERROR] Missing column '{c}' in dataset.")
            sys.exit(1)

    # ---------------------------------------------------------------------
    # 2. Separate Features & Target; Drop Rows with Missing Target
    # ---------------------------------------------------------------------
    feature_cols = [col for col in df.columns if col != "energy_usage"]
    X = df[feature_cols].values
    y = df["energy_usage"].values

    print(f"[INFO] Feature columns: {feature_cols}")
    print(f"[INFO] X.shape = {X.shape}, y.shape = {y.shape}")

    initial_rows = df.shape[0]
    df = df.dropna(subset=["energy_usage"])
    dropped_rows = initial_rows - df.shape[0]
    print(f"[INFO] Dropped {dropped_rows} rows due to missing energy_usage.")

    X = df[feature_cols].values
    y = df["energy_usage"].values
    print(f"[INFO] New X.shape = {X.shape}, y.shape = {y.shape}")

    # ---------------------------------------------------------------------
    # 3. Transform the Target Values
    # ---------------------------------------------------------------------
    y = np.log1p(y)
    print("[INFO] Applied log1p transformation to target values.")

    finite_mask = np.isfinite(y)
    if not np.all(finite_mask):
        num_nonfinite = np.sum(~finite_mask)
        print(f"[INFO] Dropping {num_nonfinite} rows with non-finite target values after log1p transformation.")
        X = X[finite_mask]
        y = y[finite_mask]
        print(f"[INFO] After filtering: X.shape = {X.shape}, y.shape = {y.shape}")

    # ---------------------------------------------------------------------
    # 4. Impute Missing Feature Values and Scale Features
    # ---------------------------------------------------------------------
    imputer = SimpleImputer(strategy="mean")
    X[np.isinf(X)] = np.nan
    X = imputer.fit_transform(X)

    if os.path.exists(SCALER_OUT):
        print(f"[INFO] Loading existing scaler from {SCALER_OUT}...")
        scaler = joblib.load(SCALER_OUT)
        if scaler.mean_.shape[0] != X.shape[1]:
            print(f"[WARNING] Existing scaler expects {scaler.mean_.shape[0]} features but input has {X.shape[1]}. Creating a new scaler...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
    else:
        print("[INFO] Creating a new StandardScaler...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    # ---------------------------------------------------------------------
    # 5. TRAIN/VAL SPLIT
    # ---------------------------------------------------------------------
    print("[INFO] Splitting data into train/validation sets (80/20).")
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(f"[INFO] Train set: {X_train.shape}, Validation set: {X_val.shape}")

    # ---------------------------------------------------------------------
    # 6. Build Pipeline with GradientBoostingRegressor (increased capacity)
    # ---------------------------------------------------------------------
    pipe = Pipeline([
        ('gbr', GradientBoostingRegressor(random_state=42, n_iter_no_change=10, tol=0.0001))
    ])
    param_dist = {
        "gbr__n_estimators": randint(50, 300),
        "gbr__learning_rate": uniform(0.01, 0.2),
        "gbr__max_depth": randint(1, 3),
        "gbr__subsample": uniform(0.7, 0.3),
        "gbr__min_samples_split": randint(2, 10),
        "gbr__min_samples_leaf": randint(1, 5)
    }
    
    print("[INFO] Running RandomizedSearchCV for GradientBoostingRegressor…")
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=20,
        scoring="neg_mean_squared_error",
        cv=5,
        random_state=42,
        verbose=1,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    print("[INFO] Best Params:", search.best_params_)
    print("[INFO] Best CV Score (neg_MSE):", round(search.best_score_, 4))
    
    best_pipe = search.best_estimator_
    y_val_pred = best_pipe.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    print(f"[INFO] Validation MSE (log scale): {val_mse:.4f}")
    print(f"[INFO] Validation R² (log scale):  {val_r2:.4f}")
    
    final_pipe = Pipeline([
        ('gbr', GradientBoostingRegressor(
            n_estimators=search.best_params_["gbr__n_estimators"],
            learning_rate=search.best_params_["gbr__learning_rate"],
            max_depth=search.best_params_["gbr__max_depth"],
            subsample=search.best_params_["gbr__subsample"],
            min_samples_split=search.best_params_["gbr__min_samples_split"],
            min_samples_leaf=search.best_params_["gbr__min_samples_leaf"],
            random_state=42,
            n_iter_no_change=10,
            tol=0.0001
        ))
    ])
    final_pipe.fit(X_scaled, y)
    y_full_pred = final_pipe.predict(X_scaled)
    full_mse = mean_squared_error(y, y_full_pred)
    full_r2 = r2_score(y, y_full_pred)
    print("[INFO] Full Dataset Evaluation (log scale):")
    print(f"  MSE: {full_mse:.4f}")
    print(f"  R² : {full_r2:.4f}")
    
    # ---------------------------------------------------------------------
    # 7. Save the Final Model (Pipeline) & Scaler
    # ---------------------------------------------------------------------
    print(f"[INFO] Saving final model pipeline to {MODEL_OUT}...")
    joblib.dump(final_pipe, MODEL_OUT)
    print("[INFO] Model saved successfully.")
    
    if not os.path.exists(SCALER_OUT) or scaler.mean_.shape[0] != X.shape[1]:
        print(f"[INFO] Saving new scaler to {SCALER_OUT}...")
        joblib.dump(scaler, SCALER_OUT)
    else:
        print("[INFO] Using existing scaler; no changes made.")
    
    print("[INFO] Training script complete.")

if __name__ == "__main__":
    main()