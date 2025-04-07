# ------------------------------------------------------------------------------
# File: system_decision_cycle.py
# Absolute File Path:
# /Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/system_decision_cycle.py
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd

def system_decision_cycle(
    conn,
    anomaly_detection_model=None,
    rl_model=None,
    ee_model=None,
    ee_scaler=None,
    pm_model=None,
    pm_transformer=None,
    logger=None,
    limit_rows=10
):
    """
    Orchestrates the calls to:
      1) 9-DOF anomaly detection (18 features: dof_1..dof_9 + vel_1..vel_9)
      2) RL model (9DOF PPO)
      3) Energy Efficiency model
      4) Predictive Maintenance model

    Parameters
    ----------
    conn : psycopg2 connection
        An active DB connection.
    anomaly_detection_model : model
        A loaded scikit-learn or joblib-based 9-DOF anomaly model (expects 18 features).
    rl_model : stable_baselines3 PPO model
        RL model for 9DOF control.
    ee_model : scikit-learn regressor
        Energy efficiency model.
    ee_scaler : scikit-learn scaler
        Scaler for energy efficiency features.
    pm_model : scikit-learn classifier
        Predictive maintenance model.
    pm_transformer : scikit-learn ColumnTransformer or scaler
        Transformer for predictive maintenance features.
    logger : logging.Logger
        Logger to log info, warnings, or errors.
    limit_rows : int
        How many rows to fetch from the DB (dof data, energy data, etc.).

    Returns
    -------
    dict
        A dictionary containing the predictions/actions from each model:
        {
          "anomaly_preds": [...],  # 0/1 labels
          "rl_actions": [...],     # 9-element arrays from RL
          "energy_preds": [...],
          "pm_preds": [...]
        }
    """
    results = {
        "anomaly_preds": [],
        "rl_actions": [],
        "energy_preds": [],
        "pm_preds": []
    }

    with conn.cursor() as cursor:
        # ------------------------------------------------------------------
        # 1) 9-DOF anomaly detection + RL
        # ------------------------------------------------------------------
        try:
            cursor.execute("""
                SELECT dof_1, dof_2, dof_3, dof_4, dof_5, dof_6, dof_7, dof_8, dof_9,
                       vel_1, vel_2, vel_3, vel_4, vel_5, vel_6, vel_7, vel_8, vel_9
                FROM sensor_data
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit_rows,))
            dof_vel_rows = cursor.fetchall()

            X_18_list = []
            row_indices = []
            for idx, row in enumerate(dof_vel_rows):
                features_18 = []
                skip_row = False
                for col_name in [
                    'dof_1','dof_2','dof_3','dof_4','dof_5','dof_6','dof_7','dof_8','dof_9',
                    'vel_1','vel_2','vel_3','vel_4','vel_5','vel_6','vel_7','vel_8','vel_9'
                ]:
                    val = row[col_name]
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        skip_row = True
                        break
                    features_18.append(val)
                if not skip_row and len(features_18) == 18:
                    X_18_list.append(features_18)
                    row_indices.append(idx)

            if X_18_list and anomaly_detection_model and rl_model:
                X_18_arr = np.array(X_18_list, dtype=np.float32)
                # a) Anomaly
                anomaly_preds = anomaly_detection_model.predict(X_18_arr)
                results["anomaly_preds"] = anomaly_preds.tolist()

                # b) RL
                rl_actions = []
                for arr in X_18_arr:
                    obs = arr.reshape(1, -1)
                    action, _ = rl_model.predict(obs, deterministic=True)
                    rl_actions.append(action.flatten().tolist())
                results["rl_actions"] = rl_actions

                if logger:
                    logger.info(f"system_decision_cycle => anomaly_preds={anomaly_preds.tolist()}, RL_actions={rl_actions}")
            elif logger:
                logger.info("system_decision_cycle => No valid 18D rows or no anomaly/rl_model loaded.")
        except Exception as e:
            if logger:
                logger.error(f"Error in anomaly+RL block: {e}")

        # ------------------------------------------------------------------
        # 2) Energy Efficiency
        # ------------------------------------------------------------------
        try:
            if ee_model and ee_scaler:
                EE_FEATURES = ['x1','x2','x3','x4','x5','x6','x7','x8']
                cursor.execute(f"""
                    SELECT {', '.join(EE_FEATURES)}
                    FROM sensor_data
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (limit_rows,))
                ee_rows = cursor.fetchall()

                ee_arr = []
                for r in ee_rows:
                    vals = [r[f] for f in EE_FEATURES]
                    if any(val is None or (isinstance(val, float) and np.isnan(val)) for val in vals):
                        continue
                    ee_arr.append(vals)

                if ee_arr:
                    X_ee = np.array(ee_arr, dtype=np.float32)
                    if X_ee.shape[1] == 8:
                        X_ee_scaled = ee_scaler.transform(X_ee)
                        ee_preds = ee_model.predict(X_ee_scaled)
                        results["energy_preds"] = ee_preds.tolist()
                        if logger:
                            logger.info(f"system_decision_cycle => energy_preds={ee_preds}")
        except Exception as e:
            if logger:
                logger.error(f"Error in energy efficiency block: {e}")

        # ------------------------------------------------------------------
        # 3) Predictive Maintenance
        # ------------------------------------------------------------------
        try:
            if pm_model and pm_transformer:
                PM_FEATURES = [
                    "Type", "Air temperature [K]", "Process temperature [K]",
                    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
                ]
                cursor.execute(f"""
                    SELECT "{PM_FEATURES[0]}", "{PM_FEATURES[1]}", "{PM_FEATURES[2]}",
                           "{PM_FEATURES[3]}", "{PM_FEATURES[4]}", "{PM_FEATURES[5]}"
                    FROM sensor_data
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (limit_rows,))
                pm_rows = cursor.fetchall()

                if pm_rows:
                    pm_arr = []
                    for row in pm_rows:
                        row_vals = [row[feat] for feat in PM_FEATURES]
                        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in row_vals):
                            continue
                        pm_arr.append(row_vals)

                    if pm_arr:
                        pm_df = pd.DataFrame(pm_arr, columns=PM_FEATURES)
                        pm_features_transformed = pm_transformer.transform(pm_df)
                        pm_preds = pm_model.predict(pm_features_transformed)
                        results["pm_preds"] = pm_preds.tolist()
                        if logger:
                            logger.info(f"system_decision_cycle => pm_preds={pm_preds}")
        except Exception as e:
            if logger:
                logger.error(f"Error in predictive maintenance block: {e}")

    return results