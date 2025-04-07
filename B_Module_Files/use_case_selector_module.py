#!/usr/bin/env python
"""
Module: use_case_selector_module.py

Purpose:
    This module dynamically detects the connected use case based on the sensor metadata
    provided (e.g., sensor names) and then returns the corresponding model file paths (or loaded models)
    from a registry. It also provides utility functions for populating UI elements like dropdowns
    with supported use cases.

Updates:
    - Added detection for the "drone" use case (4 DOF: 4 dofs and 4 velocity sensors).
    - Default unspecified use case is set to "warehouse".
    - Updated model registry to include the "drone" use case.
    - get_expected_sensors() and get_supported_use_cases() updated accordingly.
    
Usage:
    1. Call detect_use_case(sensor_list) with a list of sensor names.
    2. Call load_models_for_use_case(use_case_id) to get the model paths for that use case.
    3. Call get_expected_sensors(use_case_id) to get the expected sensor names.
    4. Call get_supported_use_cases() to get a list of all supported use case IDs.
    5. You can then load or do further logic as needed.

Example:
    sensors = ["dof_1", "dof_2", "dof_3", "dof_4",
               "vel_1", "vel_2", "vel_3", "vel_4"]
    use_case = detect_use_case(sensors)
    models = load_models_for_use_case(use_case)
    expected = get_expected_sensors(use_case)
    supported = get_supported_use_cases()
"""

import os

def detect_use_case(sensor_list):
    """
    Detects the use case based on provided sensor names.
    
    Args:
        sensor_list (list of str): e.g., ['dof_1','vel_1','imu',...]
    Returns:
        str: "drone", "6dof", "9dof", "warehouse", "urban", or "warehouse" (as default).
    """
    # Normalize sensor names to lowercase
    sensors = [s.lower() for s in sensor_list]
    dof_count = sum(1 for s in sensors if s.startswith("dof"))
    vel_count = sum(1 for s in sensors if s.startswith("vel"))
    
    # Check for drone: 4 DOFs and 4 velocity sensors
    if dof_count == 4 and vel_count == 4:
        return "drone"
    elif dof_count >= 9 and vel_count >= 9:
        return "9dof"
    elif dof_count >= 6 and vel_count >= 6 and (dof_count, vel_count) != (8, 8):
        return "6dof"
    elif dof_count == 8 and vel_count == 8:
        return "warehouse"
    elif dof_count == 5 and vel_count == 5:
        return "urban"
    else:
        # Default to warehouse if unknown
        return "warehouse"

def get_expected_sensors(use_case_id):
    """
    Returns a set of expected sensor names for the given use case.
    """
    if use_case_id == "drone":
        return {"dof_1", "dof_2", "dof_3", "dof_4",
                "vel_1", "vel_2", "vel_3", "vel_4",
                "energy_efficiency", "energy_saved"}
    elif use_case_id == "6dof":
        return {"dof_1", "dof_2", "dof_3", "dof_4", "dof_5", "dof_6",
                "vel_1", "vel_2", "vel_3", "vel_4", "vel_5", "vel_6",
                "energy_efficiency", "energy_saved"}
    elif use_case_id == "9dof":
        return {"dof_1", "dof_2", "dof_3", "dof_4", "dof_5", "dof_6", "dof_7", "dof_8", "dof_9",
                "vel_1", "vel_2", "vel_3", "vel_4", "vel_5", "vel_6", "vel_7", "vel_8", "vel_9",
                "energy_efficiency", "energy_saved"}
    elif use_case_id == "warehouse":
        return {"dof_1", "dof_2", "dof_3", "dof_4", "dof_5", "dof_6", "dof_7", "dof_8",
                "vel_1", "vel_2", "vel_3", "vel_4", "vel_5", "vel_6", "vel_7", "vel_8",
                "energy_efficiency", "energy_saved"}
    elif use_case_id == "urban":
        return {"dof_1", "dof_2", "dof_3", "dof_4", "dof_5",
                "vel_1", "vel_2", "vel_3", "vel_4", "vel_5",
                "energy_efficiency", "energy_saved"}
    else:
        return set()

def load_models_for_use_case(use_case_id):
    """
    Returns a dict mapping model types to file paths for the given use case.
    If unknown, defaults to the warehouse use case.
    """
    MODEL_DIR = os.path.expanduser(
        os.getenv("MODEL_DIR", "~/energy_optimization_project/C_Model_Files")
    )
    
    model_registry = {
        "drone": {
            "anomaly": os.path.join(MODEL_DIR, "anomaly_detection_model", "anomaly_detection_model_drone.joblib"),
            "energy_efficiency": os.path.join(MODEL_DIR, "energy_efficiency_optimization_model", "energy_efficiency_drone.joblib"),
            "predictive_maintenance": os.path.join(MODEL_DIR, "predictive_maintenance_model", "predictive_maintenance_drone.joblib"),
            "rl": os.path.join(MODEL_DIR, "reinforcement_learning_model", "ppo_drone_model.zip")
        },
        "6dof": {
            "anomaly": os.path.join(MODEL_DIR, "anomaly_detection_model", "anomaly_detection_model_6dof.joblib"),
            "energy_efficiency": os.path.join(MODEL_DIR, "energy_efficiency_optimization_model", "energy_efficiency_6dof.joblib"),
            "predictive_maintenance": os.path.join(MODEL_DIR, "predictive_maintenance_model", "predictive_maintenance_6dof.joblib"),
            "rl": os.path.join(MODEL_DIR, "reinforcement_learning_model", "ppo_genesis_6dof_model.zip")
        },
        "9dof": {
            "anomaly": os.path.join(MODEL_DIR, "anomaly_detection_model", "anomaly_detection_model_9dof.joblib"),
            "energy_efficiency": os.path.join(MODEL_DIR, "energy_efficiency_optimization_model", "energy_efficiency_9dof.joblib"),
            "predictive_maintenance": os.path.join(MODEL_DIR, "predictive_maintenance_model", "predictive_maintenance_9dof.joblib"),
            "rl": os.path.join(MODEL_DIR, "reinforcement_learning_model", "ppo_genesis_9dof_model.zip")
        },
        "warehouse": {
            "anomaly": os.path.join(MODEL_DIR, "anomaly_detection_model", "anomaly_detection_model_warehouse.joblib"),
            "energy_efficiency": os.path.join(MODEL_DIR, "energy_efficiency_optimization_model", "energy_efficiency_warehouse.joblib"),
            "predictive_maintenance": os.path.join(MODEL_DIR, "predictive_maintenance_model", "predictive_maintenance_warehouse.joblib"),
            "rl": os.path.join(MODEL_DIR, "reinforcement_learning_model", "ppo_warehouse_model.zip")
        },
        "urban": {
            "anomaly": os.path.join(MODEL_DIR, "anomaly_detection_model", "anomaly_detection_model_urban.joblib"),
            "energy_efficiency": os.path.join(MODEL_DIR, "energy_efficiency_optimization_model", "energy_efficiency_urban.joblib"),
            "predictive_maintenance": os.path.join(MODEL_DIR, "predictive_maintenance_model", "predictive_maintenance_urban.joblib"),
            "rl": os.path.join(MODEL_DIR, "reinforcement_learning_model", "ppo_urban_model.zip")
        },
        "smart_house": {
            # placeholder for future use
        },
        "autonomous_vehicle": {
            # placeholder for future use
        },
        "unknown": {}
    }

    # Default to warehouse if use_case_id is unknown
    return model_registry.get(use_case_id, model_registry["warehouse"])

def get_supported_use_cases():
    """
    Returns a sorted list of all supported use case IDs, excluding 'unknown'.
    This can be used to populate a dropdown in the UI.
    """
    supported = ["drone", "6dof", "9dof", "warehouse", "urban", "smart_house", "autonomous_vehicle"]
    return sorted([uc for uc in supported if uc != "unknown"])

# Optional convenience class for easier use
class UseCaseSelector:
    def __init__(self, sensor_list):
        self.sensor_list = sensor_list
        self.use_case = detect_use_case(sensor_list)
        self.models = load_models_for_use_case(self.use_case)
    
    def get_use_case(self):
        return self.use_case

    def get_model_paths(self):
        return self.models

if __name__ == "__main__":
    sample_sensors = [
        "dof_1", "dof_2", "dof_3", "dof_4",  # For drone, we expect 4 DOFs
        "vel_1", "vel_2", "vel_3", "vel_4"
    ]
    use_case = detect_use_case(sample_sensors)
    print("Detected Use Case:", use_case)
    print("Model Paths:", load_models_for_use_case(use_case))
    print("Supported Use Cases:", get_supported_use_cases())