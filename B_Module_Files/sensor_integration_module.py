"""
sensor_integration_module.py

Absolute File Path:
    /Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/sensor_integration_module.py

PURPOSE:
  - A module to handle sensor data ingestion, anomaly checks, and
    integration with other modules (anomaly detection, predictive
    maintenance, energy efficiency, sensor fusion, real-time decisions, etc.).
  - Reads/writes sensor data to a PostgreSQL database.
  - Optionally uses preloaded scalers/models for each advanced analytics process.

USAGE:
  # Example instantiation:
    from sensor_integration_module import SensorIntegrationModule

    db_conf = {
        "user": "db_user",
        "password": "db_pass",
        "host": "127.0.0.1",
        "port": 5432,
        "database": "energy_optimization_db"
    }

    # Suppose you have loaded these from joblib or another source:
    anomaly_scaler = ...
    anomaly_model = ...
    ee_scaler = ...
    ee_model = ...
    pm_scaler = ...
    pm_model = ...
    fusion_model = ...            # e.g. SensorFusionModel instance
    decision_model = ...          # e.g. RealTimeDecisionModel instance

    sim = SensorIntegrationModule(
        db_config=db_conf,
        anomaly_scaler=anomaly_scaler,
        anomaly_model=anomaly_model,
        ee_scaler=ee_scaler,
        ee_model=ee_model,
        pm_scaler=pm_scaler,
        pm_model=pm_model,
        fusion_model=fusion_model,
        decision_model=decision_model
    )

    # Store data
    sim.store_sensor_data([
        {"name": "TemperatureSensor", "output": 23.5, "status": "Operational"},
        {"name": "VibrationSensor", "output": 5.2, "status": "Operational"}
    ])

    # Retrieve recent data
    recent_data = sim.get_recent_sensor_data(limit=5)

    # Run advanced checks:
    import numpy as np
    new_values = np.array([[23.5, 5.2, 0.9]])  # Just an example
    anomaly_preds = sim.check_anomalies(new_values)
    pm_result = sim.check_predictive_maintenance(new_values)
    ee_result = sim.check_energy_efficiency(new_values)
    fusion_result = sim.check_sensor_fusion(new_values)         # e.g. classification
    decision_result = sim.check_real_time_decision(new_values)  # e.g. action recommendations

REFERENCES:
  - Uses psycopg2 for PostgreSQL
  - Numpy for data arrays
  - Logging for module-level logs
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor


class SensorIntegrationModule:
    """
    Manages sensor data operations: 
      1) Ingest/store data in a PostgreSQL DB.
      2) Perform anomaly detection checks (if scaler/model are set).
      3) Provide hooks for energy efficiency (EE) or predictive maintenance (PM).
      4) Provide hooks for sensor fusion (fusion_model).
      5) Provide hooks for real-time decision-making (decision_model).
    """

    def __init__(
        self,
        db_config: Dict[str, Any],
        anomaly_scaler=None,
        anomaly_model=None,
        ee_scaler=None,
        ee_model=None,
        pm_scaler=None,
        pm_model=None,
        fusion_model=None,
        decision_model=None,
        logger: Optional[logging.Logger] = None
    ):
        """
        :param db_config: PostgreSQL database configuration (keys: user, password, host, port, database).
        :param anomaly_scaler: Scaler to preprocess data for anomaly detection.
        :param anomaly_model: Model to classify or detect anomalies.
        :param ee_scaler: Scaler for energy efficiency tasks (optional).
        :param ee_model: Model for energy efficiency tasks (optional).
        :param pm_scaler: Scaler for predictive maintenance tasks (optional).
        :param pm_model: Model for predictive maintenance tasks (optional).
        :param fusion_model: A sensor fusion model instance, e.g. SensorFusionModel.
        :param decision_model: A real-time decision-making model instance.
        :param logger: Optional logger instance. If None, creates a default module logger.
        """
        self.db_config = db_config
        self.anomaly_scaler = anomaly_scaler
        self.anomaly_model = anomaly_model
        self.ee_scaler = ee_scaler
        self.ee_model = ee_model
        self.pm_scaler = pm_scaler
        self.pm_model = pm_model
        self.fusion_model = fusion_model
        self.decision_model = decision_model

        if logger is None:
            self.logger = logging.getLogger("SensorIntegrationModule")
            if not self.logger.handlers:
                # Setup minimal console handler if no handlers
                console = logging.StreamHandler()
                console.setLevel(logging.INFO)
                self.logger.addHandler(console)
                self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger

        self.logger.info("SensorIntegrationModule initialized.")

    @contextmanager
    def get_db_connection(self):
        """
        Context manager that yields a psycopg2 connection with RealDictCursor.
        Closes the connection automatically afterward.
        """
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
            yield conn
        except Exception as e:
            self.logger.error(f"Database connection error: {e}", exc_info=True)
            raise
        finally:
            if conn:
                conn.close()

    def store_sensor_data(self, sensor_data: List[Dict[str, Any]]) -> None:
        """
        Store multiple sensor data entries into the 'sensor_data' table.

        Table schema example (PostgreSQL):
            CREATE TABLE sensor_data (
                id SERIAL PRIMARY KEY,
                sensor_name VARCHAR(100),
                sensor_output DOUBLE PRECISION,
                status VARCHAR(50),
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );

        :param sensor_data: A list of dict items, e.g.:
               [
                 {"name": "TempSensor", "output": 23.5, "status": "Operational"},
                 {"name": "HumiditySensor", "output": 60.0, "status": "Operational"}
               ]
        """
        if not sensor_data:
            self.logger.warning("No sensor data provided to store.")
            return

        with self.get_db_connection() as conn:
            with conn.cursor() as cursor:
                for sd in sensor_data:
                    name = sd.get("name")
                    output = sd.get("output")
                    status = sd.get("status", "Unknown")
                    cursor.execute(
                        """
                        INSERT INTO sensor_data 
                            (sensor_name, sensor_output, status, timestamp)
                        VALUES (%s, %s, %s, NOW())
                        """,
                        (name, output, status),
                    )
            conn.commit()
        self.logger.info(f"Stored {len(sensor_data)} new sensor data entries.")

    def get_recent_sensor_data(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent sensor data from 'sensor_data' table for other tasks.
        :param limit: Number of records to fetch (ordered DESC by timestamp).
        :return: A list of dictionary rows: 
                 [
                    {"sensor_name": "...", "sensor_output": ..., "status": "...", "timestamp": ...},
                    ...
                 ]
        """
        rows = []
        with self.get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT sensor_name, sensor_output, status, timestamp
                    FROM sensor_data
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()

        self.logger.info(f"Retrieved {len(rows)} sensor data entries from DB.")
        return rows

    # -------------------------------------------------------------------------
    #                      ANOMALY DETECTION CHECK
    # -------------------------------------------------------------------------
    def check_anomalies(self, new_data: np.ndarray):
        """
        Use anomaly_scaler and anomaly_model to detect anomalies.
        :param new_data: A 2D array of shape (num_samples, num_features).
        :return: The model's output (e.g., anomaly labels or scores).
        """
        if (self.anomaly_scaler is None) or (self.anomaly_model is None):
            self.logger.warning("Anomaly scaler/model not set. Skipping anomaly detection.")
            return None

        scaled_data = self.anomaly_scaler.transform(new_data)
        predictions = self.anomaly_model.predict(scaled_data)
        self.logger.info(f"Anomaly check completed. Predictions shape: {predictions.shape}")
        return predictions

    # -------------------------------------------------------------------------
    #                      ENERGY EFFICIENCY CHECK
    # -------------------------------------------------------------------------
    def check_energy_efficiency(self, new_data: np.ndarray):
        """
        Use ee_scaler and ee_model to predict energy efficiency outcomes.
        :param new_data: 2D array of sensor readings or features.
        :return: The predicted energy efficiency metrics/scores.
        """
        if (self.ee_scaler is None) or (self.ee_model is None):
            self.logger.warning("Energy Efficiency scaler/model not loaded.")
            return None

        scaled_data = self.ee_scaler.transform(new_data)
        ee_preds = self.ee_model.predict(scaled_data)
        self.logger.info("Energy efficiency prediction completed.")
        return ee_preds

    # -------------------------------------------------------------------------
    #                   PREDICTIVE MAINTENANCE CHECK
    # -------------------------------------------------------------------------
    def check_predictive_maintenance(self, new_data: np.ndarray):
        """
        Use pm_scaler and pm_model for predictive maintenance analysis.
        :param new_data: 2D array of sensor data.
        :return: PM predictions or recommended maintenance intervals.
        """
        if (self.pm_scaler is None) or (self.pm_model is None):
            self.logger.warning("Predictive Maintenance scaler/model not loaded.")
            return None

        scaled_data = self.pm_scaler.transform(new_data)
        pm_preds = self.pm_model.predict(scaled_data)
        self.logger.info("Predictive maintenance check completed.")
        return pm_preds

    # -------------------------------------------------------------------------
    #                        SENSOR FUSION CHECK
    # -------------------------------------------------------------------------
    def check_sensor_fusion(self, new_data: np.ndarray):
        """
        Use the sensor_fusion_model (if loaded) to generate multi-sensor classification or results.
        :param new_data: 2D array of fused sensor features for the model input.
        :return: The sensor fusion model's predictions.
        """
        if self.fusion_model is None:
            self.logger.warning("Sensor Fusion model not loaded.")
            return None

        # Example usage: if the model expects scaled input, ensure you do so
        # or if the model is a Keras-based 'fusion_model'
        preds = self.fusion_model.predict({"main_sensor_input": new_data})
        self.logger.info(f"Sensor Fusion check completed. Predictions shape: {preds.shape}")
        return preds

    # -------------------------------------------------------------------------
    #                   REAL-TIME DECISION MAKING CHECK
    # -------------------------------------------------------------------------
    def check_real_time_decision(self, new_data: np.ndarray):
        """
        Use the real_time_decision_making_model to produce real-time decisions.
        :param new_data: 2D array of features (scaled or unscaled depending on model).
        :return: The decision model's output or recommended actions.
        """
        if self.decision_model is None:
            self.logger.warning("Real-Time Decision model not loaded.")
            return None

        # Example usage: if the model is e.g. a Keras or SB3 policy
        # For a Keras model:
        #    preds = self.decision_model.predict(new_data)
        # For a SB3 policy:
        #    actions, _ = self.decision_model.predict(new_data)
        # For simplicity, let's assume Keras-like:
        preds = self.decision_model.predict(new_data)
        self.logger.info("Real-time decision check completed.")
        return preds


if __name__ == "__main__":
    # Quick example usage
    logging.basicConfig(level=logging.INFO)

    # Example DB config (using your AWS RDS credentials)
    db_conf = {
        "user": "rvsalinas",
        "password": "AIsensor123!",
        "host": "vestavio.cpq6m8ka4a1l.us-east-2.rds.amazonaws.com",
        "port": 5432,
        "database": "energy_optimization_db"
    }

    # Typically, you'd load or pass your models. We'll pass None for now.
    sim = SensorIntegrationModule(db_conf)

    # Example data insertion
    sim.store_sensor_data([
        {"name": "TempSensor", "output": 23.5, "status": "Operational"},
        {"name": "HumiditySensor", "output": 60.0, "status": "Operational"}
    ])

    # Fetch recent data
    recent_data = sim.get_recent_sensor_data(limit=5)
    print("Recent sensor data:")
    for rd in recent_data:
        print(rd)

    # Example advanced checks (if models were loaded):
    # new_values = np.array([[23.5, 60.0]])
    # anomaly_preds = sim.check_anomalies(new_values)
    # ee_preds = sim.check_energy_efficiency(new_values)
    # pm_preds = sim.check_predictive_maintenance(new_values)
    # fusion_preds = sim.check_sensor_fusion(new_values)
    # decision_preds = sim.check_real_time_decision(new_values)