"""
testing_dataset_generator_module.py

Absolute File Path (example):
    /Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/testing_dataset_generator_module.py

PURPOSE:
  - Create flexible synthetic datasets for testing scenarios.
  - Produces sensor data (temperature, humidity, pressure, etc.).
  - Offers time-series generation, anomaly injection, and mixed feature sets.

NOTES:
  - You can adapt the sensor distributions, anomaly logic, or time-series parameters
    to more closely match your real-world use case.

References:
- Named "TestingDatasetGenerator" in conversation.
- Possibly used in unit tests, integration tests, or performance tests.
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union


class TestingDatasetGenerator:
    """
    A class to create diverse synthetic testing datasets for an energy/sensor project,
    including time-series sensor data, anomaly injection, and multi-feature sets.
    """

    def __init__(self, seed: int = 42):
        """
        :param seed: Random seed for reproducibility.
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def generate_sensor_dataset(
        self,
        num_samples: int = 100,
        anomaly_rate: float = 0.05,
        sensors: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate a DataFrame simulating multiple sensor readings with a given anomaly rate.
        
        :param num_samples: Number of sensor readings to generate.
        :param anomaly_rate: Fraction of readings flagged as anomalies (0.05 = 5%).
        :param sensors: List of sensor types to simulate (e.g. ["temperature", "humidity"]).
                        If None, defaults to a standard set: ["temperature", "humidity", "pressure"].
        
        Columns produced (example):
            - timestamp: A naive ascending timestamp.
            - sensor_<type> (e.g. sensor_temperature): numeric reading.
            - status: "Operational" or "Faulty".
            - sensor_type: The sensor's category (if you want separate rows for each type).
        """
        if sensors is None:
            sensors = ["temperature", "humidity", "pressure"]
        
        # Create a base timestamp to increment
        base_time = datetime.now()
        
        data_rows = []
        for i in range(num_samples):
            # For demonstration, each "reading" might combine multiple sensors at once
            # or be a single row per reading. Here, we do 1 row per sensor type.
            for s_type in sensors:
                # Synthetic reading distribution:
                if s_type == "temperature":
                    reading = 20 + 10 * np.random.rand()  # range ~ 20-30
                elif s_type == "humidity":
                    reading = 45 + 30 * np.random.rand()  # range ~ 45-75
                elif s_type == "pressure":
                    reading = 1000 + 50 * np.random.rand()  # ~ 1000-1050
                elif s_type == "vibration":
                    reading = 0.1 + 2.0 * np.random.rand()  # example range for vibrations
                else:
                    # default sensor reading
                    reading = 10.0 * np.random.rand()

                # Anomaly determination
                is_anomaly = np.random.rand() < anomaly_rate
                status = "Faulty" if is_anomaly else "Operational"

                # Possibly tweak reading to look anomalous
                if is_anomaly:
                    reading *= (1 + 2 * np.random.randn() * 0.05)  # small random multiplier

                row_time = base_time + timedelta(seconds=i)  # increment 1 second per sample for demonstration
                
                data_rows.append({
                    "timestamp": row_time.isoformat(),
                    "sensor_type": s_type,
                    f"sensor_{s_type}": reading,
                    "status": status
                })

        df = pd.DataFrame(data_rows)
        return df

    def generate_time_series_data(
        self,
        start_time: datetime,
        periods: int,
        freq_seconds: int = 60,
        sensor_types: Optional[List[str]] = None,
        anomaly_rate: float = 0.0
    ) -> pd.DataFrame:
        """
        Generate a multi-sensor time-series dataset, with readings at a fixed interval.
        
        :param start_time: Starting datetime for the series.
        :param periods: Number of intervals/readings.
        :param freq_seconds: Interval in seconds between readings.
        :param sensor_types: List of sensor types to simulate (e.g. ["temp", "humidity"]).
                             If None, defaults to ["temp", "humidity", "pressure"].
        :param anomaly_rate: Fraction of readings flagged as anomalies.
        :return: DataFrame with columns [timestamp, sensor_<type>, status].
        """
        if sensor_types is None:
            sensor_types = ["temp", "humidity", "pressure"]
        
        data_rows = []
        for i in range(periods):
            current_time = start_time + timedelta(seconds=i * freq_seconds)
            for s_type in sensor_types:
                # Basic distributions
                if s_type == "temp":
                    val = 15 + 15 * np.random.rand()  # ~ 15-30
                elif s_type == "humidity":
                    val = 40 + 30 * np.random.rand()  # ~ 40-70
                elif s_type == "pressure":
                    val = 990 + 20 * np.random.rand()  # ~ 990-1010
                else:
                    val = 10 * np.random.rand()

                is_anomaly = (np.random.rand() < anomaly_rate)
                status = "Faulty" if is_anomaly else "Operational"

                if is_anomaly:
                    # Example anomaly injection: 10% random multiplier
                    val *= (1 + 0.1 * np.random.randn())

                data_rows.append({
                    "timestamp": current_time.isoformat(),
                    f"sensor_{s_type}": val,
                    "status": status
                })
        df = pd.DataFrame(data_rows)
        return df

    def generate_mixed_feature_dataset(
        self,
        num_samples: int = 100,
        numeric_cols: int = 3,
        cat_cols: int = 2,
        cat_unique_values: int = 4
    ) -> pd.DataFrame:
        """
        Generate a DataFrame with numeric and categorical features for testing encoding or ML pipelines.
        
        :param num_samples: Number of rows to generate.
        :param numeric_cols: How many numeric columns.
        :param cat_cols: How many categorical columns.
        :param cat_unique_values: How many distinct categories each categorical column can have.
        :return: DataFrame with a mix of numeric and categorical columns.
        """
        # Numeric part
        numeric_data = np.random.randn(num_samples, numeric_cols)
        numeric_features = [f"num_col_{i}" for i in range(numeric_cols)]
        
        # Categorical part
        cat_data = []
        cat_features = [f"cat_col_{j}" for j in range(cat_cols)]
        for _ in range(num_samples):
            row = []
            for _ in range(cat_cols):
                category_id = np.random.randint(1, cat_unique_values + 1)
                row.append(f"cat_{category_id}")
            cat_data.append(row)
        
        df_numeric = pd.DataFrame(numeric_data, columns=numeric_features)
        df_categorical = pd.DataFrame(cat_data, columns=cat_features)
        
        return pd.concat([df_numeric, df_categorical], axis=1)

    def generate_custom_dataset(self, num_samples: int = 100) -> pd.DataFrame:
        """
        A placeholder for any specialized dataset. 
        You can adapt it for advanced scenarios or your custom requirements.
        """
        features = np.random.randn(num_samples, 5)
        columns = [f"feature_{i}" for i in range(1, 6)]
        df = pd.DataFrame(features, columns=columns)
        return df


if __name__ == "__main__":
    # Demonstration usage
    import logging
    logging.basicConfig(level=logging.INFO)

    generator = TestingDatasetGenerator()

    print("=== Basic Sensor Dataset with Anomalies ===")
    sensor_df = generator.generate_sensor_dataset(num_samples=10, anomaly_rate=0.3)
    print(sensor_df.head(), "\n")

    print("=== Time Series Data ===")
    start_dt = datetime.now()
    ts_df = generator.generate_time_series_data(start_dt, periods=5, freq_seconds=30, anomaly_rate=0.2)
    print(ts_df.head(), "\n")

    print("=== Mixed Feature Dataset ===")
    mixed_df = generator.generate_mixed_feature_dataset(num_samples=8, numeric_cols=3, cat_cols=2)
    print(mixed_df.head(), "\n")

    print("=== Custom Dataset ===")
    custom_df = generator.generate_custom_dataset(num_samples=6)
    print(custom_df.head(), "\n")