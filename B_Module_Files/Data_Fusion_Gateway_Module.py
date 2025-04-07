"""
Data_Fusion_Gateway_Module.py

PURPOSE:
  - Provide a central gateway for fusing data from multiple sources (sensor data, external APIs, etc.).
  - Standardize or transform the fused data according to a defined schema or transformation logic.
  - Optionally perform grouping/aggregation for downstream reporting or analytics.
  - Optionally build a fixed-length feature array from the fused data for ML/RL usage.
  - (New) Optionally check threshold-based anomalies on a DataFrame.

USAGE:
  python "Data_Fusion_Gateway_Module.py"
"""

import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


class DataFusionGateway:
    """
    A class that orchestrates data ingestion from multiple sources
    and transforms them into a unified DataFrame for downstream tasks.
    """

    def __init__(self, schema: Optional[Dict[str, str]] = None):
        """
        :param schema: Optional dictionary describing expected columns and their dtypes.
                       Example: {"name": "str", "status": "str", "output": "float"}
                       If not provided, the script will do best-effort transformations.
        """
        self.logger = logging.getLogger("DataFusionGateway")
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        # Avoid duplicating handlers if re-initialized
        if not self.logger.handlers:
            self.logger.addHandler(console_handler)

        self.schema = schema
        self.logger.info("DataFusionGateway initialized with schema=%s", schema)

    def ingest_data(
        self,
        sensor_data: List[Dict[str, Any]],
        external_data: List[Dict[str, Any]],
        sensor_source_label: str = "sensor",
        external_source_label: str = "external"
    ) -> pd.DataFrame:
        """
        Ingest two sets of data (sensor & external), annotate their source,
        and return a single fused DataFrame.

        :param sensor_data: list of dicts representing sensor data rows
        :param external_data: list of dicts representing external data rows
        :param sensor_source_label: label added to sensor_data rows in the 'source' column
        :param external_source_label: label added to external_data rows in the 'source' column
        :return: fused DataFrame containing both sensor and external data
        """
        if not sensor_data and not external_data:
            self.logger.warning("No data provided (sensor_data and external_data are both empty).")
            return pd.DataFrame()

        sensor_df = pd.DataFrame(sensor_data)
        sensor_df["source"] = sensor_source_label

        ext_df = pd.DataFrame(external_data)
        ext_df["source"] = external_source_label

        fused_df = pd.concat([sensor_df, ext_df], ignore_index=True)
        self.logger.info("Fused DataFrame shape: %s", fused_df.shape)
        return fused_df

    def apply_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attempt to cast the DataFrame’s columns according to self.schema
        (if provided). Logs any mismatches or missing columns.

        :param df: DataFrame to cast/validate
        :return: DataFrame after schema application
        """
        if not self.schema:
            self.logger.info("No schema provided; skipping schema enforcement.")
            return df

        for col, dtype_str in self.schema.items():
            if col not in df.columns:
                self.logger.warning("Column '%s' not found in DataFrame, skipping cast.", col)
                continue

            try:
                if dtype_str.lower() == "float":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif dtype_str.lower() == "int":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                elif dtype_str.lower() == "str":
                    df[col] = df[col].astype(str)
                else:
                    self.logger.warning("Unsupported dtype '%s' for column '%s'. Skipping.", dtype_str, col)
            except Exception as e:
                self.logger.error("Error casting column '%s' to %s: %s", col, dtype_str, e, exc_info=True)
        return df

    def fill_missing_values(self, df: pd.DataFrame, fill_values: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Fill missing values in the DataFrame. If fill_values is provided,
        it should be a dict mapping column names to fill values.
        For numeric columns not in fill_values, we can fill with median as an example.

        :param df: DataFrame with possible missing values
        :param fill_values: dictionary like {"status": "Unknown", "output": 0.0}
        :return: DataFrame with missing values filled
        """
        fill_values = fill_values or {}
        # Fill each column from fill_values if present
        for col, val in fill_values.items():
            if col in df.columns:
                df = df.fillna({col: val})

        # For numeric columns not in fill_values, fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col not in fill_values:
                median_val = df[col].median(skipna=True)
                df = df.fillna({col: median_val})

        return df

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Placeholder method for any additional transformations after
        schema application and missing-value fill. Could do normalization,
        encoding, etc.
        """
        # Example: convert "status" to categorical
        if "status" in df.columns:
            df["status"] = df["status"].astype("category")

        # Log final shape
        self.logger.info("Transformed DataFrame shape: %s", df.shape)
        return df

    def aggregate_data(self, df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        """
        Example aggregator: group by group_col (like "status")
        and count the number of rows in each group.

        :param df: DataFrame to aggregate
        :param group_col: column name to group by
        :return: aggregated DataFrame with counts
        """
        if group_col not in df.columns:
            self.logger.warning("Cannot aggregate by '%s' - column not in DataFrame", group_col)
            return df

        agg_df = df.groupby(group_col, observed=False).size().reset_index(name="count")
        self.logger.info("Aggregated data by '%s':\n%s", group_col, agg_df)
        return agg_df

    def build_feature_array(
        self,
        df: pd.DataFrame,
        expected_cols: List[str],
        fill_value: float = 0.0,
        single_row: bool = True
    ) -> np.ndarray:
        """
        Create a fixed-length array from the DataFrame for ML/RL usage,
        matching the columns in 'expected_cols'.

        - If a column is missing or has no data, we use 'fill_value'.
        - By default, returns a single 1D array (if 'single_row' is True),
          taking the first row that appears in the DataFrame.
        - If 'single_row' is False, returns a 2D array with shape (num_rows, len(expected_cols)).

        :param df: DataFrame that has (or may not have) the columns in expected_cols.
        :param expected_cols: The list of columns (sensors/features) we want in our final array.
        :param fill_value: Value to use if a column is missing or empty.
        :param single_row: Whether to take the first row or produce an array for all rows.
        :return: A NumPy array of shape (len(expected_cols),) or (num_rows, len(expected_cols)).
        """
        if df.empty:
            self.logger.warning("DataFrame is empty; returning a single row of fill_value if single_row=True.")
            if single_row:
                return np.full(shape=(len(expected_cols),), fill_value=fill_value)
            else:
                return np.empty((0, len(expected_cols)))  # an empty 2D array

        # If single_row=True, we take the first row of df:
        if single_row:
            row_dict = df.iloc[0].to_dict()
            array_data = []
            for col in expected_cols:
                val = row_dict.get(col, fill_value)
                if pd.isnull(val):
                    val = fill_value
                array_data.append(float(val))
            return np.array(array_data, dtype=np.float32)
        else:
            # We produce one row in the array per DataFrame row
            all_rows = []
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                row_array = []
                for col in expected_cols:
                    val = row_dict.get(col, fill_value)
                    if pd.isnull(val):
                        val = fill_value
                    row_array.append(float(val))
                all_rows.append(row_array)
            return np.array(all_rows, dtype=np.float32)

    def check_thresholds(self, df: pd.DataFrame, thresholds: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Optional convenience method to label rows with out-of-range DOF/sensor values
        according to a threshold dictionary. Adds a column 'anomaly' = 1 if any
        sensor is out of range, else 0.

        :param df: DataFrame that includes columns corresponding to the keys in 'thresholds'.
        :param thresholds: dict like {
            "dof_1": {"low": -0.1, "high": 0.2},
            "dof_2": {"low": -0.05, "high": 1.0}, ...
        }
        :return: The same DataFrame but with an added 'anomaly' column (0 or 1).
        """
        df = df.copy(deep=True)
        if "anomaly" not in df.columns:
            df["anomaly"] = 0

        for idx, row in df.iterrows():
            for sensor, rng in thresholds.items():
                if sensor not in df.columns:
                    self.logger.warning(f"Threshold key '{sensor}' not found in DataFrame columns.")
                    continue

                val = row[sensor]
                if val < rng["low"] or val > rng["high"]:
                    df.at[idx, "anomaly"] = 1
                    break  # No need to check more sensors once flagged

        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    example_schema = {
        "name": "str",
        "status": "str",
        "output": "float",
        "some_field": "int"
    }
    gateway = DataFusionGateway(schema=example_schema)

    sensor_data_example = [
        {"name": "TempSensor", "status": "Operational", "output": 22.5},
        {"name": "HumiditySensor", "status": "Operational", "output": 65.0},
        {"name": "PressureSensor", "status": None, "output": 1012.7}
    ]
    external_data_example = [
        {"name": "ExternalData1", "some_field": 123},
        {"name": "ExternalData2", "some_field": 999, "output": 99.5}
    ]

    # 1. Ingest/fuse
    fused_df = gateway.ingest_data(sensor_data_example, external_data_example)

    # 2. Apply schema
    fused_df = gateway.apply_schema(fused_df)

    # 3. Fill missing
    fused_df = gateway.fill_missing_values(fused_df, fill_values={"status": "Unknown"})

    # 4. Additional transform
    fused_df = gateway.transform_data(fused_df)

    # 5. Aggregate example
    agg_df = gateway.aggregate_data(fused_df, group_col="status")

    print("\nFused + Transformed DataFrame:\n", fused_df)
    print("\nAggregated by 'status':\n", agg_df)

    # 6. Build feature arrays
    feature_cols = ["name", "status", "output", "some_field", "random_col"]
    arr_single = gateway.build_feature_array(fused_df, expected_cols=feature_cols, fill_value=0.0, single_row=True)
    arr_multi = gateway.build_feature_array(fused_df, expected_cols=feature_cols, fill_value=0.0, single_row=False)
    print("\nSingle-row feature array:\n", arr_single)
    print("\nMulti-row feature array:\n", arr_multi)

    # 7. Optional threshold check (example)
    thresholds_example = {
        "output": {"low": 20.0, "high": 80.0},
        "some_field": {"low": 100, "high": 900}
    }
    fused_thresholded_df = gateway.check_thresholds(fused_df, thresholds_example)
    print("\nDataFrame after threshold checks:\n", fused_thresholded_df)