"""
data_preprocessing_module.py

A flexible module for data preprocessing tasks such as:
  - Missing value handling
  - Outlier capping/trimming (optional)
  - Numeric scaling (standard, minmax, etc.)
  - Categorical encoding (one-hot, etc.)
  - Generating a final unified DataFrame or NumPy array

New in this version:
  - A 'special_fallback' missing-value strategy that adds binary indicators for missingness
    and fills numeric/categorical data in a way that the model can learn from missing data.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError


class DataPreprocessor:
    """
    A flexible class for data preprocessing. It handles:
      1) Missing values (simple fill, drop, or special_fallback).
      2) Outlier capping (optional).
      3) Numeric scaling (standard or minmax).
      4) Categorical encoding (one-hot).
      5) Producing a final unified DataFrame or NumPy array.

    Usage Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> raw_data = {
        ...   "temp": [22.5, 25.3, 19.0, np.nan],
        ...   "humidity": [60, 65, 55, 72],
        ...   "status": ["ok", "ok", "faulty", None],
        ...   "pressure": [1012, 1008, 999, 1050]
        ... }
        >>> df = pd.DataFrame(raw_data)
        >>>
        >>> numeric_cols = ["temp", "humidity", "pressure"]
        >>> cat_cols = ["status"]
        >>>
        >>> preprocessor = DataPreprocessor(
        ...     numeric_strategy="standard",
        ...     cat_strategy="one_hot",
        ...     missing_value_strategy="special_fallback",
        ...     outlier_capping_enabled=True,
        ...     outlier_capping_quantiles=(0.05, 0.95)
        ... )
        >>>
        >>> processed_df = preprocessor.fit_transform(df, numeric_cols, cat_cols)
        >>> print(processed_df)
    """

    # Optionally, define which sensors your anomaly model cares about.
    ANOMALY_SENSORS = [
        "dof_1", "dof_2", "dof_3", "dof_4",
        "dof_5", "dof_6", "dof_7", "dof_8", "dof_9"
    ]

    def __init__(
        self,
        numeric_strategy: str = "standard",
        cat_strategy: str = "one_hot",
        missing_value_strategy: str = "fill_zero",
        outlier_capping_enabled: bool = False,
        outlier_capping_quantiles: tuple = (0.01, 0.99)
    ):
        """
        :param numeric_strategy: "standard" or "minmax" for numeric scaling strategy.
        :param cat_strategy: "one_hot" or "none" for categorical encoding strategy.
        :param missing_value_strategy:
            - "fill_zero": fill numeric NaNs with 0, cat with "Unknown"
            - "mean": fill numeric NaNs with column mean
            - "median": fill numeric NaNs with column median
            - "drop": drop rows containing NaNs
            - "special_fallback": Create <col>_missing indicator; fill numeric with sentinel -9999.0,
              fill cat with "Unknown".
        :param outlier_capping_enabled: whether to apply outlier capping.
        :param outlier_capping_quantiles: quantile range (low_q, high_q) to cap outliers.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)

        self.numeric_strategy = numeric_strategy
        self.cat_strategy = cat_strategy
        self.missing_value_strategy = missing_value_strategy
        self.outlier_capping_enabled = outlier_capping_enabled
        self.outlier_capping_quantiles = outlier_capping_quantiles

        # Will be set after fit
        self.numeric_pipeline = None
        self.cat_pipeline = None
        self.column_transformer = None

        self.logger.info(
            f"Initialized {self.__class__.__name__} with numeric_strategy={numeric_strategy}, "
            f"cat_strategy={cat_strategy}, missing_value_strategy={missing_value_strategy}, "
            f"outlier_capping_enabled={outlier_capping_enabled}."
        )

    def _add_missing_indicators(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        cat_cols: List[str],
        numeric_sentinel: float = -9999.0
    ) -> pd.DataFrame:
        """
        For each column with missing values, add a <col>_missing column (1/0).
        Fill numeric columns with numeric_sentinel, cat with 'Unknown'.
        """
        df = df.copy(deep=True)

        # Numeric columns
        for col in numeric_cols:
            if df[col].isna().any():
                missing_col_name = f"{col}_missing"
                df[missing_col_name] = df[col].isna().astype(int)
                # Fill with sentinel
                df[col] = df[col].fillna(numeric_sentinel)

        # Categorical columns
        for col in cat_cols:
            if df[col].isna().any():
                missing_col_name = f"{col}_missing"
                df[missing_col_name] = df[col].isna().astype(int)
                df[col] = df[col].fillna("Unknown")

        return df

    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        cat_cols: List[str]
    ) -> pd.DataFrame:
        """
        Handle missing values according to the chosen strategy.
        """
        df = df.copy(deep=True)

        if self.missing_value_strategy == "drop":
            self.logger.info("Dropping rows with any missing values in numeric or categorical columns.")
            df.dropna(subset=numeric_cols + cat_cols, how='any', inplace=True)

        elif self.missing_value_strategy == "fill_zero":
            # For numeric => fill with 0
            self.logger.info("Filling numeric NaNs with 0 (fill_zero).")
            df[numeric_cols] = df[numeric_cols].fillna(0)
            # For cat => fill with Unknown
            self.logger.info("Filling categorical NaNs with 'Unknown'.")
            df[cat_cols] = df[cat_cols].fillna("Unknown")

        elif self.missing_value_strategy in ("mean", "median"):
            # Numeric
            for col in numeric_cols:
                if self.missing_value_strategy == "mean":
                    fill_val = df[col].mean(skipna=True)
                    self.logger.info(f"Filling numeric col '{col}' NaNs with MEAN={fill_val:.3f}")
                else:
                    fill_val = df[col].median(skipna=True)
                    self.logger.info(f"Filling numeric col '{col}' NaNs with MEDIAN={fill_val:.3f}")
                df[col] = df[col].fillna(fill_val)
            # Categorical => 'Unknown'
            self.logger.info("Filling categorical NaNs with 'Unknown'.")
            df[cat_cols] = df[cat_cols].fillna("Unknown")

        elif self.missing_value_strategy == "special_fallback":
            # Add missing indicators for each col that has NaNs
            self.logger.info("Using special_fallback strategy: adding <col>_missing indicators.")
            df = self._add_missing_indicators(df, numeric_cols, cat_cols, numeric_sentinel=-9999.0)

        else:
            self.logger.warning(f"Unknown missing_value_strategy: {self.missing_value_strategy}")
            # By default, do nothing (pass-through) -> risk of NaNs left in data.

        return df

    def _cap_outliers(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """
        Cap outliers for numeric columns based on given quantiles.
        """
        if not numeric_cols:
            return df

        low_q, high_q = self.outlier_capping_quantiles
        self.logger.info(f"Applying outlier capping with quantiles {self.outlier_capping_quantiles}.")
        df = df.copy(deep=True)
        for col in numeric_cols:
            lower_bound = df[col].quantile(low_q)
            upper_bound = df[col].quantile(high_q)
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        return df

    def _build_numeric_pipeline(self) -> Pipeline:
        """
        Build a scikit-learn pipeline for numeric transformations:
         - scaling (StandardScaler or MinMaxScaler)
        """
        if self.numeric_strategy == "standard":
            scaler = StandardScaler()
        elif self.numeric_strategy == "minmax":
            scaler = MinMaxScaler()
        else:
            self.logger.warning(f"Unsupported numeric strategy: {self.numeric_strategy}. Defaulting to StandardScaler.")
            scaler = StandardScaler()

        pipeline = Pipeline([("scaler", scaler)])
        return pipeline

    def _build_cat_pipeline(self) -> Optional[Pipeline]:
        """
        Build a pipeline for categorical transformations (like one-hot).
        Return None if cat_strategy='none'.
        """
        if self.cat_strategy == "none":
            return None
        elif self.cat_strategy == "one_hot":
            encoder = OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            )
            pipeline = Pipeline([("ohe", encoder)])
            return pipeline
        else:
            self.logger.warning(f"Unsupported cat strategy: {self.cat_strategy}. No transformation.")
            return None

    def fit_transform(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        cat_cols: List[str]
    ) -> pd.DataFrame:
        """
        Fit the internal transformers using the provided DataFrame,
        then transform the data. Returns a new DataFrame with the combined transformations.

        :param df: The input DataFrame containing numeric and/or categorical columns
        :param numeric_cols: list of column names for numeric data
        :param cat_cols: list of column names for categorical data
        :return: A DataFrame with transformed numeric/cat features.
        """
        self.logger.info("Starting fit_transform...")

        # 1) Handle missing values
        df_handled = self._handle_missing_values(df, numeric_cols, cat_cols)

        # 2) Outlier capping if enabled
        if self.outlier_capping_enabled:
            df_handled = self._cap_outliers(df_handled, numeric_cols)

        # 3) Build numeric and cat pipelines
        self.numeric_pipeline = self._build_numeric_pipeline()
        self.cat_pipeline = self._build_cat_pipeline()

        # 4) Identify final numeric & cat columns to transform
        #    We need to see if "special_fallback" created new numeric or cat indicator columns
        new_columns = list(df_handled.columns)
        # The newly added <col>_missing columns are always numeric (0/1).
        # If they're in numeric_cols, they'll get scaled. If that's not desired,
        # you can skip them or treat them as cat. We'll treat them as numeric for convenience.
        # We'll detect them by name endswith "_missing" for columns that were originally numeric or cat.
        auto_missing_indicators = [
            col for col in new_columns
            if col.endswith("_missing") and col not in numeric_cols + cat_cols
        ]
        # We'll add them to numeric_cols so they're scaled (assuming 0/1 scale is harmless in standard scaling).
        final_numeric_cols = numeric_cols + auto_missing_indicators
        final_cat_cols = cat_cols  # no change

        # 5) Build a ColumnTransformer
        transformers = []
        if final_numeric_cols:
            transformers.append(("num", self.numeric_pipeline, final_numeric_cols))
        if final_cat_cols and self.cat_pipeline:
            transformers.append(("cat", self.cat_pipeline, final_cat_cols))

        self.column_transformer = ColumnTransformer(transformers=transformers, remainder='drop')

        # 6) Fit and transform
        self.logger.info("Fitting ColumnTransformer on data.")
        transformed_data = self.column_transformer.fit_transform(df_handled)

        # 7) Reconstruct feature names
        numeric_feature_names = []
        cat_feature_names = []

        if final_numeric_cols:
            numeric_feature_names = [f"{col}_scaled" for col in final_numeric_cols]

        if final_cat_cols and self.cat_pipeline:
            cat_pipeline_fitted = self.column_transformer.named_transformers_.get("cat")
            if cat_pipeline_fitted is not None:
                cat_encoder_fitted = cat_pipeline_fitted.named_steps["ohe"]
                try:
                    cat_feature_names = list(cat_encoder_fitted.get_feature_names_out(final_cat_cols))
                except NotFittedError:
                    self.logger.warning("Cat encoder not fitted yet. Feature names unknown.")
                    cat_feature_names = [f"{col}_enc" for col in final_cat_cols]
            else:
                cat_feature_names = []

        all_feature_names = numeric_feature_names + cat_feature_names
        output_df = pd.DataFrame(transformed_data, columns=all_feature_names, index=df_handled.index)

        self.logger.info(f"Finished fit_transform. Output shape={output_df.shape}")
        return output_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the already-fitted pipelines.
        :param df: input DataFrame
        :return: transformed DataFrame
        """
        if not self.column_transformer:
            self.logger.error("ColumnTransformer not fitted yet. Call fit_transform first.")
            return pd.DataFrame()

        self.logger.info("Applying transform on new data.")

        df_handled = df.copy(deep=True)

        # The pipeline might have added <col>_missing columns. Let's replicate that logic now:
        # We check which columns were used in the original numeric/cat config:
        used_numeric_cols = []
        used_cat_cols = []
        for name, trans, cols in self.column_transformer.transformers_:
            if name == "num":
                used_numeric_cols = cols
            elif name == "cat":
                used_cat_cols = cols

        # Identify which were originally numeric vs cat (strip the "_scaled")
        # but for now we'll just apply the same missing-value logic:
        if self.missing_value_strategy == "drop":
            df_handled.dropna(subset=used_numeric_cols + used_cat_cols, how='any', inplace=True)
        elif self.missing_value_strategy == "fill_zero":
            df_handled[used_numeric_cols] = df_handled[used_numeric_cols].fillna(0)
            df_handled[used_cat_cols] = df_handled[used_cat_cols].fillna("Unknown")
        elif self.missing_value_strategy in ("mean", "median"):
            self.logger.warning("Mean/median imputation in transform requires storing train stats. Skipping.")
            df_handled[used_cat_cols] = df_handled[used_cat_cols].fillna("Unknown")
        elif self.missing_value_strategy == "special_fallback":
            # Use the same approach: add <col>_missing, fill numeric with sentinel, cat with Unknown
            # But we don't have a direct record of which numeric/cat columns might appear.
            # We'll assume the columns we used plus any new columns in df:
            # So let's break it down:
            # 1) for numeric => fill with -9999 and add <col>_missing if col is in used_numeric_cols
            # 2) for cat => fill with 'Unknown' and <col>_missing if col in used_cat_cols
            for col in used_numeric_cols:
                orig_col = col.replace("_scaled", "")  # if needed
                if orig_col in df_handled.columns:
                    if df_handled[orig_col].isna().any():
                        missing_col_name = f"{orig_col}_missing"
                        df_handled[missing_col_name] = df_handled[orig_col].isna().astype(int)
                        df_handled[orig_col] = df_handled[orig_col].fillna(-9999.0)
            for col in used_cat_cols:
                if col in df_handled.columns:
                    if df_handled[col].isna().any():
                        missing_col_name = f"{col}_missing"
                        df_handled[missing_col_name] = df_handled[col].isna().astype(int)
                        df_handled[col] = df_handled[col].fillna("Unknown")

        # Outlier capping in transform is not strictly consistent unless we have stored quantiles
        if self.outlier_capping_enabled:
            self.logger.warning("Outlier capping in transform requires train quantiles. Not applied here.")

        # Actually transform via the pre-fitted ColumnTransformer
        transformed = self.column_transformer.transform(df_handled)

        # Rebuild column names
        numeric_feature_names = []
        cat_feature_names = []
        # from the pre-fitted pipeline, these are basically the same as in fit_transform
        num_trans = self.column_transformer.named_transformers_.get("num")
        if num_trans:
            # The original numeric columns were turned into X_scaled
            # We can retrieve them from "used_numeric_cols"
            numeric_feature_names = [f"{col}_scaled" for col in used_numeric_cols]

        cat_trans = self.column_transformer.named_transformers_.get("cat")
        if cat_trans is not None:
            try:
                cat_encoder_fitted = cat_trans.named_steps["ohe"]
                # We need the cat cols from the pipeline
                cat_cols_in_pipeline = cat_trans.named_steps["ohe"].feature_names_in_
                cat_feature_names = list(cat_encoder_fitted.get_feature_names_out(cat_cols_in_pipeline))
            except NotFittedError:
                self.logger.warning("Cat encoder not fitted yet. Feature names unknown.")
                cat_feature_names = [f"{col}_enc" for col in used_cat_cols]

        all_feature_names = numeric_feature_names + cat_feature_names
        output_df = pd.DataFrame(transformed, columns=all_feature_names, index=df_handled.index)

        self.logger.info(f"Transform complete. Output shape={output_df.shape}")
        return output_df

    def scale_tabular_data(
        self,
        df: pd.DataFrame,
        method: str = "standard"
    ) -> Union[pd.DataFrame, None]:
        """
        A convenience method to quickly scale numeric data in a DataFrame with one step
        (without categorical columns or advanced pipeline).

        :param df: The DataFrame containing numeric data only
        :param method: "standard" or "minmax" for scaling
        :return: scaled DataFrame with the same columns
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided to scale_tabular_data.")
            return None

        self.logger.info(f"scale_tabular_data called with method={method}.")
        data_copy = df.copy(deep=True)
        columns = data_copy.columns

        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            self.logger.warning(f"Unsupported method {method}. Defaulting to 'standard'.")
            scaler = StandardScaler()

        try:
            scaled_values = scaler.fit_transform(data_copy.values)
            scaled_df = pd.DataFrame(scaled_values, columns=columns, index=data_copy.index)
            return scaled_df
        except Exception as e:
            self.logger.error(f"Error in scale_tabular_data: {e}", exc_info=True)
            return None

    def align_dict_features_for_inference(
        self, 
        sensor_dict: Dict[str, Any],
        required_features: List[str],
        fill_value: float = 0.0
    ) -> np.ndarray:
        """
        Create a fixed-length numpy array in the correct order for inference.
        - Any missing features from sensor_dict are filled with fill_value (default 0.0).
        - Any extra keys in sensor_dict are ignored.

        :param sensor_dict: e.g. {"temp": 22.0, "humidity": 60.2, ...}
        :param required_features: canonical list of features your model requires 
                                  in the correct order.
        :param fill_value: value to place if a feature is missing
        :return: np.array with shape (len(required_features),)
        """
        final_arr = np.full(len(required_features), fill_value, dtype=np.float32)
        for i, feat in enumerate(required_features):
            if feat in sensor_dict:
                val = sensor_dict[feat]
                try:
                    final_arr[i] = float(val)
                except ValueError:
                    # If it can't be cast to float, leave the fill_value
                    pass
        return final_arr

    def align_array_features_for_inference(
        self,
        obs_array: np.ndarray,
        required_features: int,
        fill_value: float = 0.0
    ) -> np.ndarray:
        """
        Align a raw observation array to the required number of features
        by either truncating or padding with fill_value.

        :param obs_array: shape (batch_size, current_num_features)
        :param required_features: integer representing the desired feature length
        :param fill_value: value to use for padding if current_num_features < required_features
        :return: shape (batch_size, required_features)
        """
        aligned = obs_array.copy()
        if aligned.ndim == 1:
            aligned = aligned.reshape(1, -1)

        current_features = aligned.shape[1]

        if current_features < required_features:
            pad_width = required_features - current_features
            aligned = np.hstack([
                aligned,
                np.full((aligned.shape[0], pad_width), fill_value, dtype=aligned.dtype)
            ])
        elif current_features > required_features:
            aligned = aligned[:, :required_features]

        return aligned

    def align_features_for_inference(
        self,
        obs_array: np.ndarray,
        required_features: int,
        fill_value: float = 0.0
    ) -> np.ndarray:
        """
        Wrapper method to align features for inference.
        It calls align_array_features_for_inference.
        
        :param obs_array: np.ndarray of observations.
        :param required_features: integer representing the desired feature length.
        :param fill_value: value to pad with if needed.
        :return: np.ndarray with shape (batch_size, required_features)
        """
        return self.align_array_features_for_inference(obs_array, required_features, fill_value)

    def build_anomaly_features(self, sensor_dict: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Specialized helper method to build a 9-feature array for anomaly detection.
        - Uses the ANOMALY_SENSORS list as the required sensor whitelist.
        - Fills missing features with np.nan, then returns None if any sensor is missing.
          (If you prefer to keep partial data, you could fill with 0.0 or special_fallback.)
        """
        arr = self.align_dict_features_for_inference(
            sensor_dict,
            required_features=self.ANOMALY_SENSORS,
            fill_value=np.nan
        )
        # If we want to strictly require all 9 DOFs for anomaly detection,
        # we can return None if any are missing:
        if np.isnan(arr).any():
            return None
        return arr


if __name__ == "__main__":
    import pandas as pd
    logging.basicConfig(level=logging.INFO)

    sample_data = {
        "temp": [22.5, 25.3, 19.0, np.nan],
        "humidity": [60, 65, np.nan, 72],
        "status": ["ok", "ok", "faulty", None],
        "pressure": [1012, 1008, 999, np.nan]
    }

    df_sample = pd.DataFrame(sample_data)
    numeric_cols_demo = ["temp", "humidity", "pressure"]
    cat_cols_demo = ["status"]

    preprocessor = DataPreprocessor(
        numeric_strategy="standard",
        cat_strategy="one_hot",
        missing_value_strategy="special_fallback",  # new approach
        outlier_capping_enabled=True,
        outlier_capping_quantiles=(0.05, 0.95)
    )

    transformed_df = preprocessor.fit_transform(df_sample, numeric_cols_demo, cat_cols_demo)
    print("\nTransformed DataFrame:\n", transformed_df)

    # Example sensor dict usage
    sensor_dict = {"temp": 23.0, "humidity": None, "extra_sensor": 999}
    feats_needed = ["temp", "humidity", "pressure"]
    arr_from_dict = preprocessor.align_dict_features_for_inference(sensor_dict, feats_needed, fill_value=-9999.0)
    print("\nDictionary-based alignment:\n", arr_from_dict)

    # Example for anomaly
    dof_example = {
        "dof_1": 0.01,
        "dof_2": 0.02,
        "dof_5": -0.01,  # missing dof_3 and dof_4 => results in None
        "dof_9": 0.12
    }
    anomaly_arr = preprocessor.build_anomaly_features(dof_example)
    print("\nAnomaly features array (None if incomplete):", anomaly_arr)