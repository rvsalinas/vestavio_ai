"""
performance_monitoring_module.py

Absolute File Path (example):
    /Users/yourname/Desktop/energy_optimization_project/B_Module_Files/performance_monitoring_module.py

PURPOSE:
  - Continuously or periodically track and record system performance metrics such as:
      * Inference latency
      * Throughput (# of requests)
      * CPU/Memory usage
      * Domain-specific metrics (energy savings, anomalies, etc.)
  - Store metrics in a JSON file or maintain them in memory for external analysis.
  - Provide methods to query or summarize these metrics over time.

REQUIREMENTS:
  - (Optional) psutil for CPU/memory usage tracking. If not installed, CPU/memory features are skipped.
  - JSON file-based storage for metric persistence or set `store_in_json=False` for in-memory only.
"""

import logging
import json
import os
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


class PerformanceMonitoringModule:
    """
    A class to handle system performance metrics such as:
      - Inference latency
      - Throughput
      - CPU/memory usage (optional, requires psutil)
      - Additional domain metrics (e.g. anomaly rates, energy savings, etc.)

    Stores these metrics in a JSON file or an in-memory list.

    Example usage:
        pm = PerformanceMonitoringModule(
            performance_file="performance_history.json",
            store_in_json=True
        )

        # time a function call:
        @pm.monitor_inference
        def sample_inference(x):
            # do stuff
            return x * 2

        output = sample_inference(5)
        # metrics now stored in performance_history.json
    """

    def __init__(
        self,
        performance_file: str = "performance_history.json",
        store_in_json: bool = True,
        max_records: int = 10000,
    ):
        """
        :param performance_file: The JSON file to store performance records. If it doesn't exist, it is created.
        :param store_in_json: If True, metrics are persisted to a JSON file. Otherwise stored in memory only.
        :param max_records: If > 0, the system may drop oldest metrics once max_records is exceeded.
        """
        self.logger = logging.getLogger("PerformanceMonitoringModule")
        self.performance_file = performance_file
        self.store_in_json = store_in_json
        self.max_records = max_records
        self.memory_data: Dict[str, List[Dict[str, Any]]] = {"records": []}

        if self.store_in_json:
            if not os.path.exists(self.performance_file):
                with open(self.performance_file, "w") as f:
                    json.dump({"records": []}, f, indent=4)
            self.logger.info(f"PerformanceMonitoringModule initialized. Using {self.performance_file} for storage.")
        else:
            self.logger.info("PerformanceMonitoringModule initialized. Storing metrics in memory only.")

    def record_metric(
        self,
        metric_name: str,
        value: float,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save a performance metric with a timestamp.
        :param metric_name: e.g. "inference_latency" or "cpu_usage"
        :param value: numeric value
        :param extra_info: dict of additional info (e.g., model name, user_id, etc.)
        """
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metric_name": metric_name,
            "value": value,
            "extra_info": extra_info or {},
        }

        self._append_record(record)
        self.logger.info(f"Recorded metric: {metric_name} = {value:.4f}")

    def monitor_inference(self, func):
        """
        Decorator that measures inference time for a function and logs CPU/memory usage (if psutil is available).
        Usage:
            @monitor_inference
            def predict(data):
                ...
                return result
        """
        def wrapper(*args, **kwargs):
            start = time.time()
            if _PSUTIL_AVAILABLE:
                cpu_before = psutil.cpu_percent(interval=None)
                mem_before = psutil.virtual_memory().percent
            result = func(*args, **kwargs)
            end = time.time()
            latency = end - start
            self.record_metric("inference_latency", latency, extra_info={"func_name": func.__name__})

            if _PSUTIL_AVAILABLE:
                cpu_after = psutil.cpu_percent(interval=None)
                mem_after = psutil.virtual_memory().percent
                self.record_metric(
                    "cpu_usage_diff",
                    (cpu_after - cpu_before),
                    extra_info={"func_name": func.__name__},
                )
                self.record_metric(
                    "mem_usage_diff",
                    (mem_after - mem_before),
                    extra_info={"func_name": func.__name__},
                )

            return result
        return wrapper

    def record_cpu_mem_usage(self, tag: str = ""):
        """
        Manually record CPU/memory usage at a moment in time (if psutil is available).
        :param tag: some tag to group or identify this measurement (e.g. 'pre_inference')
        """
        if not _PSUTIL_AVAILABLE:
            self.logger.warning("psutil not available; skipping CPU/memory usage recording.")
            return

        cpu_percent = psutil.cpu_percent(interval=None)
        mem_percent = psutil.virtual_memory().percent

        self.record_metric("cpu_usage", cpu_percent, extra_info={"tag": tag})
        self.record_metric("mem_usage", mem_percent, extra_info={"tag": tag})

    def _append_record(self, record: Dict[str, Any]) -> None:
        """
        Internal method to append a new metric record either to a JSON file or in-memory data.
        """
        if self.store_in_json:
            # load file
            with open(self.performance_file, "r") as f:
                data = json.load(f)
            data["records"].append(record)

            # If we exceed max_records, trim the oldest
            if self.max_records > 0 and len(data["records"]) > self.max_records:
                data["records"] = data["records"][-self.max_records:]

            # save
            with open(self.performance_file, "w") as f:
                json.dump(data, f, indent=4)

        else:
            # store in self.memory_data
            self.memory_data["records"].append(record)
            if self.max_records > 0 and len(self.memory_data["records"]) > self.max_records:
                self.memory_data["records"] = self.memory_data["records"][-self.max_records:]

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return all stored performance records.
        """
        if self.store_in_json:
            with open(self.performance_file, "r") as f:
                data = json.load(f)
            return data
        else:
            return self.memory_data

    def summarize_metrics(
        self,
        metric_name: str,
        lookback_minutes: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Return aggregated stats for a specific metric (like 'inference_latency').
        Optionally filter by a recent time window (lookback_minutes).
        Returns { 'count': ..., 'mean': ..., 'min': ..., 'max': ... }.
        """
        all_data = self.get_metrics()
        records = all_data.get("records", [])

        # Filter by metric_name
        filtered = [r for r in records if r["metric_name"] == metric_name]

        # If lookback_minutes specified, filter further
        if lookback_minutes is not None and lookback_minutes > 0:
            cutoff = datetime.now() - timedelta(minutes=lookback_minutes)
            def parse_ts(ts_str: str) -> datetime:
                return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            filtered = [r for r in filtered if parse_ts(r["timestamp"]) > cutoff]

        if not filtered:
            return {
                "count": 0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        values = [r["value"] for r in filtered]
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    pm = PerformanceMonitoringModule(
        performance_file="performance_history.json",
        store_in_json=True,
        max_records=1000
    )

    # Define a dummy inference function
    @pm.monitor_inference
    def sample_inference(x: float):
        time.sleep(0.05)  # simulate some compute time
        return x * 2

    # Manually log CPU usage (if psutil is installed) before inference
    pm.record_cpu_mem_usage(tag="pre_inference")

    # Perform the sample inference
    result = sample_inference(10)
    print("Sample inference result:", result)

    # Manually log CPU usage after inference
    pm.record_cpu_mem_usage(tag="post_inference")

    # Summarize the last 60 minutes of 'inference_latency'
    summary = pm.summarize_metrics("inference_latency", lookback_minutes=60)
    print("Inference Latency (last 60 min):", summary)

    # Dump all stored metrics
    all_metrics = pm.get_metrics()
    print("All performance metrics stored so far:\n", all_metrics)