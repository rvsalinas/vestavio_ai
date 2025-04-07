"""
benchmarking_tool_module.py

Absolute File Path (example):
    /Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/benchmarking_tool_module.py

PURPOSE:
  - Benchmark endpoint performance (latency, success rate) across multiple iterations.
  - Log warnings for non-200 responses and record failures.
  - Optionally save results for later analysis or performance monitoring.

NOTES:
  - The module runs repeated requests to a given list of endpoints (or a single endpoint)
    with a payload and calculates statistics like average response time, success rate, etc.
  - You can integrate with your PerformanceMonitoringModule if desired, or just log the results.
  - Ensure the endpoints are reachable and returning valid JSON or you may see warnings/errors.
"""

import time
import requests
import logging
import statistics
import json
import os
from typing import Any, Dict, List


class BenchmarkingTool:
    def __init__(
        self,
        endpoints: List[str],
        payload: Dict[str, Any],
        iterations: int = 10,
        output_file: str = "benchmark_results.json",
    ):
        """
        :param endpoints: A list of endpoint URLs to benchmark. e.g. ["http://127.0.0.1:5002/api", ...]
        :param payload: The JSON payload to send in POST requests.
        :param iterations: Number of times to run each test per endpoint.
        :param output_file: Where to save the benchmark results in JSON format.
        """
        self.endpoints = endpoints
        self.payload = payload
        self.iterations = iterations
        self.output_file = output_file

        # Read a token from environment (if any), to be used in request headers:
        self.benchmark_token = os.getenv("BENCHMARK_TOKEN", "")

        self.logger = logging.getLogger("BenchmarkingTool")
        self.logger.setLevel(logging.INFO)

        self.logger.info(
            f"BenchmarkingTool initialized: endpoints={endpoints}, "
            f"iterations={iterations}, output_file={output_file}"
        )

    def benchmark(self) -> Dict[str, Any]:
        """
        Perform the benchmark on each endpoint with the given payload.
        Returns a dictionary of results including timings, success counts, etc.
        """
        results = {}

        for endpoint in self.endpoints:
            self.logger.info(f"Starting benchmark for endpoint: {endpoint}")
            timings: List[float] = []
            successes = 0
            failures: List[Dict[str, Any]] = []

            for i in range(self.iterations):
                start_time = time.perf_counter()
                try:
                    # Prepare headers, including token if present:
                    headers = {"Content-Type": "application/json"}
                    if self.benchmark_token:
                        headers["x-access-token"] = self.benchmark_token

                    resp = requests.post(endpoint, json=self.payload, headers=headers, timeout=5)
                    elapsed = time.perf_counter() - start_time
                    timings.append(elapsed)

                    if resp.status_code == 200:
                        successes += 1
                    else:
                        # Log a warning for non-200
                        self.logger.warning(
                            f"Iteration {i+1}: Endpoint {endpoint} returned {resp.status_code}. "
                            f"Response={resp.text}"
                        )
                        failures.append({
                            "iteration": i + 1,
                            "status_code": resp.status_code,
                            "response": resp.text,
                            "time": elapsed
                        })

                except requests.RequestException as e:
                    elapsed = time.perf_counter() - start_time
                    timings.append(elapsed)
                    self.logger.warning(
                        f"Iteration {i+1}: Endpoint {endpoint} request exception: {e}"
                    )
                    failures.append({
                        "iteration": i + 1,
                        "error": str(e),
                        "time": elapsed
                    })

            if timings:
                avg_time = statistics.mean(timings)
                min_time = min(timings)
                max_time = max(timings)
            else:
                avg_time = 0.0
                min_time = 0.0
                max_time = 0.0

            result = {
                "endpoint": endpoint,
                "iterations": self.iterations,
                "average_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "success_count": successes,
                "failure_count": len(failures),
                "failures": failures,
                "success_rate": float(successes) / float(self.iterations) if self.iterations > 0 else 0.0,
            }

            results[endpoint] = result
            self.logger.info(f"Benchmark result for {endpoint}: {result}")

        return results

    def save_results(self, results: Dict[str, Any]) -> None:
        """
        Save the benchmark results to a JSON file.
        """
        try:
            with open(self.output_file, "w") as f:
                json.dump(results, f, indent=4)
            self.logger.info(f"Benchmark results saved to {self.output_file}")
        except Exception as e:
            self.logger.error(f"Error saving benchmark results: {e}", exc_info=True)


if __name__ == "__main__":
    import json
    import os
    import logging

    logging.basicConfig(level=logging.INFO)

    # Read endpoints from environment or default
    test_endpoints = os.getenv("BENCHMARK_ENDPOINTS", "http://127.0.0.1:5002/endpoint").split(",")

    # Attempt to parse a JSON payload from env var BENCHMARK_PAYLOAD, or default to {"key":"value"}
    payload_str = os.getenv("BENCHMARK_PAYLOAD", '{"key": "value"}')
    try:
        test_payload = json.loads(payload_str)
    except Exception as e:
        logging.warning(f"Failed to parse BENCHMARK_PAYLOAD as JSON: {e}")
        test_payload = {"key": "value"}

    # Read iterations and output path
    iterations = int(os.getenv("BENCHMARK_ITERATIONS", "5"))
    output_path = os.getenv("BENCHMARK_RESULTS", "benchmark_results.json")

    # Initialize and run the benchmarking tool
    tool = BenchmarkingTool(
        endpoints=test_endpoints,
        payload=test_payload,
        iterations=iterations,
        output_file=output_path
    )
    benchmark_results = tool.benchmark()
    tool.save_results(benchmark_results)