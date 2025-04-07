"""
real_time_monitoring_module.py

Absolute File Path (example):
    /Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/real_time_monitoring_module.py

PURPOSE:
  - Periodically fetches sensor data from a backend endpoint (e.g. /snapshot).
  - Displays or logs real-time sensor readings.
  - Allows stopping the monitoring loop interactively.

NOTES:
  - The /snapshot endpoint must be running in app.py (or wherever your server is hosted).
  - This script polls that endpoint at a set interval (poll_interval).
  - If the user presses Enter, monitoring stops.
"""

import time
import json
import requests
import logging
import os
import sys
import select
from typing import Dict, Any, List


class RealTimeMonitoring:
    """
    A class for real-time monitoring of sensor data. Periodically polls a /snapshot endpoint
    and prints or logs the returned sensor data.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5002",
        endpoint: str = "/snapshot",
        poll_interval: float = 30.0
    ):
        """
        :param base_url: Base URL for the backend server (e.g. http://localhost:5002).
        :param endpoint: The endpoint to poll (e.g. /snapshot).
        :param poll_interval: Number of seconds to wait between polls.
        """
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint.lstrip("/")
        self.poll_interval = poll_interval

        # If your server requires auth, tokens, etc., set them here:
        self.headers = {
            "Content-Type": "application/json",
            # Example if you have an auth token:
            # "x-access-token": os.getenv("API_TOKEN", "dummy-token")
        }
        logging.info(
            f"RealTimeMonitoring initialized: base_url={self.base_url}, "
            f"endpoint={self.endpoint}, poll_interval={self.poll_interval}"
        )

    def fetch_sensor_data(self) -> Dict[str, Any]:
        """
        Fetch data from the backend /snapshot (or configured) endpoint.
        Returns a dictionary (parsed JSON). Example:
          {
            "sensors": [
              { "sensor_name": "TempSensor", "sensor_output": 22.5, "status": "Operational" },
              ...
            ]
          }
        If an error occurs, returns {"error": "<error_msg>"}.
        """
        url = f"{self.base_url}/{self.endpoint}"
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error fetching sensor data from {url}: {e}", exc_info=True)
            return {"error": str(e)}

    def start_monitoring(self):
        """
        Enter a loop that continuously fetches sensor data at poll_interval.
        Pressing Enter stops the loop.
        """
        logging.info("Starting real-time monitoring loop...")
        print("\nReal-Time Monitoring started.\nPress Enter at any time to stop.\n")

        try:
            while True:
                data = self.fetch_sensor_data()

                if "error" in data:
                    print(f"Error fetching sensor data: {data['error']}")
                else:
                    # Example: parse sensor list
                    sensors = data.get("sensors", [])
                    print("=== Sensor Data Snapshot ===")
                    for sensor in sensors:
                        name = sensor.get("sensor_name", "UnknownSensor")
                        output = sensor.get("sensor_output", "N/A")
                        status = sensor.get("status", "Unknown")
                        print(f"  - {name}: {output} [{status}]")
                    print("============================\n")

                # Wait up to poll_interval, but break immediately if user presses Enter
                start_time = time.time()
                while True:
                    elapsed = time.time() - start_time
                    if elapsed >= self.poll_interval:
                        break
                    if self._user_pressed_enter():
                        raise KeyboardInterrupt
                    time.sleep(0.25)

        except KeyboardInterrupt:
            print("\nReal-time monitoring stopped by user.")
        logging.info("Real-time monitoring loop ended.")

    def _user_pressed_enter(self) -> bool:
        """
        Checks non-blocking for user input. If user pressed Enter, returns True.
        Otherwise, returns False. Avoids halting the loop with input() calls.
        """
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            # Read the line to clear the input
            _ = sys.stdin.readline()
            return True
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Optionally read from env or fallback
    base_url = os.getenv("BASE_URL", "http://localhost:5002")
    poll_interval = float(os.getenv("POLL_INTERVAL", 10.0))

    monitor = RealTimeMonitoring(base_url=base_url, endpoint="/snapshot", poll_interval=poll_interval)
    monitor.start_monitoring()