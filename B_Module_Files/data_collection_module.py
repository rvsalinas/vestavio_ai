"""
data_collection_module.py

Absolute File Path (Example):
    /Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/data_collection_module.py

PURPOSE:
  - Collect data from sensor APIs or external endpoints
  - Store or forward the data to a database or other service.

NEEDS TO DO:
  - For real-world usage, replace the mock sensor API and DB endpoints in your .env file
    so that data_collection_module fetches from your actual sensors and stores in your production DB.

ENV VARS:
  - SENSOR_API_URL: The endpoint where sensor data is retrieved
  - DB_API_ENDPOINT: The endpoint (or DB service) where fetched sensor data is sent/stored
  - Other relevant environment variables, if needed
"""

import requests
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Union

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# Read environment variables for URLs
SENSOR_API_URL = os.getenv("SENSOR_API_URL", "https://jsonplaceholder.typicode.com/posts")
DB_API_ENDPOINT = os.getenv("DB_API_ENDPOINT", "https://webhook.site/your-test-webhook")

class DataCollector:
    """
    A module for data collection from sensor APIs or external endpoints,
    storing results in a database or another endpoint.
    """

    def __init__(
        self,
        sensor_api_url: str = SENSOR_API_URL,
        db_api_endpoint: str = DB_API_ENDPOINT
    ):
        """
        :param sensor_api_url: The URL from which to fetch sensor data
        :param db_api_endpoint: The URL/endpoint to which collected data is sent
        """
        self.sensor_api_url = sensor_api_url
        self.db_api_endpoint = db_api_endpoint

        # Set up a logger if none exists
        self.logger = logging.getLogger("DataCollector")
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)

        self.logger.info(
            f"DataCollector initialized with sensor_api_url={self.sensor_api_url}, "
            f"db_api_endpoint={self.db_api_endpoint}"
        )

    def fetch_sensor_data(self) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Fetch data from the sensor API.
        :return: The sensor data as a list of dicts or a single dict. Returns
                 an error dict on failure.
        """
        try:
            self.logger.info(f"Fetching sensor data from {self.sensor_api_url}")
            response = requests.get(self.sensor_api_url, timeout=10)
            response.raise_for_status()
            sensor_data = response.json()
            self.logger.info(f"Fetched {len(sensor_data) if isinstance(sensor_data, list) else 'some'} items.")
            return sensor_data
        except requests.exceptions.RequestException as e:
            error_msg = f"Error fetching sensor data: {e}"
            self.logger.error(error_msg, exc_info=True)
            return {"error": error_msg}

    def send_to_db(self, sensor_data: Union[List[Dict[str, Any]], Dict[str, Any]]):
        """
        Send the collected sensor data to the database or webhook endpoint.
        :param sensor_data: The sensor data to store or forward.
        """
        if not self.db_api_endpoint:
            self.logger.warning("DB_API_ENDPOINT is not set; cannot store data.")
            return

        try:
            payload = {
                "sensor_data": sensor_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            headers = {"Content-Type": "application/json"}
            self.logger.info(f"Sending data to DB endpoint: {self.db_api_endpoint}")
            response = requests.post(self.db_api_endpoint, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            self.logger.info("Sensor data successfully stored (or forwarded).")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error sending data to the DB endpoint: {e}", exc_info=True)

    def collect_and_store(self):
        """
        High-level method: collects sensor data then attempts to store it.
        """
        data = self.fetch_sensor_data()
        if isinstance(data, dict) and "error" in data:
            self.logger.warning("No sensor data to store due to fetch error.")
            return
        self.send_to_db(data)

if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_and_store()