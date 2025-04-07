"""
ui_integration_module.py

Absolute File Path (example):
    /Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/ui_integration_module.py

PURPOSE:
  - Provides an interface between the backend endpoints (e.g., /snapshot, /action) and a UI layer.
  - Fetches dashboard data for front-end rendering.
  - Sends user actions from the UI to the backend.

NOTES:
  - The UI typically relies on app.py or a similar server to provide data at /snapshot or /dashboard.
  - This module can also send user-triggered actions (e.g. calibrations, updates) to a backend endpoint.

EXAMPLE USAGE:
  ui = UIIntegration(base_url="http://localhost:5002")
  dashboard_data = ui.fetch_dashboard_data()
  result = ui.send_action_request({"action": "Calibrate Sensor X"})
"""

import os
import logging
import requests
from typing import Dict, Any

class UIIntegration:
    """
    A class that interfaces between the UI layer and backend REST endpoints.
    """

    def __init__(self, base_url: str = None):
        """
        :param base_url: The base URL of the backend server, e.g. "http://localhost:5002"
        """
        if base_url is None:
            base_url = os.getenv("BASE_URL", "http://localhost:5002")
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Content-Type": "application/json",
            # Optionally include token/keys if required by the backend:
            # "x-access-token": os.getenv("API_TOKEN", "dummy-token")
        }
        logging.info(f"UIIntegration initialized with base_url={self.base_url}")

    def fetch_dashboard_data(self) -> Dict[str, Any]:
        """
        Fetch data from the backend's dashboard or snapshot endpoint.
        Adjust the endpoint according to how your backend is structured.
        Returns JSON data or an error dict.
        """
        endpoint = "/snapshot"  # adjust if your backend uses a different path
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            response.raise_for_status()
            data = response.json()
            logging.info("Dashboard data fetched successfully.")
            return data
        except Exception as e:
            logging.error(f"Error fetching dashboard data: {e}", exc_info=True)
            return {"error": str(e)}

'''
    def send_action_request(self, action_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Example method to send a user action (e.g. "CalibrateSensorX") to the backend.
        Modify the endpoint to match your server route, e.g. '/action', '/user_action', etc.
        Returns JSON data or an error dict.
        """
        endpoint = "/action"
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.post(url, json=action_data, headers=self.headers, timeout=5)
            response.raise_for_status()
            data = response.json()
            logging.info(f"Action request '{action_data}' sent successfully.")
            return data
        except Exception as e:
            logging.error(f"Error sending action request: {e}", exc_info=True)
            return {"error": str(e)}
'''

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ui = UIIntegration()
    # Example: fetch dashboard data
    dashboard_data = ui.fetch_dashboard_data()
    print("Dashboard Data:", dashboard_data)

    # Example: send an action request
    # action_payload = {"action": "Calibrate Sensor X"}
    # action_response = ui.send_action_request(action_payload)
    # print("Action Response:", action_response)