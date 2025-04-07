import requests
import logging
import os
from typing import Dict, Any, List

class FeedbackLoop:
    """
    A class to encapsulate the feedback loop logic, including automatic token handling.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:5002", endpoint: str = "send_feedback"):
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint.lstrip("/")
        self.logger = logging.getLogger("FeedbackLoop")

        # --- Attempt to log in automatically to get a fresh token. ---
        self.headers = {"Content-Type": "application/json"}
        self.token = self._obtain_token()  # Private method to log in & retrieve token
        if self.token:
            self.logger.info("[FeedbackLoop] Token acquired automatically.")
            self.headers["x-access-token"] = self.token
        else:
            self.logger.error("[FeedbackLoop] Could not retrieve token. Requests may fail with 401.")

    def _obtain_token(self) -> str:
        """
        Attempt to log in to get a fresh token. 
        Adjust the username/password if needed to match your test credentials.
        """
        login_payload = {
            "username": os.getenv("TEST_USERNAME", "rvsalinas"),
            "password": os.getenv("TEST_PASSWORD", "AIsensor123!")
        }
        try:
            resp = requests.post(f"{self.base_url}/login", json=login_payload, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("token", "")
            else:
                self.logger.warning(f"[FeedbackLoop] Login failed. Code={resp.status_code}, Body={resp.text}")
                return ""
        except Exception as e:
            self.logger.error(f"[FeedbackLoop] Exception while obtaining token: {e}", exc_info=True)
            return ""

    def submit_feedback(self, feedback_list: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Submit feedback to the backend endpoint. E.g.:
          feedback_list = [
            {"sensor_name": "TemperatureSensor", "feedback": "Working well"},
            {"sensor_name": "HumiditySensor", "feedback": "Needs recalibration"},
          ]
        """
        url = f"{self.base_url}/{self.endpoint}"
        payload = {"feedback": feedback_list}
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"[FeedbackLoop] Error submitting feedback: {e}", exc_info=True)
            return {"error": str(e)}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fb = FeedbackLoop()  # Will attempt login -> token
    example_feedback = [
        {"sensor_name": "TemperatureSensor", "feedback": "Working well"},
        {"sensor_name": "HumiditySensor", "feedback": "Calibration needed soon"},
    ]
    result = fb.submit_feedback(example_feedback)
    print("Feedback Submission Result:", result)