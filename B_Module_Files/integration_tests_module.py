import unittest
import requests
import logging
import os

# Default to local app
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:5002")

# If login fails or is skipped, we fall back to a static token from env
FALLBACK_TOKEN = os.getenv(
    "API_TOKEN",
    # Example token from your logs:
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
)

# Optional credentials to try logging in automatically
DEFAULT_TEST_USERNAME = os.getenv("API_TEST_USER", "rvsalinas")
DEFAULT_TEST_PASSWORD = os.getenv("API_TEST_PASS", "AIsensor123!")

class TestIntegrationEndpoints(unittest.TestCase):
    """
    Integration tests to ensure various service endpoints are working.
    """

    @classmethod
    def setUpClass(cls):
        """
        This method runs once before *any* tests in this suite.
        We'll attempt to log in and get a fresh token.
        If login fails, we'll fall back to the fallback token (FALLBACK_TOKEN).
        """
        logging.basicConfig(level=logging.INFO)
        cls.base_url = BASE_URL.rstrip("/")

        # Attempt to log in and obtain a fresh token
        login_payload = {
            "username": DEFAULT_TEST_USERNAME,
            "password": DEFAULT_TEST_PASSWORD
        }
        try:
            resp = requests.post(f"{cls.base_url}/login", json=login_payload, timeout=5)
            resp.raise_for_status()  # Will raise if status_code != 200
            json_data = resp.json()
            cls.token = json_data.get("token", "")
            if not cls.token:
                raise ValueError("No 'token' found in login response.")
            logging.info(f"[setUpClass] Logged in as '{DEFAULT_TEST_USERNAME}'; new token acquired.")
        except Exception as e:
            logging.error(f"[setUpClass] Login failed: {e} -- falling back to FALLBACK_TOKEN.")
            cls.token = FALLBACK_TOKEN

    def setUp(self):
        """
        This runs before each test. We set up common headers here
        so all test methods can use them.
        """
        self.headers = {
            "Content-Type": "application/json",
            "x-access-token": self.token
        }

    def test_health_check(self):
        """
        Test a /health endpoint to see if the server is running.
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            self.assertEqual(response.status_code, 200, "Health check failed.")
            json_resp = response.json()
            self.assertIn("status", json_resp, "Missing 'status' key in response.")
            # Accept typical responses like "Server is running" or "OK"
            self.assertIn(json_resp["status"], ["OK", "Server is running"], "Unexpected health status.")
        except Exception as e:
            self.fail(f"Health check test failed with exception: {e}")

    def test_sensor_data_submission(self):
        """
        Test the /send_sensor_data endpoint to confirm sensor data is processed.
        """
        try:
            payload = {
                "sensor_data": [
                    {"name": "TemperatureSensor", "output": 22.5, "status": "Operational"},
                    {"name": "HumiditySensor", "output": 60.2, "status": "Operational"}
                ]
            }
            response = requests.post(
                f"{self.base_url}/send_sensor_data",
                json=payload,
                headers=self.headers,
                timeout=5
            )
            self.assertEqual(response.status_code, 200, "Failed to submit sensor data (non-200).")
            json_resp = response.json()
            self.assertIn("status", json_resp, "Missing 'status' key in response.")
            self.assertEqual(json_resp["status"], "Data processed successfully", "Unexpected status message.")
        except Exception as e:
            self.fail(f"Sensor data submission test failed with exception: {e}")

    def test_dashboard_data(self):
        """
        Test the /snapshot endpoint to confirm that the sensor snapshot is retrieved.
        """
        try:
            response = requests.get(f"{self.base_url}/snapshot", headers=self.headers, timeout=5)
            self.assertEqual(response.status_code, 200, "Snapshot endpoint failed.")
            json_resp = response.json()
            sensors = json_resp.get("sensors", [])
            self.assertIsInstance(sensors, list, "Expected 'sensors' to be a list.")
            self.assertGreater(len(sensors), 0, "Expected at least one sensor in the snapshot.")
        except Exception as e:
            self.fail(f"Dashboard data test failed with exception: {e}")

    def test_feedback_submission(self):
        """
        Test the /send_feedback endpoint to ensure feedback data is processed.
        """
        try:
            feedback_payload = {
                "feedback": [
                    {"sensor_name": "TemperatureSensor", "feedback": "Working well"},
                    {"sensor_name": "HumiditySensor", "feedback": "Needs calibration"}
                ]
            }
            response = requests.post(
                f"{self.base_url}/send_feedback",
                json=feedback_payload,
                headers=self.headers,
                timeout=5
            )
            self.assertEqual(response.status_code, 200, "Failed to send feedback (non-200).")
            json_resp = response.json()
            self.assertIn("status", json_resp, "Missing 'status' key in response.")
            self.assertEqual(json_resp["status"], "Feedback processed successfully", "Unexpected status.")
        except Exception as e:
            self.fail(f"Feedback submission test failed with exception: {e}")


if __name__ == "__main__":
    unittest.main()