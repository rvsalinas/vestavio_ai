"""
File: environmental_context_awareness_module.py

Absolute File Path:
    /Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/environmental_context_awareness_module.py

PURPOSE:
    This module integrates real-time environmental data (e.g., weather) into the system’s decision-making process.
    It fetches external data (e.g., weather info, occupancy data) to enhance predictions or decisions made by other
    modules like data_fusion_gateway_module or sensor_integration_module.

    Key Features:
    1. Reads .env variables for TIMEZONE, LOCATION, and WEATHER_API_KEY.
    2. Fetches weather data from an external weather API (e.g., OpenWeatherMap).
    3. Parses and processes relevant weather information (temperature, humidity, etc.).
    4. Provides an interface or function to retrieve the parsed environmental context data.

    USAGE EXAMPLE (standalone):
        (sensor_fusion_env) $ python environmental_context_awareness_module.py

    or you may import:
        from environmental_context_awareness_module import EnvironmentalContextAwareness

    # Sample usage in code:
        env_module = EnvironmentalContextAwareness()
        weather_data = env_module.fetch_current_weather()
        # Do something with weather_data

"""

import os
import sys
import json
import requests
from dotenv import load_dotenv
from datetime import datetime


class EnvironmentalContextAwareness:
    """
    A class that manages fetching and parsing external environmental data.
    Example: Weather data from an external API, local timezone, etc.
    """

    def __init__(self, env_file_path: str = ".env"):
        """
        Initialize the module, load environment variables, and set up defaults.
        :param env_file_path: Path to the .env file containing config.
        """
        # Load .env variables
        if not os.path.exists(env_file_path):
            print(f"[WARNING] .env file not found at {env_file_path}. Using system env variables if present.")
        else:
            load_dotenv(env_file_path)

        # Grab required env variables (fallback to None if missing)
        self.timezone = os.getenv("TIMEZONE", "UTC")
        self.location = os.getenv("LOCATION", "UnknownLocation")
        self.weather_api_key = os.getenv("WEATHER_API_KEY")

        # Example: if you have an OpenWeatherMap API:
        #   https://api.openweathermap.org/data/2.5/weather?q={city}&appid={YOUR_API_KEY}
        self.weather_api_url = "https://api.openweathermap.org/data/2.5/weather"
        self.session = requests.Session()

    def fetch_current_weather(self):
        """
        Fetch the current weather for the configured location via an external API.
        :return: A dictionary containing weather data or None if an error occurred.
        """
        if not self.weather_api_key:
            print("[ERROR] WEATHER_API_KEY is not set. Cannot fetch weather data.")
            return None

        # Additional params might include units=metric or imperial if desired
        params = {
            "q": self.location,
            "appid": self.weather_api_key,
            "units": "imperial",  # or 'metric'
        }

        try:
            print(f"[INFO] Fetching weather data for location: {self.location} in timezone: {self.timezone}")
            response = self.session.get(self.weather_api_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                print("[INFO] Weather data fetched successfully.")
                return self._parse_weather_data(data)
            else:
                print(f"[ERROR] Failed to fetch weather data. HTTP {response.status_code} - {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Exception during weather API call: {e}")
            return None

    def _parse_weather_data(self, data: dict):
        """
        Parse raw weather data from the API response. 
        Adjust fields as needed for your project’s specific context.
        :param data: A dictionary of raw weather data from the external API.
        :return: A dictionary with relevant parsed weather information.
        """
        if "main" not in data or "weather" not in data:
            print("[WARNING] Unexpected weather data format.")
            return data

        main_info = data["main"]
        weather_list = data["weather"]
        wind_info = data.get("wind", {})

        parsed = {
            "location": self.location,
            "temperature_f": main_info.get("temp"),
            "humidity": main_info.get("humidity"),
            "weather_description": weather_list[0]["description"] if weather_list else "No description",
            "wind_speed": wind_info.get("speed"),
            "timestamp": datetime.now().isoformat(),
        }
        return parsed

    def get_environmental_context(self):
        """
        Retrieve the combined environmental context data for downstream usage.
        :return: Dictionary containing environment context (weather, timezone, location, etc.).
        """
        weather_data = self.fetch_current_weather()
        if weather_data is None:
            # Return at least partial context
            return {
                "timezone": self.timezone,
                "location": self.location,
                "weather": None,
                "fetch_status": "failed",
            }

        return {
            "timezone": self.timezone,
            "location": self.location,
            "weather": weather_data,
            "fetch_status": "ok",
        }


def main():
    """
    Example usage if running this script directly.
    """
    print("[INFO] Initializing EnvironmentalContextAwareness module...")
    env_module = EnvironmentalContextAwareness()

    print("[INFO] Fetching complete environmental context...")
    context_data = env_module.get_environmental_context()

    print("[INFO] Environmental Context Data:")
    print(json.dumps(context_data, indent=2))


if __name__ == "__main__":
    main()
