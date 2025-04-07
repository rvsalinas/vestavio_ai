"""
collaborative_learning_module.py

absoulte file path:
/Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/collaborative_learning_module.py

A module for performing collaborative learning by merging local data
with external updates from a collaborative network.

Environment Variables Referenced:
  - COLLABORATIVE_API_KEY: API key for authentication
  - BASE_URL: The root server URL (e.g., http://localhost:5002)
  - COLLABORATIVE_DATA_PATH (optional): Path to the local data JSON file
"""

import json
import os
import logging
import requests
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeLearningModule:
    """
    A module for performing collaborative learning:
      - Load local data
      - Fetch external data from an endpoint
      - Merge the data based on similarity
      - Save/optionally send back
    """

    def __init__(
        self,
        local_data_path: str = None,
        server_url: str = None,
        api_key: str = None,
    ):
        """
        :param local_data_path: JSON file for storing local collaborative data.
        :param server_url: The endpoint to fetch/submit updates.
        :param api_key: Optional API key for authentication.
        """
        # Load from environment if not provided
        env_local_data = os.getenv("COLLABORATIVE_DATA_PATH", "collaborative_data.json")
        env_base_url = os.getenv("BASE_URL", "http://localhost:5002")
        env_api_key = os.getenv("COLLABORATIVE_API_KEY", "")

        self.local_data_path = local_data_path or env_local_data
        # Previously used "/fake_collab_endpoint", but app.py actually exposes "/collaborative_update"
        self.server_url = server_url or f"{env_base_url}/collaborative_update"
        self.api_key = api_key or env_api_key

        self.logger = logging.getLogger("CollaborativeLearningModule")
        if not self.logger.handlers:
            # Setup a default console handler if none exist
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)

        # For security, only show "(hidden)" if api_key is set
        key_status = "(hidden)" if self.api_key else "(none)"
        self.logger.info(
            f"CollaborativeLearningModule initialized with local_data_path={self.local_data_path}, "
            f"server_url={self.server_url}, api_key={key_status}"
        )

    def load_local_data(self) -> List[Dict[str, Any]]:
        """
        Load local data from JSON file, or return an empty list if file doesn't exist.
        """
        try:
            with open(self.local_data_path, "r") as f:
                data = json.load(f)
            self.logger.info(f"Loaded {len(data)} entries from local data file: {self.local_data_path}")
            return data
        except FileNotFoundError:
            self.logger.warning(f"No local data found at {self.local_data_path}. Starting fresh.")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON from {self.local_data_path}: {e}")
            return []

    def save_local_data(self, data: List[Dict[str, Any]]):
        """Save updated data to JSON."""
        try:
            with open(self.local_data_path, "w") as f:
                json.dump(data, f, indent=4)
            self.logger.info(f"Saved {len(data)} entries to {self.local_data_path}")
        except Exception as e:
            self.logger.error(f"Failed to save local data: {e}", exc_info=True)

    def fetch_external_updates(self) -> List[Dict[str, Any]]:
        """
        Fetch external data from a collaborative server or placeholder endpoint.
        Return a list of dictionaries.
        """
        self.logger.info(f"Fetching external updates from {self.server_url}")
        if not self.server_url:
            self.logger.error("No server URL provided. Skipping fetch.")
            return []

        headers = {}
        if self.api_key:
            headers["x-access-token"] = self.api_key  # or any other header scheme

        try:
            resp = requests.get(self.server_url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            # Expecting data to be a list or a dict containing "external_updates"
            if isinstance(data, dict) and "external_updates" in data:
                external_data = data["external_updates"]
                self.logger.info(f"Fetched {len(external_data)} updates from server.")
                return external_data
            elif isinstance(data, list):
                # If it returns a direct list
                self.logger.info(f"Fetched {len(data)} items from server (list).")
                return data
            else:
                self.logger.warning("Unexpected data format from server.")
                return []
        except requests.RequestException as e:
            self.logger.error(f"Error fetching external updates: {e}", exc_info=True)
            return []
        except ValueError as e:
            self.logger.error(f"Error parsing server response as JSON: {e}", exc_info=True)
            return []

    def send_updates_to_server(self, updates: List[Dict[str, Any]]):
        """
        Optional. Send merged data back to the server if needed.
        """
        if not self.api_key:
            self.logger.error("No API key found; cannot send updates to server.")
            return
        if not self.server_url:
            self.logger.error("No server URL provided; cannot send updates.")
            return

        try:
            headers = {
                "x-access-token": self.api_key,
                "Content-Type": "application/json"
            }
            resp = requests.post(self.server_url, json=updates, headers=headers, timeout=10)
            if resp.status_code == 200:
                self.logger.info(f"Updates sent successfully. Server responded: {resp.text}")
            else:
                self.logger.error(f"Failed to send updates: code {resp.status_code}, msg: {resp.text}")
        except requests.RequestException as e:
            self.logger.error(f"Network error sending updates: {e}", exc_info=True)

    def calculate_similarity(self, local_data: List[Dict[str, Any]], ext_data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Example: Use 'features' key in local_data and ext_data for similarity.
        Return a 2D array of shape (local_len, external_len).
        """
        local_mat = np.array([d.get("features", [0.1, 0.2, 0.3]) for d in local_data])
        ext_mat = np.array([d.get("features", [0.3, 0.3, 0.3]) for d in ext_data])
        if local_mat.size == 0 or ext_mat.size == 0:
            self.logger.warning("No features found for similarity calculation.")
            return np.array([])
        sim_matrix = cosine_similarity(local_mat, ext_mat)
        self.logger.info(f"Calculated similarity matrix of shape {sim_matrix.shape}.")
        return sim_matrix

    def merge_data(self, local_data: List[Dict[str, Any]], external_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Example merging logic: For each local entry, find the most-similar external entry.
        If similarity < threshold, append the external entry to local data.
        """
        if not external_data:
            self.logger.info("No external data provided for merge. Returning local data as-is.")
            return local_data

        similarities = self.calculate_similarity(local_data, external_data)
        if similarities.size == 0:
            self.logger.info("Similarity matrix is empty. Returning local data as-is.")
            return local_data

        merged = local_data.copy()
        threshold = 0.8

        # For each local item, check its highest similarity
        for i, _ in enumerate(local_data):
            sim_scores = similarities[i]
            j = np.argmax(sim_scores)
            if sim_scores[j] < threshold:
                merged_item = external_data[j]
                self.logger.info(f"Merging external item with similarity={sim_scores[j]:.2f} below threshold={threshold}")
                merged.append(merged_item)

        return merged

    def run_collaborative_learning(self):
        """Main method to orchestrate fetching, merging, saving, etc."""
        local_data = self.load_local_data()
        external = self.fetch_external_updates()

        if external:
            merged_data = self.merge_data(local_data, external)
            self.save_local_data(merged_data)
            # Optional: self.send_updates_to_server(merged_data)
        else:
            self.logger.info("No external updates received. Collaborative learning skipped.")


# Example usage if directly running this file:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clm = CollaborativeLearningModule()
    clm.run_collaborative_learning()