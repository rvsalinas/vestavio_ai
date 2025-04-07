"""
scaling_and_load_balancer_module.py
Absolute File Path:
  /Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/scaling_and_load_balancer_module.py.py

PURPOSE:
  - Distribute load across multiple servers in a pool.
  - Provide a simple forward_request(...) method to send payloads to one of the servers.
  - Perform periodic health checks to see which servers are available.

NOTES:
  - This script tries servers in the pool sequentially until one returns a 200 response.
  - The health_check() method attempts a GET to /health (configurable).
  - You can run it directly (python scaling_and_load_balancer_module.py) for a quick test,
    or import it into your application code for integration.

"""

import os
import logging
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any


class ScalingAndLoadBalancer:
    """
    A class that handles distributing requests across multiple servers and
    performing health checks.
    """

    def __init__(
        self,
        server_pool: List[str],
        max_requests: int = 100,
        health_check_url: str = "/health"
    ):
        """
        :param server_pool: List of server base URLs (e.g., ["http://127.0.0.1:5000", "http://127.0.0.1:5001"]).
        :param max_requests: Maximum concurrent requests allowed (for ThreadPoolExecutor).
        :param health_check_url: Endpoint for health checks (default '/health').
        """
        # Clean up server list, remove empty strings
        self.server_pool = [s.strip() for s in server_pool if s.strip()]

        # Thread pool for potential concurrency
        self.executor = ThreadPoolExecutor(max_workers=max_requests)

        self.health_check_url = health_check_url

        # Setup logger
        self.logger = logging.getLogger("ScalingAndLoadBalancer")
        self.logger.setLevel(logging.INFO)
        # If desired, add a handler (e.g., console, rotating file)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(ch)

        self.logger.info(f"Initialized load balancer with servers: {self.server_pool}")

    def forward_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward the request (POST) to the first available server in the pool.
        :param endpoint: The path on the server to post to (e.g. "/send_sensor_data").
        :param payload: The JSON payload to send in the POST request.
        :return: JSON response from the successful server, or an error dict if none succeeded.
        """
        if not self.server_pool:
            msg = "No servers in the pool."
            self.logger.error(msg)
            return {"error": msg}

        for server in self.server_pool:
            try:
                url = f"{server.rstrip('/')}/{endpoint.lstrip('/')}"
                self.logger.info(f"Forwarding request to {url} with payload: {payload}")
                response = requests.post(url, json=payload, timeout=5)
                if response.status_code == 200:
                    self.logger.info(f"Request forwarded successfully to {server}")
                    return response.json()  # Return successful response as dict
                else:
                    self.logger.warning(
                        f"Server {server} returned status {response.status_code}. "
                        f"Response body: {response.text}"
                    )
            except requests.RequestException as e:
                self.logger.error(f"Error communicating with {server}: {e}", exc_info=True)

        return {"error": "All servers are currently unavailable."}

    def health_check(self) -> List[str]:
        """
        Perform health checks on all servers in the pool.
        :return: A list of healthy server URLs that returned a 200 from /health
                 (or whichever health_check_url is configured).
        """
        healthy_servers = []
        if not self.server_pool:
            self.logger.warning("No servers in pool to check.")
            return healthy_servers

        self.logger.info("Performing health checks...")
        for server in self.server_pool:
            try:
                url = f"{server.rstrip('/')}/{self.health_check_url.lstrip('/')}"
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    healthy_servers.append(server)
                    self.logger.info(f"Server healthy: {server}")
                else:
                    self.logger.warning(f"Server {server} unhealthy. Code: {resp.status_code}")
            except requests.RequestException as e:
                self.logger.error(f"Health check error on {server}: {e}", exc_info=True)

        self.logger.info(f"Healthy servers: {healthy_servers}")
        return healthy_servers


if __name__ == "__main__":
    # Basic demonstration if run as a standalone script.
    logging.basicConfig(level=logging.INFO)
    # Possibly read from env: SERVER_POOL="http://127.0.0.1:5000, http://127.0.0.1:5001"
    server_list = os.getenv("SERVER_POOL", "http://127.0.0.1:5000").split(",")
    balancer = ScalingAndLoadBalancer(server_pool=server_list, max_requests=50)

    # Check health of servers
    healthy = balancer.health_check()
    print("Currently healthy servers:", healthy)

    # Attempt to forward a sample request if any servers exist
    if healthy:
        result = balancer.forward_request("/example_endpoint", {"key": "value"})
        print("Forward result:", result)
    else:
        print("No healthy servers to forward a request to.")