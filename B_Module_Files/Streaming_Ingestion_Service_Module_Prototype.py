"""
Streaming_Ingestion_Service_Module_Prototype.py

Absolute File Path (example):
    /Users/username/Desktop/energy_optimization_project/B_Module_Files/Streaming_Ingestion_Service_Module_Prototype.py

PURPOSE:
  - Demonstrates a prototype module for streaming data ingestion at intervals.
  - Could simulate reading from a live source (e.g., Kafka), generating synthetic data,
    or collecting data from an external API.
  - Feeds data to a callback or queue for downstream processing.

NOTES:
  - The script uses a background thread that runs until stopped.
  - You can enhance it to handle parallel streams, error handling, or real data ingestion.
"""

import logging
import time
import threading
import random
import queue
from typing import Callable, Optional, Dict, Any


class StreamingIngestionService:
    """
    A prototype streaming ingestion service:
      - Runs a loop that fetches or produces data at fixed or dynamic intervals.
      - Sends or queues data for downstream processing.
      - Provides start/stop controls with a background thread.
    """

    def __init__(
        self,
        produce_interval: float = 1.0,
        data_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_queue_size: int = 100,
    ):
        """
        :param produce_interval: Interval in seconds to produce or fetch data.
        :param data_callback: A function(data_dict) invoked each time new data arrives.
                             If None, logs data instead.
        :param max_queue_size: If using a queue-based approach, this is the max queue length.
        """
        self.logger = logging.getLogger("StreamingIngestionService")
        self.produce_interval = produce_interval
        self.data_callback = data_callback or self._default_data_handler
        self._stop_event = threading.Event()
        self._thread = None

        # Optionally, we can store data in an internal queue if desired:
        self.data_queue = queue.Queue(maxsize=max_queue_size)

        self.logger.info(
            "StreamingIngestionService initialized with produce_interval=%.2f sec",
            self.produce_interval,
        )

    def _default_data_handler(self, data: Dict[str, Any]) -> None:
        """
        Default callback if none is provided: logs the data.
        """
        self.logger.info("[Default Handler] Received data: %s", data)

    def _produce_synthetic_data(self) -> Dict[str, Any]:
        """
        Simulate or fetch streaming data. Replace with actual ingestion logic if needed.
        """
        # Example: random sensor reading
        # You can add more fields (e.g., device_id, location, etc.)
        return {
            "timestamp": time.time(),
            "device_id": f"sensor_{random.randint(1, 5)}",
            "value": round(random.uniform(10.0, 100.0), 2),
        }

    def start(self) -> None:
        """
        Start the background thread that continuously ingests data.
        """
        if self._thread and self._thread.is_alive():
            self.logger.warning("StreamingIngestionService is already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_service_loop, daemon=True)
        self._thread.start()
        self.logger.info("StreamingIngestionService started.")

    def _run_service_loop(self) -> None:
        """
        Main loop that fetches or produces data at intervals until stop is requested.
        """
        last_run = time.time()
        while not self._stop_event.is_set():
            # Control the produce interval precisely:
            now = time.time()
            elapsed = now - last_run
            if elapsed >= self.produce_interval:
                last_run = now
                try:
                    data = self._produce_synthetic_data()
                    # Optionally, place data into an internal queue
                    if not self.data_queue.full():
                        self.data_queue.put(data)
                    else:
                        self.logger.warning("Data queue is full; discarding data.")

                    # Or directly invoke the callback
                    self.data_callback(data)
                except Exception as e:
                    self.logger.error("Error producing/processing data: %s", e, exc_info=True)

            time.sleep(0.1)  # Short sleep to avoid tight CPU loop

    def stop(self) -> None:
        """
        Signal the service to stop, then wait for the thread to join.
        """
        self._stop_event.set()
        if self._thread:
            self._thread.join()
            self._thread = None
        self.logger.info("StreamingIngestionService stopped.")

    def consume_data_queue(self) -> Optional[Dict[str, Any]]:
        """
        Example method if external code wants to pop from the internal queue manually.
        Return None if no data available.
        """
        if self.data_queue.empty():
            return None
        return self.data_queue.get()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def example_callback(data: Dict[str, Any]) -> None:
        print("[Callback] Got new data:", data)

    # Create the streaming service
    sis = StreamingIngestionService(
        produce_interval=2.0,
        data_callback=example_callback
    )

    # Start the streaming ingestion
    sis.start()

    # Let it run for ~6 seconds
    time.sleep(6)

    # Stop the service
    sis.stop()

    # If using an internal queue approach, you can retrieve any leftover data:
    leftover = sis.consume_data_queue()
    while leftover is not None:
        print("[Main] Consumed leftover data from queue:", leftover)
        leftover = sis.consume_data_queue()