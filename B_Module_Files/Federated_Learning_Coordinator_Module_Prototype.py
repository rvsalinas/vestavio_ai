"""
Federated_Learning_Coordinator_Module_Prototype.py

Absolute File Path (example):
    /Users/username/Desktop/energy_optimization_project/B_Module_Files/Federated_Learning_Coordinator_Module_Prototype.py

PURPOSE:
  - Prototype module for coordinating federated learning across multiple participants.
  - Maintains a global model, distributes it, receives local updates, and aggregates them.
  - This skeleton can be extended for more complex federated setups.

NOTES:
  - For real-world usage, consider secure aggregation, partial updates, differential privacy, etc.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, List, Callable, Optional


class FederatedLearningCoordinator:
    """
    A prototype for coordinating federated learning:
      - Initialize or load a global model.
      - Send model to participants.
      - Collect and aggregate updates.
      - Update the global model accordingly.
    """

    def __init__(
        self,
        model_init_fn: Optional[Callable[[], nn.Module]] = None,
        aggregation_method: str = "mean",
    ):
        """
        :param model_init_fn: A callable returning a `torch.nn.Module` instance as the initial global model.
        :param aggregation_method: Method for aggregating participant updates ("mean", "sum", or custom).
        """
        self.logger = logging.getLogger("FederatedLearningCoordinator")
        self.aggregation_method = aggregation_method.lower().strip()
        self.model_init_fn = model_init_fn or self._default_model_init
        self.global_model = self.model_init_fn()

        self.logger.info("[FederatedCoordinator] Initialized. Using aggregation method: %s",
                         self.aggregation_method)

    def _default_model_init(self) -> nn.Module:
        """Default minimal model: a small feed-forward net."""
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        """
        Return the global model's parameters (state_dict).
        Participants can load these parameters locally to continue training.
        """
        return self.global_model.state_dict()

    def receive_updates(self, updates: List[Dict[str, torch.Tensor]]) -> None:
        """
        Receive model updates from participants (e.g., their state_dict after local training),
        then aggregate them into the global model.

        :param updates: A list of parameter dictionaries (each one is a state_dict).
        """
        num_updates = len(updates)
        if num_updates == 0:
            self.logger.warning("[FederatedCoordinator] No participant updates received.")
            return

        self.logger.info("[FederatedCoordinator] Received %d participant updates.", num_updates)

        # --- Initialize an aggregate dictionary matching the shape of the first update ---
        aggregated = {}
        for k, param_tensor in updates[0].items():
            aggregated[k] = torch.zeros_like(param_tensor)

        # --- Sum or accumulate each participant's parameters ---
        for state_dict in updates:
            for k, param_tensor in state_dict.items():
                aggregated[k] += param_tensor

        # --- Perform final aggregation step (e.g. average) ---
        if self.aggregation_method == "mean":
            for k in aggregated:
                aggregated[k] /= float(num_updates)
        elif self.aggregation_method == "sum":
            pass  # Already summed; no further action needed.
        else:
            self.logger.warning("Unknown aggregation method '%s'. Defaulting to 'mean'.", self.aggregation_method)
            for k in aggregated:
                aggregated[k] /= float(num_updates)

        # --- Update global model ---
        self.global_model.load_state_dict(aggregated)
        self.logger.info("[FederatedCoordinator] Global model updated via '%s' aggregation.", self.aggregation_method)

    def run_federated_round(self, participant_updates: List[Dict[str, torch.Tensor]]) -> None:
        """
        Orchestrate one complete round of federated learning:
         - (1) Distribute the global model (participants would do local training offline).
         - (2) Collect participant updates (provided as `participant_updates`).
         - (3) Aggregate these updates.
         - (4) Update the global model.

        This method simulates the entire flow for demonstration.
        """
        # Step 1: Distribute model (in a real scenario, you might send state_dict to each participant).
        # model_params = self.get_global_state()  # (Just conceptual; not used here.)

        # Step 2 & 3: Aggregate updates
        self.receive_updates(participant_updates)

        # Step 4: Global model is now updated
        self.logger.info("[FederatedCoordinator] One federated round is complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage:
    def init_simple_model():
        """A function returning a basic feed-forward net."""
        net = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        return net

    # Create the coordinator
    coordinator = FederatedLearningCoordinator(
        model_init_fn=init_simple_model,
        aggregation_method="mean"
    )

    # Simulate 2 participant updates for demonstration
    global_params = coordinator.get_global_state()
    dummy_update_1 = {k: v + 1.0 for k, v in global_params.items()}
    dummy_update_2 = {k: v + 2.0 for k, v in global_params.items()}

    # Run a federated round with these updates
    coordinator.run_federated_round([dummy_update_1, dummy_update_2])