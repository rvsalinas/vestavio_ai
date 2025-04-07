"""
task_planning_module.py

Absolute File Path:
    /Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/task_planning_module.py

PURPOSE:
    - Handles generating or scheduling tasks based on certain inputs (e.g. sensor data,
      user goals, system constraints).
    - Potentially uses NLP or heuristic methods to plan tasks (e.g., decide priorities,
      resources, ordering).
    - Integrates with other modules for context or receives context from app.py.

FEATURES:
    - A basic example of a rule/heuristic-based approach.
    - Could be extended to incorporate advanced logic (NLP, LLM-based reasoning, or
      external heuristics like OR-Tools for scheduling).

USAGE EXAMPLE:
    from task_planning_module import TaskPlanningModule

    planner = TaskPlanningModule()
    context = {
        "temperature": 80,         # F
        "humidity": 25,            # %
        "requested_tasks": ["start_ventilation"]
    }
    task_list = planner.plan_tasks(context)
    for t in task_list:
        print(t)

    # Potential output:
    # [
    #   {"task_name": "Cool Down System", "priority": "HIGH", ...},
    #   {"task_name": "Increase Humidity", "priority": "MEDIUM", ...},
    #   {"task_name": "Start Ventilation", "priority": "MEDIUM", ...}
    # ]
"""

import logging
from typing import List, Dict, Any, Optional


class TaskPlanningModule:
    """
    A module that handles generating or scheduling tasks based on certain inputs
    (such as sensor data, user goals, or system constraints).

    Potential expansions:
      - NLP-based interpretation of user goals or natural-language instructions.
      - Integration with advanced solvers or orchestrators (e.g., OR-Tools).
      - Communication with external modules for reasoning or multi-step workflows.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        :param logger: Optional logger instance. If None, use root or basic logger.
        """
        if logger is None:
            self.logger = logging.getLogger("TaskPlanningModule")
            if not self.logger.handlers:
                console = logging.StreamHandler()
                console.setLevel(logging.INFO)
                self.logger.addHandler(console)
                self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger

        self.logger.info("Initializing TaskPlanningModule...")

    def plan_tasks(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Given a context dictionary (could contain sensor data, user goals, etc.),
        produce a list of tasks or an action plan.

        :param context: dictionary with relevant keys (e.g., 'temperature', 'humidity',
                        'requested_tasks', 'priority_overrides', etc.)
        :return: a list of tasks, each a dictionary, e.g.:
                 {
                     "task_name": "<some name>",
                     "priority": "<HIGH|MEDIUM|LOW>",
                     "details": "<explanation or reason>",
                 }
        """
        self.logger.info("Planning tasks based on context...")

        tasks = []

        # 1) Example rule: If temperature > 75 F, schedule a "Cool Down System" task
        temperature = context.get("temperature")
        if temperature is not None and temperature > 75:
            tasks.append({
                "task_name": "Cool Down System",
                "priority": "HIGH",
                "details": f"Temperature above 75 F ({temperature} F)."
            })

        # 2) Example rule: If humidity < 30%, schedule "Increase Humidity"
        humidity = context.get("humidity")
        if humidity is not None and humidity < 30:
            tasks.append({
                "task_name": "Increase Humidity",
                "priority": "MEDIUM",
                "details": f"Humidity below 30% ({humidity}%)."
            })

        # 3) Check if user explicitly requested tasks in context
        requested_tasks = context.get("requested_tasks", [])
        for req in requested_tasks:
            # Simple mapping or direct acceptance. Here we just turn them into tasks.
            tasks.append({
                "task_name": req.capitalize().replace("_", " "),
                "priority": "MEDIUM",
                "details": f"User-requested task: {req}"
            })

        self.logger.info(f"Planned {len(tasks)} tasks.")
        return tasks


if __name__ == "__main__":
    # Minimal usage demo
    planner = TaskPlanningModule()
    sample_context = {
        "temperature": 80,
        "humidity": 25,
        "requested_tasks": ["start_ventilation"]
    }
    plan = planner.plan_tasks(sample_context)
    print("Planned tasks:")
    for item in plan:
        print(item)