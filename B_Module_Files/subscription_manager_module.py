#!/usr/bin/env python
"""
Module: subscription_manager_module.py

Handles subscription logic, such as:
  - Setting the user's subscription plan.
  - Updating the user's max_robots in the 'users' table.
  - Checking if a user can register an additional robot.
"""

import logging
import json
from typing import Optional, Dict

# If you have a shared get_db_connection function in B_Module_Files/database_module
# you can import it:
from B_Module_Files.database_module import get_db_connection

class SubscriptionManager:
    """
    Manages subscription plans for each user, storing and retrieving
    the max_robots limit in the users table.
    """

    # Example plan dictionary: plan_name -> max_robots
    # Adjust or expand as needed.
    PLAN_LIMITS = {
        "free": 1,
        "2robots": 2,
        "3robots": 3,
        "4robots": 4,
        "5plus": 999999  # or treat 5plus as "unlimited"
    }

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        :param logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger("SubscriptionManager")
        if not self.logger.handlers:
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            self.logger.addHandler(console)
            self.logger.setLevel(logging.INFO)

    def set_user_plan(self, user_id: int, plan_name: str) -> bool:
        """
        Sets the user's plan_name, updating max_robots in the 'users' table.
        :param user_id: The user's ID
        :param plan_name: e.g. "free", "2robots", "3robots", "4robots", "5plus"
        :return: True if update succeeded, False otherwise
        """
        if plan_name not in self.PLAN_LIMITS:
            self.logger.warning(f"Unknown plan_name '{plan_name}' for user {user_id}.")
            return False

        max_robots = self.PLAN_LIMITS[plan_name]

        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE users
                        SET max_robots = %s
                        WHERE id = %s
                    """, (max_robots, user_id))
                conn.commit()
            self.logger.info(f"User {user_id} plan set to '{plan_name}' => max_robots={max_robots}.")
            return True
        except Exception as e:
            self.logger.error(f"Error setting plan for user {user_id}: {e}", exc_info=True)
            return False

    def get_user_plan_info(self, user_id: int) -> Dict[str, int]:
        """
        Returns the user's current max_robots and possibly the plan_name (inferred).
        :param user_id: The user's ID
        :return: dict with { "max_robots": int, "plan_name": str } or empty dict on error
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT max_robots
                        FROM users
                        WHERE id = %s
                        LIMIT 1
                    """, (user_id,))
                    row = cursor.fetchone()
            if not row:
                self.logger.warning(f"No user found with id={user_id}.")
                return {}

            max_robots = row[0]
            # Try to infer plan_name by reverse lookup in PLAN_LIMITS
            plan_name = None
            for k, v in self.PLAN_LIMITS.items():
                if v == max_robots:
                    plan_name = k
                    break
            if not plan_name:
                # If it doesn't match exactly, set plan_name to "custom"
                plan_name = "custom"

            return {"max_robots": max_robots, "plan_name": plan_name}

        except Exception as e:
            self.logger.error(f"Error getting plan info for user {user_id}: {e}", exc_info=True)
            return {}

    def can_add_robot(self, user_id: int) -> bool:
        """
        Checks if the user can register another robot under their current max_robots plan.
        :param user_id: The user's ID
        :return: True if user can add another robot, False otherwise
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # 1) Fetch max_robots
                    cursor.execute("SELECT max_robots FROM users WHERE id = %s", (user_id,))
                    user_row = cursor.fetchone()
                    if not user_row:
                        self.logger.warning(f"No user found with id={user_id}.")
                        return False
                    max_robots = user_row[0]

                    # 2) Count how many robots the user already has
                    cursor.execute("""
                        SELECT COUNT(*) AS robot_count
                        FROM user_robots
                        WHERE user_id = %s
                    """, (user_id,))
                    count_row = cursor.fetchone()
                    current_robot_count = count_row[0]

            # 3) Compare current_robot_count < max_robots
            return current_robot_count < max_robots

        except Exception as e:
            self.logger.error(f"Error checking can_add_robot for user {user_id}: {e}", exc_info=True)
            return False


if __name__ == "__main__":
    # Quick test usage
    logging.basicConfig(level=logging.INFO)
    sub_manager = SubscriptionManager()

    # Example: set plan for user_id=22
    sub_manager.set_user_plan(22, "3robots")

    # Check plan info
    info = sub_manager.get_user_plan_info(22)
    print("User plan info:", info)

    # See if user can add a new robot
    can_add = sub_manager.can_add_robot(22)
    print("Can add robot?", can_add)