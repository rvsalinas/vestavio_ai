#!/usr/bin/env python
"""
File: anomaly_maintenance_helper_module.py

Purpose:
  - Provides utility functions to fetch recent anomaly and maintenance information
    for a given user. Fetches data from system_logs (or a relevant table) without
    creating circular imports.

Usage Example:
  from B_Module_Files.anomaly_maintenance_helper_module import (
      fetch_recent_anomalies,
      fetch_predictive_maintenance_status
  )

  anomalies = fetch_recent_anomalies(user_id=22, limit=3)
  maintenance = fetch_predictive_maintenance_status(user_id=22, limit=5)
"""

import logging
from typing import List, Dict, Any
from B_Module_Files.database_module import get_db_connection

logger = logging.getLogger("AnomalyMaintenanceHelper")

def fetch_recent_anomalies(user_id: int, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Fetch recent anomalies for the given user from the system_logs table.
    Each record is a dict: {"message": str, "timestamp": datetime}
    Returns up to 'limit' records.
    """
    anomalies = []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT message, timestamp
                    FROM system_logs
                    WHERE user_id = %s
                      AND log_type ILIKE 'anomaly'
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (user_id, limit)
                )
                rows = cursor.fetchall()
                for r in rows:
                    anomalies.append({
                        "message": r["message"],
                        "timestamp": r["timestamp"]
                    })
    except Exception as e:
        logger.error(f"Error fetching anomalies for user {user_id}: {e}", exc_info=True)
    return anomalies

def fetch_predictive_maintenance_status(user_id: int, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Fetch recent maintenance items for the given user from the system_logs table.
    Each record is a dict: {"message": str, "timestamp": datetime}
    Returns up to 'limit' records.
    """
    maintenance_items = []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT message, timestamp
                    FROM system_logs
                    WHERE user_id = %s
                      AND log_type ILIKE 'maintenance'
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (user_id, limit)
                )
                rows = cursor.fetchall()
                for r in rows:
                    maintenance_items.append({
                        "message": r["message"],
                        "timestamp": r["timestamp"]
                    })
    except Exception as e:
        logger.error(f"Error fetching maintenance info for user {user_id}: {e}", exc_info=True)
    return maintenance_items