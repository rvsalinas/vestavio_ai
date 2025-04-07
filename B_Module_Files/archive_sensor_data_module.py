#!/usr/bin/env python
"""
Module: archive_sensor_data_module.py

Purpose:
    Archives old sensor data from the main sensor_data table into an archival table (sensor_data_archive).
    This script is designed to work for all use cases by simply archiving all rows older than a specified threshold.
    
Usage:
    This script can be scheduled (e.g., via cron) to run periodically.
"""

import os
import sys
import psycopg2
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def archive_old_sensor_data():
    try:
        # Build the connection string from environment variables
        DB_HOST = os.getenv("DB_HOST")
        DB_PORT = os.getenv("DB_PORT", "5432")
        DB_NAME = os.getenv("DB_NAME")
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        
        dsn = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"
        logger.info(f"Connecting to database using DSN: {dsn}")

        conn = psycopg2.connect(dsn)
        cursor = conn.cursor()
        
        # Define the archiving criteria; for example, archive data older than 30 days.
        archive_threshold = datetime.utcnow() - timedelta(days=30)
        logger.info(f"Archiving sensor data older than: {archive_threshold}")

        # Insert old sensor data into the archive table and mark them with the archived_at timestamp
        insert_sql = """
            INSERT INTO sensor_data_archive
            SELECT *, NOW() AS archived_at
            FROM sensor_data
            WHERE timestamp < %s
        """
        cursor.execute(insert_sql, (archive_threshold,))
        archived_rows = cursor.rowcount
        logger.info(f"Archived {archived_rows} rows into sensor_data_archive.")

        # Optionally, delete the archived rows from the main table
        delete_sql = """
            DELETE FROM sensor_data
            WHERE timestamp < %s
        """
        cursor.execute(delete_sql, (archive_threshold,))
        deleted_rows = cursor.rowcount
        logger.info(f"Deleted {deleted_rows} rows from sensor_data.")

        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Archiving process completed successfully.")

    except Exception as e:
        logger.error(f"Error during archiving: {e}", exc_info=True)
        # If conn was never created, there's no need to close it.
        try:
            conn.close()
        except Exception:
            pass
        sys.exit(1)

if __name__ == "__main__":
    archive_old_sensor_data()