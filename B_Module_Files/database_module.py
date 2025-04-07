"""
File: database_module.py
Location: B_Module_Files/database_module.py

Purpose:
  - Manages a connection pool to the PostgreSQL database using psycopg2.
  - Provides a context manager (get_db_connection) for convenient 'with' usage.

Usage Example:
  from B_Module_Files.database_module import get_db_connection

  with get_db_connection() as conn:
      with conn.cursor() as cursor:
          cursor.execute("SELECT 1;")
          result = cursor.fetchone()
          print(result)
"""

import os
import logging
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# Read DB credentials from environment variables (no placeholders)
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# Global connection pool reference
db_pool = None

# ------------------------------------------------------------------------------
# Initialize the connection pool
# ------------------------------------------------------------------------------
try:
    db_pool = pool.SimpleConnectionPool(
        1,          # minconn
        20,         # maxconn
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        cursor_factory=RealDictCursor
    )
    logger.info("Database connection pool created successfully in database_module.")
except Exception as e:
    logger.error(f"Error creating database connection pool: {e}", exc_info=True)
    raise e

# ------------------------------------------------------------------------------
# Context Manager for obtaining and releasing a DB connection
# ------------------------------------------------------------------------------
class ConnectionContext:
    def __enter__(self):
        global db_pool
        if db_pool is None:
            logger.error("Database connection pool is not initialized.")
            raise RuntimeError("DB connection pool not initialized.")

        try:
            self.conn = db_pool.getconn()
            if not self.conn:
                logger.error("Failed to retrieve a connection from the pool.")
                raise RuntimeError("No available DB connection from the pool.")
            return self.conn
        except Exception as ex:
            logger.error(f"DB connection error in database_module: {ex}", exc_info=True)
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Always release the connection back to the pool
        global db_pool
        if self.conn and db_pool:
            db_pool.putconn(self.conn)
            self.conn = None

        # If an exception occurred inside the 'with' block, log it
        if exc_type:
            logger.error(f"Error in ConnectionContext: {exc_val}", exc_info=True)
        # Returning False will propagate the exception if any
        return False

# ------------------------------------------------------------------------------
# Public function to obtain a connection via 'with get_db_connection() as conn:'
# ------------------------------------------------------------------------------
def get_db_connection():
    """
    Use this function in a 'with' statement to get a psycopg2 connection from the pool.
    Example:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1;")
                result = cursor.fetchone()
                print(result)
    """
    return ConnectionContext()

# ------------------------------------------------------------------------------
# (Optional) Utility to close all connections if needed during shutdown
# ------------------------------------------------------------------------------
def close_all_connections():
    global db_pool
    if db_pool:
        db_pool.closeall()
        db_pool = None
        logger.info("All DB connections closed.")