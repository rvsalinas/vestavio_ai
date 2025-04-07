import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_db_connection():
    """
    Establishes and returns a new database connection using environment variables.
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT")
        )
        return conn
    except Exception as e:
        logger.error(f"Error establishing DB connection: {e}")
        raise

def get_user_alert_preferences(user_id: int) -> dict:
    """
    Retrieves alert preferences for the specified user_id.
    
    :param user_id: The ID of the user.
    :return: A dictionary with the user's alert preferences or None if not found.
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM public.alert_preferences WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            return row
    except Exception as e:
        logger.error(f"Error fetching alert preferences for user {user_id}: {e}")
        raise
    finally:
        conn.close()

def update_user_alert_preferences(user_id: int, preferences: dict, email_alerts: bool = True, sms_alerts: bool = False, phone_number: str = None) -> None:
    """
    Updates or inserts alert preferences for a given user.
    
    :param user_id: The ID of the user.
    :param preferences: A dictionary of additional alert preferences.
    :param email_alerts: Boolean indicating if email alerts are enabled (default True).
    :param sms_alerts: Boolean indicating if SMS alerts are enabled (default False).
    :param phone_number: The user's phone number for SMS alerts.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Check if preferences already exist for the user
            cur.execute("SELECT 1 FROM public.alert_preferences WHERE user_id = %s", (user_id,))
            exists = cur.fetchone()
            preferences_json = json.dumps(preferences)
            if exists:
                cur.execute("""
                    UPDATE public.alert_preferences
                    SET preferences = %s,
                        email_alerts = %s,
                        sms_alerts = %s,
                        phone_number = %s
                    WHERE user_id = %s
                """, (preferences_json, email_alerts, sms_alerts, phone_number, user_id))
            else:
                cur.execute("""
                    INSERT INTO public.alert_preferences (user_id, preferences, email_alerts, sms_alerts, phone_number)
                    VALUES (%s, %s, %s, %s, %s)
                """, (user_id, preferences_json, email_alerts, sms_alerts, phone_number))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error updating alert preferences for user {user_id}: {e}")
        raise
    finally:
        conn.close()

def delete_user_alert_preferences(user_id: int) -> None:
    """
    Deletes the alert preferences for a given user.
    
    :param user_id: The ID of the user.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM public.alert_preferences WHERE user_id = %s", (user_id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error deleting alert preferences for user {user_id}: {e}")
        raise
    finally:
        conn.close()