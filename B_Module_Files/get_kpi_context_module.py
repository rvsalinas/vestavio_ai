from psycopg2.extras import RealDictCursor
import logging
from B_Module_Files.database_module import get_db_connection

def get_kpi_context(user_id):
    """
    Fetch KPI context for a given user from the database.
    Returns a summary string including:
      - Average Efficiency (%)
      - Cumulative Energy Saved (7-day window)
      - System Health Score
      - Milestone message if applicable
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Fetch latest average efficiency and system health from user_kpi_history
                cursor.execute("""
                    SELECT avg_efficiency, system_health
                    FROM user_kpi_history
                    WHERE user_id = %s
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (user_id,))
                latest_kpi = cursor.fetchone()

                # Calculate cumulative energy saved over the last 7 days
                cursor.execute("""
                    SELECT COALESCE(SUM(energy_saved_increment::float), 0) AS cumulative_energy_saved
                    FROM user_kpi_history
                    WHERE user_id = %s
                      AND timestamp >= NOW() - INTERVAL '7 days'
                """, (user_id,))
                energy_sum = cursor.fetchone()

        if latest_kpi is not None and energy_sum is not None:
            avg_eff = round(float(latest_kpi['avg_efficiency']), 1)
            total_saved = round(float(energy_sum['cumulative_energy_saved']), 1)
            system_health = round(float(latest_kpi['system_health']), 1)

            milestone = ""
            if total_saved > 500:
                milestone = f"Milestone reached: Over {total_saved} units saved."
            elif avg_eff >= 95.0:
                milestone = f"Milestone reached: {avg_eff}% efficiency record."

            return (
                f"Current KPIs → Average Efficiency: {avg_eff}%, "
                f"Energy Saved (7-day): {total_saved} units, "
                f"System Health: {system_health}. {milestone}"
            )
        else:
            logging.info(f"KPI data not available for user {user_id}.")
            return "KPI data is currently unavailable."
    except Exception as e:
        logging.error(f"Error fetching KPI context for user {user_id}: {e}", exc_info=True)
        return "Unable to retrieve KPI context at this time."