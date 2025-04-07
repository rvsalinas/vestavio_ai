#!/usr/bin/env python3
"""
File: seed_demo.py
Location: A_Core_App_Files/seed_demo.py
Purpose:
  - Create/update a "demo" user (demo@vestavio.com).
  - Seed camera feed placeholders, KPI history, and system logs for a 4-DOF (drone) demo.
  - Then, continuously insert new 4-DOF sensor data every 60 seconds for 4 drones so we can test multiple robots at once.
    (Includes dof_1..dof_4, vel_1..vel_4, energy_efficiency, energy_saved, pm_risk).
"""

import os
import time
import bcrypt
import random
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta, time as dtime
from dotenv import load_dotenv

###############################################################################
# Database Connection
###############################################################################
def get_db_connection():
    load_dotenv()  # Load .env if present

    db_host = os.getenv('DB_HOST', 'vestavio.cpq6m8ka4a1l.us-east-2.rds.amazonaws.com')
    db_port = os.getenv('DB_PORT', '5432')
    db_user = os.getenv('DB_USER', 'rvsalinas')
    db_password = os.getenv('DB_PASSWORD', 'ChangeMe!')
    db_name = os.getenv('DB_NAME', 'energy_optimization_db')

    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        dbname=db_name
    )
    return conn

###############################################################################
# Create or Get Demo User
###############################################################################
def create_or_get_demo_user(conn):
    demo_email = "demo@vestavio.com"
    demo_password = "demo123"

    hashed_password = bcrypt.hashpw(
        demo_password.encode('utf-8'),
        bcrypt.gensalt()
    ).decode('utf-8')

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute("""
            SELECT id FROM users
            WHERE email = %s
            LIMIT 1
        """, (demo_email,))
        row = cursor.fetchone()

        if row:
            print("Demo user already exists.")
            return row['id']
        else:
            cursor.execute("""
                INSERT INTO users (username, password_hash, email)
                VALUES (%s, %s, %s)
                RETURNING id
            """, (demo_email, hashed_password, demo_email))
            new_id = cursor.fetchone()['id']
            conn.commit()
            print(f"Created new demo user with id={new_id}")
            return new_id

###############################################################################
# Seed Chat Logs (Removed Seed Text)
###############################################################################
def seed_chat_logs(conn, user_id):
    # Removed sample chat messages; user will add these manually.
    print("Skipping seeding chat_logs; user will add these manually.")

###############################################################################
# Seed KPI History (For Dashboard Charts - Cumulative Energy Saved)
###############################################################################
def seed_kpi_history(conn, user_id):
    with conn.cursor() as cursor:
        base_time = datetime.now()
        cumulative_energy = 0.0
        for i in range(14):
            current_time = base_time - timedelta(days=i)
            avg_eff = 80 + i * 0.5 + random.uniform(-1, 2)
            energy_saved_increment = 20 + random.uniform(-10, 10)
            cumulative_energy += energy_saved_increment
            system_health = 70 + i * 0.3 + random.uniform(-2, 3)
            cursor.execute("""
                INSERT INTO user_kpi_history (user_id, timestamp, avg_efficiency, energy_saved, energy_saved_increment, system_health)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (user_id, current_time, avg_eff, cumulative_energy, energy_saved_increment, system_health))
        conn.commit()
    print("Seeded user_kpi_history with 14 days of random drone KPI data.")

###############################################################################
# Seed Action Logs (Removed Seed Text)
###############################################################################
def seed_action_logs(conn, user_id):
    print("Skipping seeding user_actions; user will add these manually.")

###############################################################################
# Seed System Logs
###############################################################################
def seed_system_logs(conn, user_id):
    now = datetime.now()
    logs = [
        (now - timedelta(hours=3), "INFO", "Drone environment loaded successfully."),
        (now - timedelta(hours=1), "WARNING", "Motor temperature spiked briefly."),
        (now - timedelta(minutes=20), "ERROR", "Minor collision detected near rotor 2.")
    ]
    with conn.cursor() as cursor:
        for ts, log_type, message in logs:
            cursor.execute("""
                INSERT INTO system_logs (user_id, timestamp, log_type, message, is_resolved, notes)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (user_id, ts, log_type, message, False, "Demo drone log entry"))
        conn.commit()
    print("Seeded system_logs with drone-specific entries.")

###############################################################################
# Stream 4DOF Sensor Data in Real Time (Every 60 seconds) for 4 Drones
###############################################################################
def stream_4dof_sensor_data(conn, user_id):
    print("Starting real-time 4DOF sensor data stream for 4 drones (updates every 60s). Press Ctrl+C to stop.")
    try:
        while True:
            current_time = datetime.utcnow()
            for robot in range(1, 5):
                robot_time = current_time + timedelta(microseconds=robot * 10)
                dofs = [0.10 + random.uniform(-0.02, 0.02) for _ in range(4)]
                vels = [0.02 + random.uniform(-0.01, 0.01) for _ in range(4)]
                eff = 80 + random.uniform(-2, 2)
                saved = 200 + random.uniform(-10, 10)
                pm_risk = 30 + random.uniform(-5, 5)
                with conn.cursor() as cursor:
                    for i in range(4):
                        # Add a microsecond offset for uniqueness
                        offset = timedelta(microseconds=i*5)
                        cursor.execute("""
                            INSERT INTO sensor_data (sensor_name, sensor_output, status, timestamp, user_id, robot_number)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (f"dof_{i+1}", str(dofs[i]), "Operational", robot_time + offset, user_id, robot))
                        cursor.execute("""
                            INSERT INTO sensor_data (sensor_name, sensor_output, status, timestamp, user_id, robot_number)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (f"vel_{i+1}", str(vels[i]), "Operational", robot_time + offset, user_id, robot))
                    cursor.execute("""
                        INSERT INTO sensor_data (sensor_name, sensor_output, status, timestamp, user_id, robot_number)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, ("energy_efficiency", str(eff), "Operational", robot_time, user_id, robot))
                    cursor.execute("""
                        INSERT INTO sensor_data (sensor_name, sensor_output, status, timestamp, user_id, robot_number)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, ("energy_saved", str(saved), "Operational", robot_time, user_id, robot))
                    cursor.execute("""
                        INSERT INTO sensor_data (sensor_name, sensor_output, status, timestamp, user_id, robot_number)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, ("pm_risk", str(pm_risk), "Operational", robot_time, user_id, robot))
            conn.commit()
            print(f"[{datetime.now()}] Inserted new 4DOF sensor data rows for 4 drones.")
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nStopping real-time data stream (Ctrl+C).")
    except Exception as e:
        print(f"\nError streaming data: {e}")
        conn.rollback()

###############################################################################
# Seed RL Predict
###############################################################################
def seed_rl_predict_data(conn, user_id):
    """
    Seeds sensor_data records for the RL predict endpoint to reflect a drone use case.
    Inserts one set of 4 DOF and 4 velocity readings for robot 1 with robot_type set to 'drone'.
    """
    current_time = datetime.utcnow()
    sensors_values = {
        "dof_1": str(0.12 + random.uniform(-0.01, 0.01)),
        "dof_2": str(0.11 + random.uniform(-0.01, 0.01)),
        "dof_3": str(0.13 + random.uniform(-0.01, 0.01)),
        "dof_4": str(0.10 + random.uniform(-0.01, 0.01)),
        "vel_1": str(0.02 + random.uniform(-0.005, 0.005)),
        "vel_2": str(0.03 + random.uniform(-0.005, 0.005)),
        "vel_3": str(0.025 + random.uniform(-0.005, 0.005)),
        "vel_4": str(0.015 + random.uniform(-0.005, 0.005))
    }
    with conn.cursor() as cursor:
        for sensor, value in sensors_values.items():
            cursor.execute("""
                INSERT INTO sensor_data (sensor_name, sensor_output, status, timestamp, user_id, robot_number, robot_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (sensor, value, "Operational", current_time, user_id, 1, "drone"))
        conn.commit()
    print("Seeded RL predict sensor data for drone use case.")

###############################################################################
# Main
###############################################################################
def main():
    conn = get_db_connection()
    try:
        demo_user_id = create_or_get_demo_user(conn)
        # One-time seeds
        seed_chat_logs(conn, demo_user_id)
        seed_kpi_history(conn, demo_user_id)
        seed_action_logs(conn, demo_user_id)
        seed_system_logs(conn, demo_user_id)
        seed_rl_predict_data(conn, demo_user_id)
        
        print("One-time seeding complete. Now streaming 4DOF sensor data for 4 drones every 60s...")
        stream_4dof_sensor_data(conn, demo_user_id)
    except Exception as e:
        print(f"Error in seed_demo script: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    main()