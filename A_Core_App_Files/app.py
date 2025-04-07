# ------------------------------------------------------------------------------
# File: app.py
# Absolute File Path: /Users/robertsalinas/Desktop/energy_optimization_project/A_Core_App_Files/app.py
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
import os
# Comment this line out if you want to use the GPU:
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

import logging
import datetime
import json
import numpy as np
import requests
import bcrypt
import jwt
import tensorflow as tf
import subprocess
import time
import joblib
import pytz
import openai
import io
import sys
sys.path.insert(0, '/home/ec2-user/energy_optimization_project')
sys.path.insert(0, '/home/ec2-user/energy_optimization_project/G_Genesis_Files/Genesis_main/genesis')
import stripe

from dotenv import load_dotenv
load_dotenv(dotenv_path="/home/ec2-user/energy_optimization_project/.env", override=True)

# GPU Memory Control
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable dynamic memory allocation
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory configuration set successfully.")
    except RuntimeError as e:
        print(f"Error setting up GPU memory configuration: {e}")

from functools import wraps
from flask import Flask, request, jsonify, abort, render_template, redirect, url_for, session, g, flash, current_app, send_from_directory
from flask_swagger_ui import get_swaggerui_blueprint
from logging.handlers import RotatingFileHandler
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin
from flask_jwt_extended import JWTManager, create_access_token
from contextlib import contextmanager
from datetime import timedelta, datetime
from stable_baselines3 import PPO
from PIL import Image

APP_START_TIME = time.time()
TOTAL_REQUESTS = 0
TOTAL_REQUEST_DURATION = 0.0

REQUIRED_FEATURE_LIST = 18

# ------------------------------------------------------------------------------
# Load Environment Varibales
# ------------------------------------------------------------------------------
load_dotenv(override=True)

app = Flask(
    __name__,
    template_folder='../templates',  # one level up from A_Core_App_Files
    static_folder='static'          # static is now inside A_Core_App_Files
)
app.secret_key = os.getenv('SECRET_KEY', 'your-very-secure-secret-key')
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

@app.before_request
def start_timer():
    """Mark the time the request started."""
    g.request_start_time = time.time()

@app.after_request
def log_request_duration(response):
    """
    Increments the TOTAL_REQUESTS and total duration for each request.
    """
    global TOTAL_REQUESTS, TOTAL_REQUEST_DURATION
    elapsed = time.time() - g.request_start_time
    TOTAL_REQUESTS += 1
    TOTAL_REQUEST_DURATION += elapsed
    return response

# ------------------------------------------------------------------------------
# Logging Setup (Rotating)
# ------------------------------------------------------------------------------
log_dir = os.getenv('LOG_DIR', 'F_Log_Files')  # Fallback is 'F_Log_Files' if none in .env
log_path = os.path.join(log_dir, 'flask.log')
handler = RotatingFileHandler(log_path, maxBytes=10_000, backupCount=5)

# Read LOG_LEVEL from environment, default to INFO:
log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
supported_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
if log_level_str not in supported_levels:
    log_level_str = 'INFO'  # fallback

handler.setLevel(log_level_str)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)

# Also attach handler to the root logger to capture everything
root_logger = logging.getLogger()
root_logger.addHandler(handler)
root_logger.setLevel(log_level_str)

app.logger.propagate = True

app.logger.info(f"Log level set to {log_level_str}.")

# ------------------------------------------------------------------------------
# Database Configuration
# ------------------------------------------------------------------------------
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

try:
    db_pool = pool.SimpleConnectionPool(
        1, 20,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        cursor_factory=RealDictCursor
    )
    app.logger.info("Database connection pool created successfully.")
except Exception as e:
    app.logger.error(f"Error creating database connection pool: {e}", exc_info=True)
    abort(500, description="Database connection pool creation failed.")

@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = db_pool.getconn()
        if conn:
            yield conn
    except Exception as e:
        app.logger.error(f"DB connection error: {e}", exc_info=True)
        raise e
    finally:
        if conn:
            db_pool.putconn(conn)

# ------------------------------------------------------------------------------
# Importing modules
# ------------------------------------------------------------------------------

# 1) Agents & Guardrails
from B_Module_Files.agent_guardrails_module import AgentGuardrails, apply_guardrails
from B_Module_Files.agents_config_module import (
    energy_efficiency_agent,
    predictive_maintenance_agent,
    alert_notification_agent,
    subscription_upgrade_agent,
    guardrail_policy_agent,
    dashboard_agent,
    visual_agent
)
from B_Module_Files.agents_runner_module import run_sync_request

# 2) Anomaly & Subscription
from B_Module_Files.anomaly_maintenance_helper_module import fetch_recent_anomalies, fetch_predictive_maintenance_status
from B_Module_Files.anomaly_severity_module import get_anomaly_severity
from B_Module_Files.subscription_manager_module import SubscriptionManager

# 3) Collaboration & Environment
from B_Module_Files.collaborative_learning_module import CollaborativeLearningModule
from B_Module_Files.environmental_context_awareness_module import EnvironmentalContextAwareness

# 4) Computer Vision
from B_Module_Files.coco_detection_module import CocoDetectionModule
from B_Module_Files.imagenet_module import ImageNetModule
from B_Module_Files.object_detection_module import ObjectDetectionModule
from B_Module_Files.vision_transformer_module import VisionTransformerModule

# 5) Data & Model
from B_Module_Files.data_preprocessing_module import DataPreprocessor
from B_Module_Files.Data_Fusion_Gateway_Module import DataFusionGateway
from B_Module_Files.Model_Registry_Integration_Module import ModelRegistryIntegration
# from B_Module_Files.multi_model_loader_module import MODELS (add later)

# 6) Database & Security
from B_Module_Files.database_module import (
    DB_HOST,
    DB_PORT,
    DB_USER,
    DB_PASSWORD,
    DB_NAME,
    get_db_connection
)
from B_Module_Files.security_layer_module import SecurityLayer

# 7) Function Schemas & Structured Data
from B_Module_Files.function_schemas_module import get_all_tools
from B_Module_Files.structured_data_schema_module import (
    MathReasoningSchema,
    UIElement,
    ModerationSchema,
    ActionLogEntry
)

# 8) Natural Language & Use Cases
from B_Module_Files.natural_language_module import NaturalLanguageModule
from B_Module_Files.use_case_selector_module import UseCaseSelector

# 9) Performance, Planning, & System
from B_Module_Files.Performance_Monitoring_Module import PerformanceMonitoringModule
from B_Module_Files.task_planning_module import TaskPlanningModule
from B_Module_Files.system_decision_cycle import system_decision_cycle
from B_Module_Files.get_kpi_context_module import get_kpi_context

# 10) The updated Genesis usage
# from B_Module_Files.genesis_module import run_genesis_simulation

# ------------------------------------------------------------------------------
# Importing models (joblib / Keras / PPO) with dynamic use-case selection
# Now includes both "drone" (4 DOF) and "urban" (5 DOF) scenarios, alongside 6dof, 9dof, and warehouse.
# ------------------------------------------------------------------------------
MODEL_DIR = os.path.expanduser(os.getenv('MODEL_DIR', '~/energy_optimization_project/C_Model_Files'))
from joblib import load
from B_Module_Files.use_case_selector_module import load_models_for_use_case

# Determine the use case (drone, urban, 6dof, 9dof, or warehouse), defaulting to 'warehouse'.
use_case = os.getenv('USE_CASE', 'warehouse').lower()
app.logger.info(f"Detected use case: {use_case}")

model_paths = load_models_for_use_case(use_case)
app.logger.info(f"Model paths for use case {use_case}: {model_paths}")

# 1) Anomaly Detection Model
if "anomaly" in model_paths:
    anomaly_path = model_paths["anomaly"]
    if anomaly_path is not None:
        try:
            anomaly_detection_model = load(anomaly_path)
            # Handle older joblib exports that store sub-models in a dict
            if isinstance(anomaly_detection_model, dict):
                if "xgb_model" in anomaly_detection_model:
                    anomaly_detection_model = anomaly_detection_model["xgb_model"]
                elif "model" in anomaly_detection_model:
                    anomaly_detection_model = anomaly_detection_model["model"]
            app.logger.info(
                f"{use_case.upper()} anomaly detection model loaded successfully. "
                f"Model type: {type(anomaly_detection_model)}"
            )
        except Exception as e:
            app.logger.error(
                f"Error loading anomaly detection model from {anomaly_path}: {e}",
                exc_info=True
            )
            anomaly_detection_model = None
    else:
        # If use_case == 'urban', we intentionally skip anomaly detection
        if use_case == "urban":
            app.logger.info("Anomaly detection is intentionally disabled for the URBAN use case.")
        else:
            app.logger.error(f"Anomaly detection model path not found for use case: {use_case}")
        anomaly_detection_model = None
else:
    app.logger.error("No 'anomaly' key found in model_paths registry!")
    anomaly_detection_model = None

# 2) Energy Efficiency Optimization Model & Scaler
if "energy_efficiency" in model_paths and model_paths["energy_efficiency"]:
    EE_SCALER_PATH = os.path.join(
        MODEL_DIR,
        "energy_efficiency_optimization_model",
        f"scaler_energy_efficiency_{use_case}.joblib"
    )
    EE_MODEL_PATH = model_paths["energy_efficiency"]
    try:
        ee_scaler = load(EE_SCALER_PATH)
        ee_model = load(EE_MODEL_PATH)
        app.logger.info(f"{use_case.upper()} energy efficiency optimization model and scaler loaded successfully.")
    except Exception as e:
        app.logger.error(
            f"Error loading energy efficiency model/scaler for {use_case}: {e}",
            exc_info=True
        )
        ee_scaler, ee_model = None, None
else:
    app.logger.error(f"Energy efficiency model path not found for use case: {use_case}")
    ee_scaler, ee_model = None, None

# 3) Predictive Maintenance Model & Scaler
if "predictive_maintenance" in model_paths and model_paths["predictive_maintenance"]:
    PM_SCALER_PATH = os.path.join(
        MODEL_DIR,
        "predictive_maintenance_model",
        f"scaler_predictive_maintenance_{use_case}.joblib"
    )
    PM_MODEL_PATH = model_paths["predictive_maintenance"]
    try:
        pm_transformer = load(PM_SCALER_PATH)
        pm_model = load(PM_MODEL_PATH)
        app.logger.info(f"Predictive maintenance {use_case.upper()} model and scaler loaded successfully.")
    except Exception as e:
        app.logger.error(
            f"Error loading predictive maintenance model/scaler for {use_case}: {e}",
            exc_info=True
        )
        pm_transformer, pm_model = None, None
else:
    app.logger.error(f"Predictive maintenance model path not found for use case: {use_case}")
    pm_transformer, pm_model = None, None

# 4) RL Model (PPO) - Load based on use case
urban_flag = (use_case == "urban")
if urban_flag:
    RL_MODEL_PATH = os.path.join(MODEL_DIR, "reinforcement_learning_model", "ppo_urban_model.zip")
else:
    RL_MODEL_PATH = model_paths.get("rl")

try:
    from stable_baselines3 import PPO
    rl_model = PPO.load(RL_MODEL_PATH, device="cpu")
    app.logger.info(f"{use_case.upper()} RL model loaded successfully (CPU forced).")
except Exception as e:
    app.logger.error(f"Error loading RL model from {RL_MODEL_PATH}: {e}", exc_info=True)
    rl_model = None

# 5) Sensor Fusion Model (common to all use cases)
SENSOR_FUSION_SCALER_PATH = os.path.join(MODEL_DIR, "sensor_fusion_model", "scaler_sensor_fusion_model.joblib")
SENSOR_FUSION_MODEL_PATH = os.path.join(MODEL_DIR, "sensor_fusion_model", "sensor_fusion_model.keras")
try:
    sensor_fusion_scaler = load(SENSOR_FUSION_SCALER_PATH)
    sensor_fusion_model = tf.keras.models.load_model(SENSOR_FUSION_MODEL_PATH)
    app.logger.info("Sensor Fusion model and scaler loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading Sensor Fusion model/scaler: {e}", exc_info=True)
    sensor_fusion_scaler, sensor_fusion_model = None, None

# 6) Real-Time Decision-Making (common)
REAL_TIME_DECISION_SCALER_PATH = os.path.join(
    MODEL_DIR,
    "real_time_decision_making_model",
    "scaler_real_time_decision_making_model.joblib"
)
REAL_TIME_DECISION_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "real_time_decision_making_model",
    "real_time_decision_making_model.keras"
)
try:
    real_time_decision_scaler = load(REAL_TIME_DECISION_SCALER_PATH)
    real_time_decision_model = tf.keras.models.load_model(REAL_TIME_DECISION_MODEL_PATH)
    app.logger.info("Real-Time Decision-Making model and scaler loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading Real-Time Decision-Making model/scaler: {e}", exc_info=True)
    real_time_decision_scaler, real_time_decision_model = None, None

# 7) Visual Model (common)
VISUAL_MODEL_PATH = os.path.join(MODEL_DIR, "visual_model", "visual_model.keras")
try:
    visual_model = tf.keras.models.load_model(VISUAL_MODEL_PATH)
    app.logger.info("Visual model loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading Visual model: {e}", exc_info=True)
    visual_model = None

# Finally, instantiate the DataPreprocessor (shared across all use cases)
preprocessor = DataPreprocessor(
    numeric_strategy="standard",
    cat_strategy="none",
    missing_value_strategy="special_fallback",
    outlier_capping_enabled=False
)

# ------------------------------------------------------------------------------
# Security & Login
# ------------------------------------------------------------------------------
security = SecurityLayer(encryption_key=os.getenv('ENCRYPTION_KEY'))

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # First, try to get token from the Authorization header
        if "Authorization" in request.headers:
            parts = request.headers["Authorization"].split()
            if len(parts) == 2 and parts[0] == "Bearer":
                token = parts[1]
        # If no token in headers, try getting it from cookies
        if not token:
            token = request.cookies.get('access_token')
        if not token:
            app.logger.warning("Token is missing.")
            return jsonify({"message": "Token is missing!"}), 401

        try:
            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            # Set the decoded token payload to the global context
            g.current_user = data
        except jwt.ExpiredSignatureError:
            app.logger.warning("Token has expired!")
            return jsonify({"message": "Token has expired!"}), 401
        except jwt.InvalidTokenError:
            app.logger.warning("Invalid token!")
            return jsonify({"message": "Invalid token!"}), 401

        return f(*args, **kwargs)
    return decorated

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username=None, email=None):
        self.id = id
        self.username = username
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# ------------------------------------------------------------------------------
# Advanced Features
# ------------------------------------------------------------------------------
app.logger.info("HELLO FROM THE CUSTOM A_Core_App_Files/app.py -- verifying we’re in the correct file!")

data_fusion_gateway = DataFusionGateway()
performance_monitor = PerformanceMonitoringModule()
model_registry = ModelRegistryIntegration()

# ------------------------------------------------------------------------------
# Validate Sensor Data
# ------------------------------------------------------------------------------
def validate_sensor_data(sensor_name, sensor_output):
    try:
        name_lower = sensor_name.lower()

        # Allow 'Type' as text
        if 'type' in name_lower:
            return True

        val = float(sensor_output)  # only for numeric sensors

        if 'temperature' in name_lower:
            if '[k]' in name_lower:
                # Kelvin range (roughly 0–100 °C => 273–373 K)
                return 273 <= val <= 373
            else:
                # Assume Fahrenheit in the range [32..500]
                return 32 <= val <= 500

        elif 'humidity' in name_lower:
            return 0 <= val <= 100
        elif 'pressure' in name_lower:
            return 300 <= val <= 1100

        return True

    except ValueError:
        return False

# ------------------------------------------------------------------------------
# Sendgrid Email
# ------------------------------------------------------------------------------
def send_email(to_email, subject, body):
    """
    Sends an email using SendGrid.
    Reads SENDGRID_API_KEY and FROM_EMAIL from environment variables.
    """
    import os
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail

    sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
    from_email = os.getenv("FROM_EMAIL", "no-reply@example.com")

    if not sendgrid_api_key or not from_email:
        app.logger.warning("SendGrid credentials not set. Logging email instead.")
        app.logger.info(f"Email to {to_email}: {subject} - {body}")
        return

    try:
        message = Mail(
            from_email=from_email,
            to_emails=to_email,
            subject=subject,
            plain_text_content=body
        )

        sg = SendGridAPIClient(sendgrid_api_key)
        response = sg.send(message)
        app.logger.info(
            f"Email sent to {to_email}. Status code: {response.status_code}"
        )
    except Exception as e:
        app.logger.error(f"Failed to send email via SendGrid: {e}", exc_info=True)

# ------------------------------------------------------------------------------
# Run Genesis for Experiments
# ------------------------------------------------------------------------------
'''
def run_genesis_subprocess(data):
    """
    Spawns the 'genesis_sim_driver.py' script in a separate process,
    passing 'data' (a Python dict) as a JSON argument.
    """
    payload_str = json.dumps(data)
    driver_script_path = os.getenv(
    "SIM_DRIVER_PATH",
    "/home/ec2-user/energy_optimization_project/G_Genesis_Files/experiments/genesis_sim_driver.py"
)

    try:
        proc = subprocess.Popen(
            ["python3", driver_script_path, payload_str],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = proc.communicate()
        
        if proc.returncode != 0:
            return {
                "status": "error",
                "message": f"Driver script failed. Return code={proc.returncode}, stderr={stderr.decode().strip()}"
            }
        
        output_str = stdout.decode().strip()
        try:
            result_data = json.loads(output_str)
            return result_data
        except json.JSONDecodeError:
            return {
                "status": "error",
                "message": f"Driver script returned invalid JSON: {output_str}"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Exception calling driver script: {str(e)}"
        }
'''
################################################################################
# ------------------------------------------------------------------------------
# Endpoints Begin
# ------------------------------------------------------------------------------
################################################################################

# ------------------------------------------------------------------------------
# Health and Test Endpoints
# ------------------------------------------------------------------------------
@app.route('/health', methods=['GET', 'POST'])
def health_check():
    return jsonify({"status": "Server is running"}), 200

@app.route('/test_logging', methods=['GET'])
def test_logging():
    for i in range(2000):
        app.logger.info(f"Rotating File Test: line #{i}")
    return jsonify({"message": "Log test complete."}), 200

@app.route('/test_email')
def test_email():
    try:
        send_email("rsalinas@vestavio.com", "Test Subject", "Hello from SendGrid!")
        return jsonify({"message": "Test email sent."})
    except Exception as e:
        app.logger.error(f"Error in /test_email: {e}", exc_info=True)
        return jsonify({"error": "Failed to send test email."}), 500

# ------------------------------------------------------------------------------
# Favicon Endpoint
# ------------------------------------------------------------------------------
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/x-icon')

# ------------------------------------------------------------------------------
# Metrics Endpoint
# ------------------------------------------------------------------------------
@app.route('/metrics', methods=['GET'])
def metrics():
    """
    Returns basic stats:
      - uptime_seconds
      - total_requests
      - average_response_time
    """
    global APP_START_TIME, TOTAL_REQUESTS, TOTAL_REQUEST_DURATION
    current_time = time.time()
    uptime_seconds = current_time - APP_START_TIME

    # Avoid division by zero if no requests yet
    if TOTAL_REQUESTS > 0:
        average_response_time = TOTAL_REQUEST_DURATION / TOTAL_REQUESTS
    else:
        average_response_time = 0.0

    return jsonify({
        "uptime_seconds": round(uptime_seconds, 2),
        "total_requests": TOTAL_REQUESTS,
        "average_response_time": round(average_response_time, 4)
    }), 200

failed_attempts = {}  # { "username": {"count": int, "locked_until": datetime} }

# ------------------------------------------------------------------------------
# Subscription Manager Endpoint
# ------------------------------------------------------------------------------
sub_manager = SubscriptionManager()

@app.route("/some_subscription_endpoint", methods=["POST"])
def some_subscription_endpoint():
    user_id = ... # get from token or form
    plan_name = ... # e.g., "3robots"
    success = sub_manager.set_user_plan(user_id, plan_name)
    if success:
        return jsonify({"message": f"Plan updated to {plan_name}"}), 200
    else:
        return jsonify({"error": "Failed to update plan"}), 500
    
# ------------------------------------------------------------------------------
# API Access Endpoint (View + Generate/Rotate Token)
# ------------------------------------------------------------------------------
@app.route("/api_access", methods=["GET"])
@token_required
def api_access():
    """
    Displays the user's current API token (if any) and offers a button
    to generate (or rotate) a new one. Renders api_access.html.
    """
    user_id = g.current_user["user_id"]
    
    # Fetch the user's current API token from the DB
    user_token = None
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT api_token
                    FROM users
                    WHERE id = %s
                    LIMIT 1
                """, (user_id,))
                row = cursor.fetchone()
                if row and row.get("api_token"):
                    user_token = row["api_token"]
    except Exception as e:
        app.logger.error(f"Error fetching API token for user {user_id}: {e}", exc_info=True)
        # We'll still render the page, just without a token
    
    # Render the page, passing the current token (or None)
    return render_template("api_access.html", user_token=user_token)


@app.route("/generate_api_token", methods=["POST"])
@token_required
def generate_api_token():
    """
    Generates (or rotates) a new API token for the user and stores it in the DB.
    Then redirects back to /api_access so the user sees the updated token.
    """
    user_id = g.current_user["user_id"]
    
    # Generate a new random token; you can use secrets, JWT, or any secure method
    import secrets
    new_token = secrets.token_hex(32)  # 64-char hex string
    
    # Store it in the users table (assumes an 'api_token' column)
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE users
                    SET api_token = %s
                    WHERE id = %s
                """, (new_token, user_id))
            conn.commit()
    except Exception as e:
        app.logger.error(f"Error updating API token for user {user_id}: {e}", exc_info=True)
        flash("Could not generate a new token due to a server error.", "danger")
        return redirect(url_for("api_access"))
    
    flash("New API token generated successfully!", "success")
    return redirect(url_for("api_access"))

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
TOKEN_EXPIRATION_HOURS = 1   # Access token valid for 1 hour
REFRESH_TOKEN_EXPIRATION_DAYS = 7  # Refresh token valid for 7 days

# ------------------------------------------------------------------------------
# Login Endpoint
# ------------------------------------------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    # Determine whether the request is JSON or form data
    if request.is_json:
        data = request.get_json()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        json_request = True
    else:
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        json_request = False

    if not email or not password:
        if json_request:
            return jsonify({"error": "Email and password are required."}), 400
        else:
            flash("Email and password are required.", "danger")
            return redirect(url_for('login'))

    # Check lockout status
    lock_info = failed_attempts.get(email, {})
    locked_until = lock_info.get("locked_until")
    if locked_until and datetime.utcnow() < locked_until:
        if json_request:
            return jsonify({"error": "Account temporarily locked due to repeated failures."}), 403
        else:
            flash("Account temporarily locked due to repeated failures.", "danger")
            return redirect(url_for('login')), 403

    # Query DB for user (using email as username)
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, password_hash 
                    FROM users 
                    WHERE username = %s 
                    LIMIT 1
                """, (email,))
                user_row = cursor.fetchone()
    except Exception as e:
        app.logger.error(f"DB error in login for '{email}': {e}", exc_info=True)
        if json_request:
            return jsonify({"error": "Database error while accessing user."}), 500
        else:
            flash("Database error while accessing user.", "danger")
            return redirect(url_for('login')), 500

    # If user not found
    if not user_row:
        return handle_failed_login(email, json_request=json_request)

    user_id = user_row['id']
    hashed_pw = user_row['password_hash']
    if not bcrypt.checkpw(password.encode('utf-8'), hashed_pw.encode('utf-8')):
        return handle_failed_login(email, json_request=json_request)

    # Clear lock data on success
    if email in failed_attempts:
        del failed_attempts[email]

    # 1) Generate short-lived access token
    access_token = jwt.encode(
        {
            "user_id": user_id,
            "type": "access",
            "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRATION_HOURS)
        },
        app.config["SECRET_KEY"],
        algorithm="HS256"
    )

    # 2) Generate longer-lived refresh token
    refresh_token = jwt.encode(
        {
            "user_id": user_id,
            "type": "refresh",
            "exp": datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRATION_DAYS)
        },
        app.config["SECRET_KEY"],
        algorithm="HS256"
    )

    if json_request:
        # Return both tokens in JSON
        return jsonify({
            "access_token": access_token,
            "refresh_token": refresh_token
        }), 200
    else:
        flash("Login successful!", "success")
        response = redirect(url_for('dashboard'))
        # 3) Store both tokens in secure cookies
        response.set_cookie('access_token', access_token, httponly=True, samesite='Lax')
        response.set_cookie('refresh_token', refresh_token, httponly=True, samesite='Lax')
        return response

# ------------------------------------------------------------------------------
# Handling Failed Login
# ------------------------------------------------------------------------------
def handle_failed_login(email, json_request=False):
    """
    Increment fail count for the given email, possibly locking out after 5 attempts.
    Returns either a JSON error (if json_request is True) or flashes an error and redirects.
    """
    info = failed_attempts.get(email, {"count": 0, "locked_until": None})
    info["count"] += 1

    if info["count"] >= 5:
        info["locked_until"] = datetime.utcnow() + timedelta(minutes=15)
        failed_attempts[email] = info
        app.logger.warning(f"User '{email}' locked for 15 minutes.")
        if json_request:
            return jsonify({
                "error": {
                    "code": "TOO_MANY_ATTEMPTS",
                    "message": "Too many invalid attempts. Locked 15 minutes."
                }
            }), 403
        else:
            flash("Too many invalid attempts. Locked 15 minutes.", "danger")
            return redirect(url_for('login'))
    else:
        failed_attempts[email] = info
        app.logger.info(f"Invalid credentials for '{email}', attempt #{info['count']}.")
        if json_request:
            return jsonify({
                "error": {
                    "code": "INVALID_CREDENTIALS",
                    "message": "The username or password is incorrect."
                }
            }), 401
        else:
            flash("The username or password is incorrect.", "danger")
            return redirect(url_for('login'))

# ------------------------------------------------------------------------------
# Registration Endpoint
# ------------------------------------------------------------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    
    # Support both JSON and form data for registration
    if request.is_json:
        data = request.get_json()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        confirm_password = data.get('confirm_password', '')
    else:
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

    if not email or not password or not confirm_password:
        flash("Email, password, and confirm password are required.", "danger")
        return redirect(url_for('register'))
    
    if password != confirm_password:
        flash("Passwords do not match.", "danger")
        return redirect(url_for('register'))
    
    # Check if a user with this email already exists
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id FROM users WHERE username = %s LIMIT 1", (email,))
                if cursor.fetchone():
                    flash("A user with this email already exists.", "danger")
                    return redirect(url_for('register'))
    except Exception as e:
        app.logger.error(f"DB error checking existence for '{email}': {e}", exc_info=True)
        flash("Database error while checking user.", "danger")
        return redirect(url_for('register'))
    
    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Insert new user with pilot account defaults
                cursor.execute("""
                    INSERT INTO users (username, password_hash, email, account_type, pilot_start_date, pilot_end_date, subscription_status)
                    VALUES (%s, %s, %s, 'pilot', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP + INTERVAL '60 days', 'pilot')
                    RETURNING id
                """, (email, hashed_password, email))
                new_id = cursor.fetchone()['id']
            conn.commit()
        app.logger.info(f"New user '{email}' registered successfully with id {new_id}.")
    except Exception as e:
        app.logger.error(f"Error during user registration for '{email}': {e}", exc_info=True)
        flash("Could not register user due to a server error.", "danger")
        return redirect(url_for('register'))
    
    token = jwt.encode(
        {
            "user_id": new_id,
            "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRATION_HOURS)
        },
        app.config["SECRET_KEY"],
        algorithm="HS256"
    )
    
    flash("Registration successful. You are now logged in!", "success")
    response = redirect(url_for('dashboard'))
    response.set_cookie('token', token, httponly=True, samesite='Lax')
    return response

# ------------------------------------------------------------------------------
# Forgot Password Endpoint
# ------------------------------------------------------------------------------
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'GET':
        return render_template('forgot_password.html')
    
    # POST: Process the form submission
    email = request.form.get('email', '').strip()
    if not email:
        flash("Please enter your email address.", "danger")
        return redirect(url_for('forgot_password'))
    
    # Check if user exists (using email as username)
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id FROM users WHERE username = %s LIMIT 1", (email,))
                user_row = cursor.fetchone()
    except Exception as e:
        app.logger.error(f"DB error in forgot_password for {email}: {e}", exc_info=True)
        flash("Database error.", "danger")
        return redirect(url_for('forgot_password'))
    
    if not user_row:
        flash("No user found with that email address.", "warning")
        return redirect(url_for('forgot_password'))
    
    user_id = user_row['id']
    
    # Generate a password reset token valid for 30 minutes
    reset_token = jwt.encode(
        {
            "user_id": user_id,
            "reset_password": True,
            "exp": datetime.utcnow() + timedelta(minutes=30)
        },
        app.config["SECRET_KEY"],
        algorithm="HS256"
    )
    
    # Create a password reset link (ensure _external=True to generate a full URL)
    reset_link = url_for('reset_password', token=reset_token, _external=True, _scheme='https')
    subject = "Password Reset Instructions for Vestavio"
    body = (
        f"Hello,\n\n"
        f"To reset your password, please click the link below:\n\n"
        f"{reset_link}\n\n"
        f"This link will expire in 30 minutes.\n\n"
        f"If you did not request a password reset, please ignore this email."
    )
    
    try:
        send_email(email, subject, body)
        flash("Password reset instructions have been sent to your email.", "info")
    except Exception as e:
        app.logger.error(f"Error sending password reset email to {email}: {e}", exc_info=True)
        flash("Failed to send password reset email.", "danger")
    
    return redirect(url_for('login'))

# ------------------------------------------------------------------------------
# Reset Password Endpoint
# ------------------------------------------------------------------------------
@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    # Get token from query parameters (GET) or form data (POST)
    token = request.args.get('token') if request.method == 'GET' else request.form.get('token')
    if not token:
        flash("Missing password reset token.", "danger")
        return redirect(url_for('login'))
    
    try:
        payload = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
        if not payload.get("reset_password"):
            flash("Invalid password reset token.", "danger")
            return redirect(url_for('login'))
        user_id = payload["user_id"]
    except jwt.ExpiredSignatureError:
        flash("The password reset link has expired.", "danger")
        return redirect(url_for('forgot_password'))
    except jwt.InvalidTokenError:
        flash("Invalid password reset token.", "danger")
        return redirect(url_for('forgot_password'))
    
    if request.method == 'GET':
        return render_template('reset_password.html', token=token)
    
    # POST: Process the new password submission
    new_password = request.form.get('new_password', '')
    confirm_password = request.form.get('confirm_password', '')
    
    if not new_password or not confirm_password:
        flash("Please enter and confirm your new password.", "danger")
        return redirect(url_for('reset_password', token=token))
    
    if new_password != confirm_password:
        flash("Passwords do not match.", "danger")
        return redirect(url_for('reset_password', token=token))
    
    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE users
                    SET password_hash = %s
                    WHERE id = %s
                """, (hashed_password, user_id))
            conn.commit()
        flash("Your password has been updated successfully.", "success")
    except Exception as e:
        app.logger.error(f"Error updating password for user id {user_id}: {e}", exc_info=True)
        flash("Failed to update password due to a server error.", "danger")
        return redirect(url_for('reset_password', token=token))
    
    return redirect(url_for('login'))

# ------------------------------------------------------------------------------
# Refresh Endpoint
# ------------------------------------------------------------------------------
@app.route('/refresh', methods=['POST'])
def refresh():
    """
    Exchanges a valid refresh token for a new short-lived access token.
    Expects the refresh token in an httpOnly cookie (or request JSON).
    """
    # 1) Retrieve the refresh token
    refresh_token = request.cookies.get('refresh_token')
    if not refresh_token:
        # If you prefer JSON body or headers, parse it there instead
        return jsonify({"error": "No refresh token provided."}), 401

    try:
        # 2) Decode the refresh token
        data = jwt.decode(refresh_token, app.config["SECRET_KEY"], algorithms=["HS256"])
        if data.get("type") != "refresh":
            return jsonify({"error": "Invalid token type."}), 401
        user_id = data["user_id"]
    except jwt.ExpiredSignatureError:
        return jsonify({"error": "Refresh token expired."}), 401
    except jwt.InvalidTokenError:
        return jsonify({"error": "Invalid refresh token."}), 401

    # 3) Generate a new short-lived access token
    new_access_token = jwt.encode(
        {
            "user_id": user_id,
            "type": "access",
            "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRATION_HOURS)
        },
        app.config["SECRET_KEY"],
        algorithm="HS256"
    )

    # 4) Return the new access token
    #    (Set a cookie or return in JSON as needed)
    response = jsonify({"access_token": new_access_token})
    # Optional: also set_cookie if your frontend expects the token in a cookie
    response.set_cookie("access_token", new_access_token, httponly=True, samesite='Lax')
    return response, 200

# ------------------------------------------------------------------------------
# Alerts Toggle Endpoint
# ------------------------------------------------------------------------------
@app.route('/update_alert_toggle', methods=['POST'])
@token_required
def update_alert_toggle():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No data provided."}), 400

        alerts_enabled = data.get("alerts_enabled")
        if alerts_enabled is None:
            return jsonify({"error": "Missing 'alerts_enabled' parameter."}), 400

        # Ensure the value is interpreted as a boolean
        if isinstance(alerts_enabled, bool):
            email_alerts = alerts_enabled
        else:
            value = str(alerts_enabled).lower()
            if value in ["true", "1", "yes", "on"]:
                email_alerts = True
            elif value in ["false", "0", "no", "off"]:
                email_alerts = False
            else:
                return jsonify({"error": "Invalid value for 'alerts_enabled'."}), 400

        user_id = g.current_user["user_id"]
        app.logger.info(f"Updating alert toggle for user {user_id} to {email_alerts}")

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Check if a record already exists
                cursor.execute("""
                    SELECT * FROM alert_preferences
                    WHERE user_id = %s
                    LIMIT 1
                """, (user_id,))
                record = cursor.fetchone()
                if record:
                    # Update the email_alerts value
                    cursor.execute("""
                        UPDATE alert_preferences
                        SET email_alerts = %s
                        WHERE user_id = %s
                    """, (email_alerts, user_id))
                else:
                    # Insert a new record with default values for other fields
                    cursor.execute("""
                        INSERT INTO alert_preferences (user_id, preferences, email_alerts, sms_alerts, phone_number)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (user_id, json.dumps([]), email_alerts, False, None))
            conn.commit()

            # Re-read the stored value to confirm the update
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT email_alerts
                        FROM alert_preferences
                        WHERE user_id = %s
                        LIMIT 1
                    """, (user_id,))
                    updated_record = cursor.fetchone()
            stored_value = updated_record["email_alerts"] if updated_record else None

        app.logger.info(f"Alert toggle for user {user_id} stored as {stored_value}")
        return jsonify({"message": "Alert toggle updated successfully.", "alerts_enabled": stored_value}), 200

    except Exception as e:
        app.logger.error(f"Error updating alert toggle for user {g.current_user.get('user_id')}: {e}", exc_info=True)
        return jsonify({"error": "Failed to update alert toggle."}), 500


################################################################################
################################################################################
# ------------------------------------------------------------------------------
# Dashboard Endpoint (Personalized, Authenticated) - Debug + Username + KPI data
# ------------------------------------------------------------------------------
@app.route('/', methods=['GET'])
def dashboard():
    """
    Shows the latest sensor_data for the current user, grouped by sensor_name,
    with extra debugging to isolate 'tuple index out of range' in the query.
    If no token => display a minimal “please log in” page.
    Also fetches KPI data for charting (kpi_data) and uses the most recent
    user_kpi_history row to populate the KPI summary metrics.
    Now, for energy saved, we accumulate the value over the last 7 days.
    """
    # Define empty KPI data dictionary
    empty_kpi_data = {
        "timestamps": [],
        "avg_eff_history": [],
        "energy_saved_history": [],
        "system_health_history": []
    }

    # 1) Check for token in either Authorization header or cookie
    token = None
    if "Authorization" in request.headers:
        parts = request.headers["Authorization"].split()
        if len(parts) == 2 and parts[0] == "Bearer":
            token = parts[1]
    if not token:
        # try cookies
        token = request.cookies.get('access_token')

    if not token:
        # Show the minimal “please log in” version
        return render_template(
            'dashboard.html',
            sensors=[],
            system_status="",
            status_color="",
            has_issue=False,
            issue_message="",
            summary_metrics={},
            optimization_status="",
            last_updated="",
            anomaly_resolved_time="",
            milestone=None,
            improvement_note="",
            use_case="N/A",
            is_authenticated=False,
            current_user_id=None,
            username="Guest",
            kpi_data=empty_kpi_data
        )

    # 2) Decode token
    from flask import g
    try:
        data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
        g.current_user = data
    except jwt.ExpiredSignatureError:
        return render_template(
            'dashboard.html',
            sensors=[],
            system_status="Token expired",
            status_color="gray",
            has_issue=False,
            issue_message="Please log in again.",
            summary_metrics={},
            optimization_status="",
            last_updated="",
            anomaly_resolved_time="",
            milestone=None,
            improvement_note="",
            use_case="N/A",
            is_authenticated=False,
            current_user_id=None,
            username="Guest",
            kpi_data=empty_kpi_data
        )
    except jwt.InvalidTokenError:
        return render_template(
            'dashboard.html',
            sensors=[],
            system_status="Invalid token",
            status_color="gray",
            has_issue=False,
            issue_message="Please log in again.",
            summary_metrics={},
            optimization_status="",
            last_updated="",
            anomaly_resolved_time="",
            milestone=None,
            improvement_note="",
            use_case="N/A",
            is_authenticated=False,
            current_user_id=None,
            username="Guest",
            kpi_data=empty_kpi_data
        )

    # 3) We have a valid token => proceed with user_id
    user_id = g.current_user.get('user_id')
    if not user_id:
        app.logger.error("No user_id found in g.current_user.")
        return jsonify({"message": "No user ID in token"}), 400

    # 3b) Fetch the username
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT username
                FROM users
                WHERE id = %s
                LIMIT 1
            """, (user_id,))
            user_row = cursor.fetchone()
    username = user_row["username"] if user_row else f"User {user_id}"

    # 3c) Retrieve the alert preference (ensure alerts_enabled is defined for all branches)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT email_alerts
                FROM alert_preferences
                WHERE user_id = %s
                LIMIT 1
            """, (user_id,))
            pref_row = cursor.fetchone()
    alerts_enabled = pref_row["email_alerts"] if pref_row and "email_alerts" in pref_row else True

    # 4) Build your debug query using only sensor data from the last 5 minutes
    debug_query = f"""
    WITH latest AS (
        SELECT sensor_name, MAX(timestamp) AS latest_ts
        FROM sensor_data
        WHERE sensor_name <> 'Type'
        AND user_id = {user_id}
        AND timestamp >= NOW() - INTERVAL '5 minutes'
        GROUP BY sensor_name
    )
    SELECT sd.sensor_name, sd.sensor_output, sd.status, sd.timestamp
    FROM sensor_data sd
    JOIN latest l
    ON sd.sensor_name = l.sensor_name
    AND sd.timestamp = l.latest_ts
    WHERE sd.user_id = {user_id}
    ORDER BY
    CASE
        WHEN sd.sensor_name NOT ILIKE 'dof_%'
            AND sd.sensor_name NOT ILIKE 'vel_%' THEN 1
        WHEN sd.sensor_name ILIKE 'dof_%' THEN 2
        WHEN sd.sensor_name ILIKE 'vel_%' THEN 3
        ELSE 4
    END,
    sd.sensor_name
    """
    app.logger.info("=== DASHBOARD DEBUG QUERY ===")
    app.logger.info(f"user_id: {user_id}")
    app.logger.info(f"Full SQL:\n{debug_query}")

    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(debug_query)
                rows = cursor.fetchall()
    except Exception as exec_err:
        app.logger.error(f"Error running dashboard debug_query: {exec_err}", exc_info=True)
        return jsonify({
            "message": "Error fetching dashboard data (EXECUTE)",
            "error": str(exec_err)
        }), 500

    # 5) If no sensor data => blank
    if not rows:
        empty_summary = {
            'avg_energy_efficiency': None,
            'total_energy_saved': None,
            'system_health_score': None
        }
        return render_template(
            'dashboard.html',
            username=username,
            sensors=[],
            system_status="No sensor data available",
            status_color="gray",
            has_issue=False,
            issue_message="",
            summary_metrics=empty_summary,
            optimization_status="No optimization data available",
            last_updated="",
            anomaly_resolved_time="",
            milestone=None,
            improvement_note="",
            use_case="N/A",
            is_authenticated=True,
            current_user_id=user_id,
            kpi_data=empty_kpi_data,
            alerts_enabled=alerts_enabled
        )

    # 6) Process rows for the sensor snapshot
    sensors = []
    system_status = "SYSTEM FUNCTIONING PROPERLY"
    status_color = "#00ffe0"
    has_issue = False
    issue_message = ""

    for row in rows:
        sensor_name = row.get("sensor_name", "Unknown")
        raw_output = row.get("sensor_output")
        sensor_status = row.get("status", "Unknown")

        if raw_output is not None:
            try:
                output_str = f"{float(raw_output):.2f}"
            except (ValueError, TypeError):
                output_str = str(raw_output)
        else:
            output_str = "N/A"

        if sensor_status.lower() != "operational" and not has_issue:
            system_status = f"Issue with {sensor_name}"
            status_color = "#FF00FF"
            has_issue = True
            issue_message = f"Issue with {sensor_name}"

        sensors.append({
            "sensor_name": sensor_name,
            "sensor_output": output_str,
            "status": sensor_status
        })

    # 7) Pull KPI data from user_kpi_history for the last 7 days (time series)
    kpi_data = {
        "timestamps": [],
        "avg_eff_history": [],
        "energy_saved_history": [],
        "system_health_history": []
    }
    cumulative_saved = 0.0
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT timestamp,
                        avg_efficiency,
                        energy_saved_increment,
                        system_health
                    FROM user_kpi_history
                    WHERE user_id = %s
                    AND timestamp >= NOW() - INTERVAL '7 days'
                    ORDER BY timestamp ASC
                """, (user_id,))
                kpi_rows = cursor.fetchall()

        for r in kpi_rows:
            kpi_data["timestamps"].append(r["timestamp"].strftime("%Y-%m-%d %H:%M:%S"))
            kpi_data["avg_eff_history"].append(float(r["avg_efficiency"]))
            cumulative_saved += float(r["energy_saved_increment"])
            kpi_data["energy_saved_history"].append(cumulative_saved)
            kpi_data["system_health_history"].append(float(r["system_health"]))
    except Exception as e:
        app.logger.error(f"Error fetching KPI data: {e}", exc_info=True)

    # 7a) Retrieve the latest KPI row for avg_efficiency and system_health,
    # and calculate cumulative energy_saved_increment over the last 7 days.
    summary_metrics = {
        'avg_energy_efficiency': None,
        'total_energy_saved': None,
        'system_health_score': None
    }
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Retrieve the latest row for avg_efficiency and system_health
                cursor.execute("""
                    SELECT avg_efficiency, system_health
                    FROM user_kpi_history
                    WHERE user_id = %s
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (user_id,))
                latest_kpi = cursor.fetchone()
                
                # Calculate cumulative energy_saved_increment over the last 7 days
                cursor.execute("""
                    SELECT COALESCE(SUM(energy_saved_increment::float), 0) AS cumulative_energy_saved
                    FROM user_kpi_history
                    WHERE user_id = %s
                    AND timestamp >= NOW() - INTERVAL '7 days'
                """, (user_id,))
                sum_row = cursor.fetchone()

        if latest_kpi:
            summary_metrics['avg_energy_efficiency'] = round(float(latest_kpi['avg_efficiency']), 1)
            summary_metrics['system_health_score'] = round(float(latest_kpi['system_health']), 1)
        else:
            summary_metrics['avg_energy_efficiency'] = 0.0
            summary_metrics['system_health_score'] = 0.0

        if sum_row and sum_row['cumulative_energy_saved'] is not None:
            summary_metrics['total_energy_saved'] = round(float(sum_row['cumulative_energy_saved']), 1)
        else:
            summary_metrics['total_energy_saved'] = 0.0

    except Exception as e:
        app.logger.error(f"Error fetching KPI summary: {e}", exc_info=True)
        summary_metrics = {
            'avg_energy_efficiency': 0.0,
            'total_energy_saved': 0.0,
            'system_health_score': 0.0
        }

    # 8) Additional fields for the template
    import datetime
    from datetime import timedelta
    last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    anomaly_resolved_time = (datetime.datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")

    milestone = None
    if summary_metrics['total_energy_saved'] and summary_metrics['total_energy_saved'] > 100:
        milestone = f"🎉 You have saved {summary_metrics['total_energy_saved']} units of energy!"

    improvement_note = "Your energy efficiency improved by 5% compared to last month."

    # Detect use case from the recent sensor data
    sensor_names = list({r["sensor_name"] for r in rows if r.get("sensor_name")})
    from B_Module_Files.use_case_selector_module import detect_use_case
    detected_use_case = detect_use_case(sensor_names)
    if detected_use_case == "unknown":
        use_case_env = os.getenv("USE_CASE", "warehouse").lower()
        use_case = use_case_env
    else:
        use_case = detected_use_case

    # 8a) Retrieve distinct robot numbers for the current user from recent sensor_data
    robot_numbers = []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT DISTINCT robot_number
                    FROM sensor_data
                    WHERE user_id = %s
                      AND timestamp >= NOW() - INTERVAL '5 minutes'
                    ORDER BY robot_number ASC
                """, (user_id,))
                rows_robot = cursor.fetchall()
                robot_numbers = [row[0] for row in rows_robot if row[0] is not None]
    except Exception as e_robot:
        app.logger.error(f"Error fetching robot numbers for user {user_id}: {e_robot}", exc_info=True)

    # 9 After fetching the username (e.g., after line retrieving user_row)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT email_alerts
                FROM alert_preferences
                WHERE user_id = %s
                LIMIT 1
            """, (user_id,))
            pref_row = cursor.fetchone()
    app.logger.info(f"Alert preferences for user {user_id}: {pref_row}")
    alerts_enabled = pref_row["email_alerts"] if pref_row and "email_alerts" in pref_row else True

    # 10 Then pass 'alerts_enabled' to your render_template call:
    return render_template(
        'dashboard.html',
        username=username,
        sensors=sensors,
        system_status=system_status,
        status_color=status_color,
        has_issue=has_issue,
        issue_message=issue_message,
        summary_metrics=summary_metrics,
        optimization_status="System is currently optimizing...",
        last_updated=last_updated,
        anomaly_resolved_time=anomaly_resolved_time,
        milestone=milestone,
        improvement_note=improvement_note,
        use_case=use_case,
        is_authenticated=True,
        current_user_id=user_id,
        kpi_data=kpi_data,
        robot_numbers=robot_numbers,
        alerts_enabled=alerts_enabled  # New: pass alert toggle preference
    )
  
# ------------------------------------------------------------------------------
# Snapshot Endpoint (Personalized with partial-data fallbacks + pagination + robot filter)
# ------------------------------------------------------------------------------
@app.route('/snapshot', methods=['GET'])
@token_required
def get_snapshot():
    """
    Returns the latest sensor data for the current user (only from the last minute),
    optionally filtered by robot_number and paginated (10 per page).
    Also returns summary metrics including average energy efficiency, total energy saved,
    and system health score.

    The system health score is calculated as:
      max(0, avg_efficiency - (weighted anomalies) - 0.5 * avg_pm + avg_rl_bonus)

    If an issue is detected and the user’s alert preferences include
    critical/warning anomalies, we insert a system chat message in chat_logs
    to inform them in real-time.
    """
    try:
        # Use g.current_user['user_id'] from token_required
        user_id = g.current_user['user_id']
        app.logger.debug(f"Fetching snapshot for user_id: {user_id}")

        # 0) Read optional query params for pagination & robot filtering
        page = request.args.get('page', 1, type=int)
        robot_number = request.args.get('robot_number', None, type=int)
        limit = 12
        offset = (page - 1) * limit

        # 1) Fetch user alert preferences
        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT preferences
                        FROM alert_preferences
                        WHERE user_id = %s
                        LIMIT 1
                    """, (user_id,))
                    pref_row = cursor.fetchone()
            if pref_row and pref_row.get("preferences") is not None:
                raw_pref = pref_row["preferences"]
                if isinstance(raw_pref, str):
                    user_alert_prefs = json.loads(raw_pref)
                else:
                    user_alert_prefs = raw_pref if raw_pref else []
            else:
                user_alert_prefs = []
        except Exception as e_pref:
            app.logger.warning(f"Could not fetch alert preferences for user {user_id}: {e_pref}")
            user_alert_prefs = []

        time_window = "1 minute"

        # 2) Fetch sensor data with pagination & optional robot filter
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                if robot_number is None:
                    cursor.execute(f"""
                        SELECT DISTINCT robot_number
                        FROM sensor_data
                        WHERE user_id = %s
                        AND timestamp >= NOW() - INTERVAL '{time_window}'
                        ORDER BY robot_number ASC
                    """, (user_id,))
                    distinct_rows = cursor.fetchall()
                    distinct_robot_numbers = [row['robot_number'] for row in distinct_rows if row['robot_number'] is not None]
                    if not distinct_robot_numbers or (page - 1) >= len(distinct_robot_numbers):
                        rows = []  # No robot available for the requested page
                    else:
                        robot_number = distinct_robot_numbers[page - 1]
                        offset = 0
                        limit = 1000  # return all sensors for that robot

                if 'rows' not in locals():
                    query = f"""
                        WITH latest AS (
                            SELECT sensor_name,
                                robot_number,
                                MAX(timestamp) AS latest_ts
                            FROM sensor_data
                            WHERE sensor_name <> 'Type'
                            AND user_id = %s
                            AND timestamp >= NOW() - INTERVAL '{time_window}'
                            AND (%s IS NULL OR robot_number = %s)
                            GROUP BY sensor_name, robot_number
                        )
                        SELECT s.sensor_name,
                            s.sensor_output,
                            s.status,
                            s.timestamp,
                            s.robot_number
                        FROM sensor_data s
                        JOIN latest l
                        ON s.sensor_name = l.sensor_name
                        AND s.robot_number = l.robot_number
                        AND s.timestamp = l.latest_ts
                        WHERE s.user_id = %s
                        AND (%s IS NULL OR s.robot_number = %s)
                        ORDER BY s.robot_number ASC,
                                CASE
                                WHEN s.sensor_name NOT ILIKE 'dof_%%'
                                        AND s.sensor_name NOT ILIKE 'vel_%%' THEN 1
                                WHEN s.sensor_name ILIKE 'dof_%%' THEN 2
                                WHEN s.sensor_name ILIKE 'vel_%%' THEN 3
                                ELSE 4
                                END,
                                s.sensor_name
                        LIMIT {limit} OFFSET {offset}
                    """
                    cursor.execute(
                        query,
                        (user_id, robot_number, robot_number, user_id, robot_number, robot_number)
                    )
                    rows = cursor.fetchall()

        if not rows:
            summary_metrics = {
                "avg_energy_efficiency": None,
                "total_energy_saved": None,
                "system_health_score": None
            }
            return jsonify({
                "sensors": [],
                "summary_metrics": summary_metrics,
                "milestone": None,
                "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "system_status": "NO SENSOR DATA",
                "status_color": "gray",
                "has_issue": False,
                "issue_message": None,
                "use_case": "Unspecified"
            }), 200

        # 3) Build snapshot & track system status
        snapshot_data = []
        any_issue = False
        issue_message = None
        latest_ts = rows[0]["timestamp"]

        for r in rows:
            if r["timestamp"] > latest_ts:
                latest_ts = r["timestamp"]

            sensor_output = r["sensor_output"] if r["sensor_output"] is not None else "N/A"
            status = r["status"] if r["status"] else "Unknown"
            if status.lower() != "operational":
                any_issue = True
                issue_message = f"Issue with {r['sensor_name']}"

            snapshot_data.append({
                "sensor_name": r["sensor_name"],
                "sensor_output": sensor_output,
                "status": status,
                "robot_number": r["robot_number"]
            })

        # 4) Summary metrics
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Energy efficiency remains over the 1-minute window
                cursor.execute("""
                    SELECT AVG(sensor_output::float) AS avg_eff
                    FROM sensor_data
                    WHERE sensor_name ILIKE 'energy_efficiency'
                    AND user_id = %s
                    AND timestamp >= NOW() - INTERVAL %s
                """, (user_id, time_window))
                row_eff = cursor.fetchone()
                avg_eff = row_eff["avg_eff"] if row_eff and row_eff["avg_eff"] else None

                # For energy saved, we now accumulate over a 1-hour window
                cumulative_interval = "1 hour"
                cursor.execute("""
                    SELECT SUM(sensor_output::float) AS total_saved
                    FROM sensor_data
                    WHERE sensor_name ILIKE 'energy_saved'
                    AND user_id = %s
                    AND timestamp >= NOW() - INTERVAL %s
                """, (user_id, cumulative_interval))
                row_saved = cursor.fetchone()
                total_saved = row_saved["total_saved"] if row_saved and row_saved["total_saved"] else None

                cursor.execute("""
                    SELECT COALESCE(SUM(
                        CASE 
                            WHEN anomaly_severity ILIKE 'low' THEN 2
                            WHEN anomaly_severity ILIKE 'medium' THEN 5
                            WHEN anomaly_severity ILIKE 'high' THEN 10
                            ELSE 10
                        END
                    ), 0) AS weighted_anomalies
                    FROM sensor_data
                    WHERE status ILIKE 'faulty'
                    AND user_id = %s
                    AND timestamp >= NOW() - INTERVAL %s
                """, (user_id, time_window))
                row_anom = cursor.fetchone()
                weighted_anomalies = row_anom["weighted_anomalies"] if row_anom else 0

                cursor.execute("""
                    SELECT AVG(sensor_output::float) AS avg_pm
                    FROM sensor_data
                    WHERE sensor_name ILIKE 'pm_risk'
                    AND user_id = %s
                    AND timestamp >= NOW() - INTERVAL %s
                """, (user_id, time_window))
                row_pm = cursor.fetchone()
                avg_pm = row_pm["avg_pm"] if row_pm and row_pm["avg_pm"] else 0

                cursor.execute("""
                    SELECT AVG(sensor_output::float) AS avg_rl_bonus
                    FROM sensor_data
                    WHERE sensor_name ILIKE 'rl_bonus'
                    AND user_id = %s
                    AND timestamp >= NOW() - INTERVAL %s
                """, (user_id, time_window))
                row_rl = cursor.fetchone()
                avg_rl_bonus = row_rl["avg_rl_bonus"] if row_rl and row_rl["avg_rl_bonus"] else 0

        def safe_float(val):
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        avg_efficiency = safe_float(avg_eff)
        if avg_efficiency is not None:
            system_health_score = max(
                0,
                avg_efficiency - weighted_anomalies - 0.5 * avg_pm + avg_rl_bonus
            )
        else:
            system_health_score = None

        summary_metrics = {
            "avg_energy_efficiency": avg_efficiency,
            "total_energy_saved": safe_float(total_saved),
            "system_health_score": system_health_score
        }

        # 5) Optional milestone
        milestone_text = None
        if avg_efficiency and avg_efficiency >= 95.0:
            milestone_text = f"We just hit {avg_efficiency:.1f}% efficiency—new record!"
        if total_saved and total_saved > 500:
            milestone_text = "Over 500 units of energy saved—great job!"

        # 6) Determine system status
        if any_issue:
            system_status = "ISSUE DETECTED"
            status_color = "#FF00FF"
        else:
            system_status = "SYSTEM FUNCTIONING PROPERLY"
            status_color = "#00ffe0"

        # 7) Use-case detection & filter
        from B_Module_Files.use_case_selector_module import detect_use_case, get_expected_sensors
        sensor_names = [s["sensor_name"].lower() for s in snapshot_data]
        detected_use_case = detect_use_case(sensor_names)
        if detected_use_case == "unknown":
            display_use_case = "Unspecified"
            filter_use_case = "warehouse"
        else:
            display_use_case = detected_use_case
            filter_use_case = detected_use_case

        expected_sensors = {s.lower() for s in get_expected_sensors(filter_use_case)}
        snapshot_data = [
            s for s in snapshot_data
            if s["sensor_name"].lower() in expected_sensors
        ]

        local_tz = pytz.timezone("America/Chicago")
        last_updated_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        #######################
        # 8) If there's an issue, log in system_logs, send email alert, and insert into chat_logs
        if any_issue:
            try:
                with get_db_connection() as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                        alert_msg = f"Alert: {issue_message or 'Unknown Issue'}"

                        # 1) Lock both tables to avoid concurrent duplicates
                        cursor.execute("LOCK TABLE chat_logs IN EXCLUSIVE MODE;")
                        cursor.execute("LOCK TABLE system_logs IN EXCLUSIVE MODE;")

                        # 2) Check for duplicates in both tables (last 5 min)
                        cursor.execute("""
                            SELECT EXISTS (
                                SELECT 1
                                FROM (
                                    SELECT message, created_at AS ts
                                    FROM chat_logs
                                    WHERE user_id = %s
                                    AND speaker = 'System [Anomaly]'
                                    AND created_at >= NOW() - INTERVAL '5 minutes'
                                    UNION ALL
                                    SELECT message, timestamp AS ts
                                    FROM system_logs
                                    WHERE user_id = %s
                                    AND log_type = 'Anomaly'
                                    AND timestamp >= NOW() - INTERVAL '5 minutes'
                                ) AS recent_alerts
                                WHERE message = %s
                            ) AS alert_exists;
                        """, (user_id, user_id, alert_msg))
                        exists_result = cursor.fetchone()

                        # Determine whether to insert into logs (only if duplicate doesn't exist)
                        if not (exists_result and exists_result.get("alert_exists")):
                            # 3) Insert alert into chat_logs
                            cursor.execute("""
                                INSERT INTO chat_logs (user_id, speaker, message, created_at)
                                VALUES (%s, %s, %s, NOW())
                            """, (user_id, "System [Anomaly]", alert_msg))
                            insert_into_logs = True
                        else:
                            app.logger.info(f"Duplicate alert found for user {user_id}, skipping log insertion.")
                            insert_into_logs = False

                        # 4) Dynamically generate a resolution via the agent
                        try:
                            resolution_text = run_sync_request(
                                f"Provide a resolution recommendation for the following alert: '{alert_msg}'",
                                agent_name="alert_notification_agent"
                            )
                        except Exception as e_dyn:
                            app.logger.error(f"Dynamic resolution generation failed: {e_dyn}", exc_info=True)
                            resolution_text = "No dynamic resolution available. Please contact support."

                        # 5) Insert into system_logs with resolution in notes if not duplicate
                        if insert_into_logs:
                            cursor.execute("""
                                INSERT INTO system_logs (user_id, log_type, message, notes, timestamp)
                                VALUES (%s, %s, %s, %s, NOW())
                            """, (user_id, "Anomaly", alert_msg, resolution_text))
                    # 6) Commit once for both inserts (atomic transaction)
                    conn.commit()

                    # 7) Retrieve the logged-in user's email from the users table and check alert preference
                    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                        cursor.execute("""
                            SELECT email
                            FROM users
                            WHERE id = %s
                            LIMIT 1
                        """, (user_id,))
                        user_row = cursor.fetchone()
                    recipient = user_row["email"] if user_row and user_row.get("email") else "fallback@vestavio.com"

                    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                        cursor.execute("""
                            SELECT email_alerts
                            FROM alert_preferences
                            WHERE user_id = %s
                            LIMIT 1
                        """, (user_id,))
                        pref_row = cursor.fetchone()
                    email_alerts_enabled = pref_row["email_alerts"] if pref_row and "email_alerts" in pref_row else True

                    app.logger.info(f"Prepared to send alert email to {recipient} with subject 'Alert Notification'. Email alerts enabled: {email_alerts_enabled}")

                    # 8) Send the alert email only if it's not a duplicate and if email alerts are enabled
                    if insert_into_logs and email_alerts_enabled:
                        send_email(
                            recipient,
                            "Alert Notification",
                            f"{alert_msg}\n\n{resolution_text}"
                        )
            except Exception as e_alert:
                app.logger.warning(f"Failed to process alert for user {user_id}: {e_alert}", exc_info=True)

        # 9) Return final JSON
        return jsonify({
            "sensors": snapshot_data,
            "summary_metrics": summary_metrics,
            "milestone": milestone_text,
            "last_updated": last_updated_str,
            "system_status": system_status,
            "status_color": status_color,
            "has_issue": any_issue,
            "issue_message": issue_message,
            "use_case": display_use_case
        }), 200

    except Exception as e:
        app.logger.error(f"Error fetching snapshot data: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------------
# Refresh Snapshot / Send Sensor Data Endpoint (Dynamic, Personalized)
# --------------------------------------------------------
@app.route('/send_sensor_data', methods=['POST'])
@token_required
def send_sensor_data():
    """
    Handles incoming sensor data for:
      - DOF and velocity readings (for anomaly detection, RL, and energy efficiency)
      - Predictive maintenance fields (Type, Torque, etc.)

    Inserts data into the 'sensor_data' table along with the current user's ID,
    then triggers model inferences. The pivot queries dynamically adjust based on the use case.

    The overall system health score is calculated as:
      max(0, avg_efficiency - (weighted anomalies) - 0.5 * avg_pm + avg_rl_bonus)

    Request JSON Example:
      {
        "sensor_data": [
          { "name": "dof_1", "output": 0.12, "status": "Operational" },
          { "name": "vel_1", "output": 0.05, "status": "Operational" },
          ...
        ]
      }

    Returns (JSON):
      {
        "status": "Data processed successfully",
        "results": {
          "anomaly_preds": [...],
          "rl_actions": [...],
          "energy_predictions": [...],
          "pm_predictions": [...]
        },
        "skipped_count": <int>,
        "inserted_count": <int>
      }
    """
    app.logger.debug("Request reached /send_sensor_data endpoint")

    data = request.get_json() or {}
    sensor_data = data.get('sensor_data', [])

    if not sensor_data:
        app.logger.error("No sensor data provided.")
        return jsonify({
            "error": {
                "code": "NO_SENSOR_DATA",
                "message": "No 'sensor_data' provided in request JSON."
            }
        }), 400

    results = {
        "anomaly_preds": [],
        "rl_actions": [],
        "energy_predictions": [],
        "pm_predictions": []
    }
    skipped_count = 0
    inserted_count = 0

    # Determine the use case from environment (default to 'warehouse')
    use_case = os.getenv('USE_CASE', 'warehouse').lower()
    app.logger.info(f"send_sensor_data: Using use case {use_case}")

    try:
        # Retrieve the current user's ID from the token
        user_id = g.current_user['user_id']

        # --------------------------------------------------------
        # Retrieve user alert preferences with additional fields
        # --------------------------------------------------------
        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT preferences, email_alerts, sms_alerts, phone_number
                        FROM alert_preferences
                        WHERE user_id = %s
                        LIMIT 1
                    """, (user_id,))
                    pref_row = cursor.fetchone()
            if pref_row:
                user_alert_prefs = {
                    "preferences": json.loads(pref_row["preferences"]) if pref_row.get("preferences") else [],
                    "email_alerts": pref_row.get("email_alerts", True),
                    "sms_alerts": pref_row.get("sms_alerts", False),
                    "phone_number": pref_row.get("phone_number")
                }
            else:
                user_alert_prefs = {"preferences": [], "email_alerts": True, "sms_alerts": False, "phone_number": None}
        except Exception as e_pref:
            app.logger.warning(f"Could not fetch alert preferences for user {user_id}: {e_pref}")
            user_alert_prefs = {"preferences": [], "email_alerts": True, "sms_alerts": False, "phone_number": None}

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                current_time = datetime.utcnow()
                from datetime import timedelta
                time_lower = current_time - timedelta(seconds=1)
                time_upper = current_time + timedelta(seconds=1)

                # 1) Insert incoming sensor data with user_id
                for sensor in sensor_data:
                    sensor_name = sensor.get('name')
                    sensor_output = sensor.get('output')
                    status = sensor.get('status', 'Operational')

                    if not sensor_name or sensor_output is None:
                        app.logger.warning(f"Incomplete sensor data: {sensor}")
                        skipped_count += 1
                        continue

                    if not validate_sensor_data(sensor_name, sensor_output):
                        app.logger.warning(f"Invalid data for sensor {sensor_name}")
                        skipped_count += 1
                        continue

                    cursor.execute(
                        """
                        INSERT INTO sensor_data (sensor_name, sensor_output, status, timestamp, user_id)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (sensor_name, sensor_output, status, current_time, user_id)
                    )
                    inserted_count += 1

                conn.commit()

                if inserted_count == 0:
                    app.logger.warning("All sensor rows were invalid or skipped.")
                    return jsonify({
                        "error": {
                            "code": "ALL_SKIPPED",
                            "message": "All sensor rows were invalid or incomplete."
                        },
                        "skipped_count": skipped_count,
                        "inserted_count": inserted_count
                    }), 400

                # -----------------------------------------------------------
                # 2) DOF + Velocity Pivot => Anomaly, RL, Energy Efficiency
                # -----------------------------------------------------------
                if use_case == '9dof':
                    dof_vel_cols = [
                        'dof_1','dof_2','dof_3','dof_4','dof_5','dof_6','dof_7','dof_8','dof_9',
                        'vel_1','vel_2','vel_3','vel_4','vel_5','vel_6','vel_7','vel_8','vel_9'
                    ]
                    expected_features = 18
                elif use_case == '6dof':
                    dof_vel_cols = [
                        'dof_1','dof_2','dof_3','dof_4','dof_5','dof_6',
                        'vel_1','vel_2','vel_3','vel_4','vel_5','vel_6'
                    ]
                    expected_features = 12
                elif use_case == 'drone':
                    dof_vel_cols = [
                        'dof_1','dof_2','dof_3','dof_4',
                        'vel_1','vel_2','vel_3','vel_4'
                    ]
                    expected_features = 8
                elif use_case == 'urban':
                    dof_vel_cols = [
                        'dof_1','dof_2','dof_3','dof_4','dof_5',
                        'vel_1','vel_2','vel_3','vel_4','vel_5'
                    ]
                    expected_features = 10
                elif use_case == 'warehouse':
                    dof_vel_cols = [
                        'dof_1','dof_2','dof_3','dof_4','dof_5','dof_6','dof_7','dof_8',
                        'vel_1','vel_2','vel_3','vel_4','vel_5','vel_6','vel_7','vel_8'
                    ]
                    expected_features = 16
                else:
                    # Fallback to warehouse
                    dof_vel_cols = [
                        'dof_1','dof_2','dof_3','dof_4','dof_5','dof_6','dof_7','dof_8',
                        'vel_1','vel_2','vel_3','vel_4','vel_5','vel_6','vel_7','vel_8'
                    ]
                    expected_features = 16

                dof_vel_sql = f"""
                    SELECT {', '.join(
                        f"MAX(CASE WHEN sensor_name='{col}' THEN sensor_output END) AS {col}"
                        for col in dof_vel_cols
                    )}
                    FROM sensor_data
                    WHERE user_id = %s
                      AND timestamp BETWEEN %s AND %s
                """
                cursor.execute(dof_vel_sql, (user_id, time_lower, time_upper))
                pivot_row = cursor.fetchone()

                if pivot_row:
                    import numpy as np
                    dof_vel_features = []
                    for col in dof_vel_cols:
                        val = pivot_row[col]
                        if val is None:
                            val = 0.0
                        dof_vel_features.append(float(val))
                    X_obs = np.array(dof_vel_features, dtype=np.float32).reshape(1, -1)
                    aligned_obs = preprocessor.align_features_for_inference(
                        X_obs, expected_features, fill_value=0.0
                    )

                    # 2a) Anomaly Detection (skip if urban)
                    if use_case not in ['urban'] and anomaly_detection_model is not None and hasattr(anomaly_detection_model, 'predict'):
                        try:
                            anomaly_score = anomaly_detection_model.predict(aligned_obs)
                            from B_Module_Files.anomaly_severity_module import get_anomaly_severity
                            anomaly_severity = get_anomaly_severity(anomaly_score[0], 0.5, 0.8)
                            results["anomaly_preds"] = anomaly_score.tolist()
                            results["anomaly_severity"] = anomaly_severity
                            if anomaly_severity == "high":
                                if user_alert_prefs.get("email_alerts", True):
                                    recipient = user_alert_prefs.get("phone_number") or "alerts@vestavio.com"
                                    send_email(
                                        recipient,
                                        "High Severity Anomaly Detected!",
                                        f"Anomaly score: {anomaly_score[0]:.2f} ({anomaly_severity} severity)."
                                    )
                        except Exception as ad_err:
                            app.logger.warning(f"Anomaly detection error: {ad_err}")
                    else:
                        results["anomaly_preds"] = []
                        results["anomaly_severity"] = "none"

                    # 2b) RL Inference
                    if rl_model is not None:
                        action, _ = rl_model.predict(aligned_obs, deterministic=True)
                        results["rl_actions"] = action.flatten().tolist()
                    else:
                        app.logger.warning("RL model not available.")

                    # 2c) Energy Efficiency Inference
                    if ee_scaler is not None and ee_model is not None:
                        try:
                            X_ee_scaled = ee_scaler.transform(aligned_obs)
                            ee_pred = ee_model.predict(X_ee_scaled)
                            results["energy_predictions"] = ee_pred.tolist()
                            app.logger.debug(f"Energy efficiency predictions: {ee_pred}")
                        except Exception as ee_err:
                            app.logger.warning(f"Energy model inference failed: {ee_err}")
                    else:
                        app.logger.warning("Energy efficiency models not available.")
                else:
                    app.logger.debug("No pivot row found for DOF/velocity pivot. Skipping anomaly/energy/rl inference.")

                # -----------------------------------------------------------
                # 3) Predictive Maintenance pivot (skip for urban unless we add it)
                # -----------------------------------------------------------
                if use_case in ['9dof','6dof','warehouse','drone']:
                    if use_case == '9dof':
                        pm_cols = [
                            "dof_1","dof_2","dof_3","dof_4","dof_5","dof_6","dof_7","dof_8","dof_9",
                            "vel_1","vel_2","vel_3","vel_4","vel_5","vel_6","vel_7","vel_8","vel_9",
                            "Type","Air temperature [K]","Process temperature [K]",
                            "Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"
                        ]
                    elif use_case == '6dof':
                        pm_cols = [
                            "dof_1","dof_2","dof_3","dof_4","dof_5","dof_6",
                            "vel_1","vel_2","vel_3","vel_4","vel_5","vel_6",
                            "Type","Air temperature [K]","Process temperature [K]",
                            "Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"
                        ]
                    elif use_case == 'drone':
                        pm_cols = [
                            "dof_1","dof_2","dof_3","dof_4",
                            "vel_1","vel_2","vel_3","vel_4",
                            "Type","Air temperature [K]","Process temperature [K]",
                            "Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"
                        ]
                    else:  # warehouse
                        pm_cols = [
                            "dof_1","dof_2","dof_3","dof_4","dof_5","dof_6","dof_7","dof_8",
                            "vel_1","vel_2","vel_3","vel_4","vel_5","vel_6","vel_7","vel_8",
                            "Type","Air temperature [K]","Process temperature [K]",
                            "Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"
                        ]

                    pm_select = ", ".join(
                        f"MAX(CASE WHEN sensor_name='{col}' THEN sensor_output END) AS \"{col}\""
                        for col in pm_cols
                    )
                    pm_sql = f"""
                        SELECT {pm_select}
                        FROM sensor_data
                        WHERE user_id = %s AND timestamp BETWEEN %s AND %s
                    """
                    cursor.execute(pm_sql, (user_id, time_lower, time_upper))
                    pm_pivot = cursor.fetchone()

                    if pm_pivot:
                        try:
                            import pandas as pd
                            row_dict = {}
                            for feat in pm_cols:
                                val = pm_pivot[feat]
                                if val is None:
                                    val = 0.0
                                row_dict[feat] = val

                            if use_case == '9dof':
                                pm_df = pd.DataFrame([{
                                    **{f"dof_{i}": float(row_dict[f"dof_{i}"]) for i in range(1, 10)},
                                    **{f"vel_{i}": float(row_dict[f"vel_{i}"]) for i in range(1, 10)},
                                    "Air temperature [K]": float(row_dict["Air temperature [K]"]),
                                    "Process temperature [K]": float(row_dict["Process temperature [K]"]),
                                    "Rotational speed [rpm]": float(row_dict["Rotational speed [rpm]"]),
                                    "Torque [Nm]": float(row_dict["Torque [Nm]"]),
                                    "Tool wear [min]": float(row_dict["Tool wear [min]"]),
                                    "Type_L": 1.0 if str(row_dict["Type"]) == "L" else 0.0,
                                    "Type_M": 1.0 if str(row_dict["Type"]) == "M" else 0.0,
                                    "Type_H": 1.0 if str(row_dict["Type"]) == "H" else 0.0
                                }])
                            elif use_case == '6dof':
                                pm_df = pd.DataFrame([{
                                    **{f"dof_{i}": float(row_dict[f"dof_{i}"]) for i in range(1, 7)},
                                    **{f"vel_{i}": float(row_dict[f"vel_{i}"]) for i in range(1, 7)},
                                    "Air temperature [K]": float(row_dict["Air temperature [K]"]),
                                    "Process temperature [K]": float(row_dict["Process temperature [K]"]),
                                    "Rotational speed [rpm]": float(row_dict["Rotational speed [rpm]"]),
                                    "Torque [Nm]": float(row_dict["Torque [Nm]"]),
                                    "Tool wear [min]": float(row_dict["Tool wear [min]"]),
                                    "Type_L": 1.0 if str(row_dict["Type"]) == "L" else 0.0,
                                    "Type_M": 1.0 if str(row_dict["Type"]) == "M" else 0.0,
                                    "Type_H": 1.0 if str(row_dict["Type"]) == "H" else 0.0
                                }])
                            elif use_case == 'drone':
                                pm_df = pd.DataFrame([{
                                    **{f"dof_{i}": float(row_dict[f"dof_{i}"]) for i in range(1, 5)},
                                    **{f"vel_{i}": float(row_dict[f"vel_{i}"]) for i in range(1, 5)},
                                    "Air temperature [K]": float(row_dict["Air temperature [K]"]),
                                    "Process temperature [K]": float(row_dict["Process temperature [K]"]),
                                    "Rotational speed [rpm]": float(row_dict["Rotational speed [rpm]"]),
                                    "Torque [Nm]": float(row_dict["Torque [Nm]"]),
                                    "Tool wear [min]": float(row_dict["Tool wear [min]"]),
                                    "Type_L": 1.0 if str(row_dict["Type"]) == "L" else 0.0,
                                    "Type_M": 1.0 if str(row_dict["Type"]) == "M" else 0.0,
                                    "Type_H": 1.0 if str(row_dict["Type"]) == "H" else 0.0
                                }])
                            else:  # warehouse
                                pm_df = pd.DataFrame([{
                                    **{f"dof_{i}": float(row_dict[f"dof_{i}"]) for i in range(1, 9)},
                                    **{f"vel_{i}": float(row_dict[f"vel_{i}"]) for i in range(1, 9)},
                                    "Air temperature [K]": float(row_dict["Air temperature [K]"]),
                                    "Process temperature [K]": float(row_dict["Process temperature [K]"]),
                                    "Rotational speed [rpm]": float(row_dict["Rotational speed [rpm]"]),
                                    "Torque [Nm]": float(row_dict["Torque [Nm]"]),
                                    "Tool wear [min]": float(row_dict["Tool wear [min]"]),
                                    "Type_L": 1.0 if str(row_dict["Type"]) == "L" else 0.0,
                                    "Type_M": 1.0 if str(row_dict["Type"]) == "M" else 0.0,
                                    "Type_H": 1.0 if str(row_dict["Type"]) == "H" else 0.0
                                }])

                            pm_features_transformed = pm_transformer.transform(pm_df)
                            pm_preds = pm_model.predict(pm_features_transformed)
                            results["pm_predictions"] = pm_preds.tolist()
                            app.logger.debug(f"{use_case.upper()} PM predictions: {pm_preds}")

                        except Exception as pm_err:
                            app.logger.warning(f"PM model inference failed: {pm_err}")
                    else:
                        app.logger.debug("No pivot row found for PM features.")
                else:
                    app.logger.debug(f"Skipping PM pivot for use_case={use_case}")

        # Return combined results
        return jsonify({
            "status": "Data processed successfully",
            "results": results,
            "skipped_count": skipped_count,
            "inserted_count": inserted_count
        }), 200

    except Exception as e:
        app.logger.error(f"Error processing sensor data: {e}", exc_info=True)
        return jsonify({"error": "Failed to process sensor data"}), 500
    
# ------------------------------------------------------------------------------
# Raw Data Endpoint
# ------------------------------------------------------------------------------
@app.route('/raw_data', methods=['GET'])
@token_required
def raw_data():
    aggregated = {}
    # Retrieve the current access token from cookies for internal requests.
    token = request.cookies.get('access_token')
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    client = app.test_client()

    try:
        aggregated['health'] = client.get('/health', headers=headers).get_json()
    except Exception as e:
        aggregated['health'] = {"error": str(e)}
    try:
        aggregated['metrics'] = client.get('/metrics', headers=headers).get_json()
    except Exception as e:
        aggregated['metrics'] = {"error": str(e)}
    try:
        aggregated['snapshot'] = client.get('/snapshot', headers=headers).get_json()
    except Exception as e:
        aggregated['snapshot'] = {"error": str(e)}
    
    # Process snapshot data for dynamic use-case detection
    snapshot_data = aggregated.get('snapshot', [])

    # /rl_predict Endpoint with dynamic observation length and sensor metadata
    try:
        from B_Module_Files.use_case_selector_module import detect_use_case, get_expected_sensors
        # Dynamically determine use case from snapshot data
        if isinstance(snapshot_data, list) and snapshot_data:
            sensor_names = [s["sensor_name"].lower() for s in snapshot_data if "sensor_name" in s]
            detected_use_case = detect_use_case(sensor_names)
            use_case_for_rl = detected_use_case if detected_use_case != "unknown" else "warehouse"
        else:
            use_case_for_rl = "warehouse"

        # Define a dynamic function to get expected observation length based on use case
        def dynamic_expected_length(use_case):
            uc = use_case.lower()
            if uc == "drone":
                return 8
            elif uc == "urban":
                return 10
            elif uc == "6dof":
                return 12
            elif uc == "9dof":
                return 18
            else:
                return 16

        expected_length = dynamic_expected_length(use_case_for_rl)
        sample_obs = [0.0] * expected_length
        sample_sensors = list(get_expected_sensors(use_case_for_rl))
        aggregated['rl_predict'] = client.post(
            '/rl_predict',
            json={"obs": sample_obs, "sensors": sample_sensors},
            headers=headers
        ).get_json()

        # POST-PROCESS: Remove the "use_case" field from the RL endpoint's response, if present
        if isinstance(aggregated['rl_predict'], dict) and 'use_case' in aggregated['rl_predict']:
            del aggregated['rl_predict']['use_case']

    except Exception as e:
        aggregated['rl_predict'] = {"error": str(e)}
    
    # End of Day Summary
    try:
        eod_response = client.get('/end_of_day_summary', headers=headers).get_json()
        aggregated['end_of_day_summary'] = eod_response.get('end_of_day_summary')
        aggregated['summary_date'] = eod_response.get('summary_date')
    except Exception as e:
        aggregated['end_of_day_summary'] = {"error": str(e)}
        aggregated['summary_date'] = {"error": str(e)}

    return render_template('raw_data.html', aggregated=aggregated), 200

# ------------------------------------------------------------------------------
# End of Day Endpoint
# ------------------------------------------------------------------------------
'''
@app.route('/end_of_day_summary', methods=['GET'])
@token_required
def end_of_day_summary():
    import pytz
    from datetime import datetime, time, timedelta
    from B_Module_Files.natural_language_module import NaturalLanguageModule

    user_id = g.current_user['user_id']
    tz = pytz.timezone("America/Chicago")
    now_local = datetime.now(tz)

    # Ensure that the endpoint is accessed after 5:00 PM local time.
    if now_local.time() < time(17, 0, 0):
        return jsonify({"error": "End of Day Summary is available after 5:00 PM local time."}), 400

    # Define the summary window: from yesterday at 5:00 PM to today at 5:00 PM.
    today = now_local.date()
    end_time_local = tz.localize(datetime.combine(today, time(17, 0, 0)))
    start_time_local = end_time_local - timedelta(days=1)
    start_time_utc = start_time_local.astimezone(pytz.utc)
    end_time_utc = end_time_local.astimezone(pytz.utc)

    # Query the sensor_data table for the previous 24 hours.
    summary_data = ""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT sensor_name, sensor_output, timestamp
                    FROM sensor_data
                    WHERE user_id = %s
                      AND timestamp >= %s
                      AND timestamp < %s
                    ORDER BY timestamp ASC
                """, (user_id, start_time_utc, end_time_utc))
                rows = cursor.fetchall()
                # Concatenate sensor records into a summary string.
                for row in rows:
                    ts = row['timestamp'].astimezone(tz).strftime("%Y-%m-%d %H:%M:%S")
                    summary_data += f"{ts} - {row['sensor_name']}: {row['sensor_output']}\n"
    except Exception as e:
        app.logger.error(f"Error fetching sensor data for end_of_day_summary: {e}", exc_info=True)
        return jsonify({"error": "Failed to fetch sensor data for summary."}), 500

    # Generate the summary using the NaturalLanguageModule.
    try:
        nlp = NaturalLanguageModule(model_name="gpt-4o-mini", logger=app.logger)
        prompt = (
            f"Generate an End of Day Summary for {end_time_local.strftime('%Y-%m-%d')} "
            f"using the following sensor data from the past 24 hours:\n{summary_data}"
        )
        summary_text = nlp.generate_summary(prompt)
    except Exception as e:
        app.logger.error(f"Error generating summary: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate summary."}), 500

    return jsonify({
        "end_of_day_summary": summary_text,
        "summary_date": end_time_local.strftime('%Y-%m-%d')
    }), 200
'''
# ------------------------------------------------------------------------------
# Send Feedback Endpoint
# ------------------------------------------------------------------------------   
@app.route('/send_feedback', methods=['POST'])
@token_required
def send_feedback():
    data = request.get_json() or {}
    feedback = data.get('feedback')

    if not feedback:
        app.logger.warning("No feedback provided.")
        return jsonify({"error": "No feedback provided"}), 400

    try:
        app.logger.info(f"Received feedback: {feedback}")
        feedback_json = json.dumps(feedback)

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO feedback_table (user_id, feedback_text, timestamp)
                    VALUES (%s, %s, NOW())
                """, (g.current_user['user_id'], feedback_json))
            conn.commit()

        return jsonify({"status": "Feedback processed successfully"}), 200
    except Exception as e:
        app.logger.error(f"Error processing feedback: {e}", exc_info=True)
        return jsonify({"error": "Failed to process feedback"}), 500

# ------------------------------------------------------------------------------
# Collaborative Update Endpoint
# ------------------------------------------------------------------------------
@app.route('/collaborative_update', methods=['GET'])
def collaborative_update():
    updates = [
        {
            "id": 101,
            "title": "External Update #1",
            "body": "This is the first external collaboration update.",
            "features": [0.4, 0.5, 0.6]
        },
        {
            "id": 102,
            "title": "External Update #2",
            "body": "This is the second external collaboration update.",
            "features": [0.2, 0.1, 0.5]
        }
    ]
    return jsonify(updates), 200

REQUIRED_FEATURE_LIST = 18

# ------------------------------------------------------------------------------
# RL Predict Endpoint (Dynamic Use-Case, Authenticated)
# ------------------------------------------------------------------------------
@app.route("/rl_predict", methods=["POST"])
@token_required
def rl_predict():
    """
    Allows partial or excessively long 'obs' arrays:
      - We'll pad with 0.0 if obs is shorter than expected
      - We'll truncate obs if it's longer than expected
    
    Example JSON:
    {
      "obs": [0.12, 0.05, ...],
      "sensors": ["dof_1", "vel_1", ...]  <-- optional
    }
    """
    data = request.get_json() or {}
    obs = data.get("obs", [])

    # Log the user invoking the endpoint
    app.logger.info(f"User {g.current_user['user_id']} invoking RL predict")

    # Optionally detect use-case from "sensors" list
    sensor_metadata = data.get("sensors", [])
    if sensor_metadata and isinstance(sensor_metadata, list):
        from B_Module_Files.use_case_selector_module import UseCaseSelector
        selector = UseCaseSelector(sensor_metadata)
        current_use_case = selector.get_use_case()
        app.logger.info(f"Detected use case from sensor metadata: {current_use_case}")
    else:
        # Fallback to environment variable with default set to 'warehouse'
        current_use_case = os.getenv("USE_CASE", "warehouse").lower()
        app.logger.info(f"Using default use case from environment: {current_use_case}")

    # Decide the expected observation length based on the current use case
    if current_use_case == "9dof":
        expected_length = 18
    elif current_use_case == "6dof":
        expected_length = 12
    elif current_use_case == "drone":
        expected_length = 8
    elif current_use_case == "urban":
        # 5 DOFs + 5 velocities = 10 total
        expected_length = 10
    elif current_use_case == "warehouse":
        expected_length = 16
    else:
        # Fallback for unknown use cases
        expected_length = 16

    if not isinstance(obs, list):
        return jsonify({"error": "No valid 'obs' list provided."}), 400

    global rl_model
    if rl_model is None:
        return jsonify({"error": "RL model not loaded."}), 500

    try:
        import numpy as np
        # Convert obs to a float32 array
        obs_array = np.array(obs, dtype=np.float32)

        # Use the data_preprocessing_module to pad or truncate the observation array
        aligned_obs = preprocessor.align_features_for_inference(
            obs_array,
            required_features=expected_length,
            fill_value=0.0
        )

        # Ensure the array is 2D for the RL model
        if aligned_obs.ndim == 1:
            aligned_obs = aligned_obs.reshape(1, -1)

        # Run RL inference
        action, _ = rl_model.predict(aligned_obs, deterministic=True)
        action_list = action.flatten().tolist()

        return jsonify({
            "action": action_list,
            "use_case": current_use_case,
            "message": "RL inference successful."
        }), 200

    except Exception as e:
        app.logger.error(f"RL model inference error: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred."}), 500

# ------------------------------------------------------------------------------
# Benchmark Endpoint (Personalized)
# ------------------------------------------------------------------------------
@app.route('/benchmark', methods=['POST'])
@token_required
def run_benchmark():
    endpoints = os.getenv('BENCHMARK_ENDPOINTS', '').split(',')
    payload_str = os.getenv('BENCHMARK_PAYLOAD', '{}')
    try:
        payload = json.loads(payload_str)
    except Exception as ex:
        payload = {}
    iterations = int(os.getenv('BENCHMARK_ITERATIONS', "10"))

    if not endpoints:
        return jsonify({"message": "No endpoints configured"}), 400

    # Personalize the payload by adding the current user's ID
    payload['user_id'] = g.current_user['user_id']

    from B_Module_Files.benchmarking_tool_module import BenchmarkingTool
    tool = BenchmarkingTool(endpoints=endpoints, payload=payload, iterations=iterations)
    results = tool.benchmark()
    tool.save_results(results)
    
    return jsonify({
        "message": "Benchmarking completed",
        "user_id": g.current_user['user_id'],
        "results": results
    }), 200

# ------------------------------------------------------------------------------
# System Logs Endpoint (Requires JWT token, Personalized)
# ------------------------------------------------------------------------------
@app.route('/logs', methods=['GET'])
@token_required
def logs():
    """
    Fetches system logs for the current user in descending timestamp order.
    Allows optional search by message and filter by log_type, plus pagination.
    Renders logs.html with the logs and pagination context.
    The message field now combines the original message with resolution or 
    recommendation details from the notes column.
    """
    try:
        user_id = g.current_user['user_id']

        # 1) Read query params
        search_query = request.args.get('search', '').strip()
        log_type = request.args.get('log_type', '').strip()
        page = request.args.get('page', 1, type=int)

        limit = 15
        offset = (page - 1) * limit

        # 2) Build dynamic WHERE clause
        base_query = """
            SELECT id, log_type, message, is_resolved, notes, timestamp
            FROM system_logs
            WHERE user_id = %s
        """
        count_query = """
            SELECT COUNT(*) AS total
            FROM system_logs
            WHERE user_id = %s
        """
        params = [user_id]

        # Optional search by message
        if search_query:
            base_query += " AND message ILIKE %s"
            count_query += " AND message ILIKE %s"
            params.append(f"%{search_query}%")

        # Optional filter by log_type
        if log_type:
            base_query += " AND log_type = %s"
            count_query += " AND log_type = %s"
            params.append(log_type)

        # 3) Add order, limit, offset
        base_query += " ORDER BY timestamp DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Fetch the logs
                cursor.execute(base_query, params)
                rows = cursor.fetchall()

            # For pagination: count total rows (w/o limit/offset)
            # but using the same filters
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Remove the last two parameters (limit, offset)
                count_params = params[:-2]
                cursor.execute(count_query, count_params)
                total_count = cursor.fetchone()['total']

        # 4) Build logs_data for rendering
        logs_data = []
        for r in rows:
            # Combine the original message with resolution/recommendation notes, if present.
            full_message = r["message"] or ""
            if r["notes"]:
                full_message += " | Resolution/Recommendation: " + r["notes"]
            logs_data.append({
                "id": r["id"],
                "type": r["log_type"],
                "message": full_message,
                "resolved": r["is_resolved"],
                "timestamp": r["timestamp"]
            })

        # 5) Calculate total pages
        import math
        total_pages = math.ceil(total_count / limit)

        # 6) Prepare pagination context
        pagination = {
            "current_page": page,
            "total_pages": total_pages
        }

        # 7) Render logs.html, passing logs, search/filter, pagination
        return render_template(
            'logs.html',
            logs=logs_data,
            search_query=search_query,
            log_type=log_type,
            pagination=pagination
        )

    except Exception as e:
        app.logger.error(f"Error in /logs route: {e}", exc_info=True)
        return jsonify({"error": "Failed to fetch logs"}), 500

# ------------------------------------------------------------------------------
# Alerts Page (GET & POST in one route)
# ------------------------------------------------------------------------------
@app.route('/alerts', methods=['GET', 'POST'])
@token_required
def alerts():
    """
    GET: Renders alerts.html with the user's alert preferences.
         If no alert preferences exist, sets defaults ("big_anomalies" and 
         "predictive_maintenance" with email alerts enabled).

    POST: Processes the form submission from alerts.html and saves the updated 
          preferences (including email_alerts, sms_alerts, phone_number).
          Then redirects back to /alerts.
    """
    try:
        user_id = g.current_user['user_id']

        # -----------------------------------
        # 1) If GET => Show the alerts page
        # -----------------------------------
        if request.method == 'GET':
            app.logger.debug("Entered /alerts route [GET]")
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT preferences, email_alerts, sms_alerts, phone_number
                        FROM alert_preferences
                        WHERE user_id = %s
                        LIMIT 1
                    """, (user_id,))
                    result = cursor.fetchone()

                    if result is None or not result.get("preferences"):
                        default_alerts = ["big_anomalies", "predictive_maintenance"]
                        cursor.execute("""
                            INSERT INTO alert_preferences (
                                user_id, preferences, email_alerts, sms_alerts, phone_number
                            )
                            VALUES (%s, %s, %s, %s, %s)
                        """, (user_id, json.dumps(default_alerts), True, False, None))
                        conn.commit()
                        user_settings = {
                            "preferences": default_alerts,
                            "email_alerts": True,
                            "sms_alerts": False,
                            "phone_number": None
                        }
                    else:
                        try:
                            raw_pref = result["preferences"]
                            if isinstance(raw_pref, str):
                                preferences_list = json.loads(raw_pref)
                            else:
                                preferences_list = raw_pref if raw_pref else []

                            user_settings = {
                                "preferences": preferences_list,
                                "email_alerts": result.get("email_alerts", True),
                                "sms_alerts": result.get("sms_alerts", False),
                                "phone_number": result.get("phone_number")
                            }
                        except json.JSONDecodeError:
                            user_settings = {
                                "preferences": ["big_anomalies", "predictive_maintenance"],
                                "email_alerts": True,
                                "sms_alerts": False,
                                "phone_number": None
                            }
            return render_template("alerts.html", user_settings=user_settings)

        # -----------------------------------
        # 2) If POST => Save the new preferences
        # -----------------------------------
        else:
            app.logger.debug("Entered /alerts route [POST]")
            selected_alerts = request.form.getlist('alerts')  # e.g. ['warning','critical']
            email_alerts = request.form.get('email_alerts', '').lower() in ['on', 'true', 'checked', '1']
            sms_alerts = request.form.get('sms_alerts', '').lower() in ['on', 'true', 'checked', '1']
            phone_number = request.form.get('phone_number', '').strip() or None

            app.logger.info(
                f"[POST /alerts] user {user_id}, selected_alerts={selected_alerts}, "
                f"email_alerts={email_alerts}, sms_alerts={sms_alerts}, phone_number={phone_number}"
            )

            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # Debug log: see which user and what data we’re about to save
                    app.logger.debug(
                        f"[update_alerts] Attempting to update user_id={user_id} "
                        f"with alerts={selected_alerts}, email={email_alerts}, "
                        f"sms={sms_alerts}, phone={phone_number}"
                    )

                    # Attempt the UPDATE
                    cursor.execute(
                        """
                        UPDATE alert_preferences
                        SET preferences = %s,
                            email_alerts = %s,
                            sms_alerts = %s,
                            phone_number = %s
                        WHERE user_id = %s
                        """,
                        (json.dumps(selected_alerts), email_alerts, sms_alerts, phone_number, user_id)
                    )

                    # Log how many rows were updated
                    app.logger.debug(f"[update_alerts] rowcount after UPDATE={cursor.rowcount}")

                    # If rowcount == 0 => no row updated => do an INSERT
                    if cursor.rowcount == 0:
                        app.logger.debug("[update_alerts] No row updated, inserting new row in alert_preferences.")
                        cursor.execute(
                            """
                            INSERT INTO alert_preferences (
                                user_id, preferences, email_alerts, sms_alerts, phone_number
                            )
                            VALUES (%s, %s, %s, %s, %s)
                            """,
                            (user_id, json.dumps(selected_alerts), email_alerts, sms_alerts, phone_number)
                        )

                    conn.commit()

            # Confirm success
            app.logger.debug(f"[update_alerts] Successfully saved alerts for user_id={user_id}.")
            flash("Alert preferences saved!", "success")
            return redirect(url_for('alerts'))

    except Exception as e:
        app.logger.error(f"Error in /alerts route: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred."}), 500

# --------------------------------------------------------
# Send Action Endpoint (Requires JWT token)
# --------------------------------------------------------
@app.route("/send_action", methods=["POST"])
@token_required
def send_action():
    """
    Handle the user action form submission.
    The form's <input name="action"> text is retrieved from request.form.
    Returns a confirmation page instead of a JSON response.
    """

    # Define a set of actions that DO need a system response.
    ACTIONS_NEED_RESPONSE = {"submit_form", "start_simulation", "some_other_action"}

    try:
        user_action = request.form.get("action", "").strip()
        if not user_action:
            return render_template("action_confirmation.html", message="No action provided"), 400

        # Decide which initial system_response to store:
        if user_action in ACTIONS_NEED_RESPONSE:
            system_response = "No response yet"
        else:
            system_response = "No response needed"

        # ---------------------------------------------------------------
        # Store the user action in the "user_actions" table along with the user's ID
        # and the determined system_response.
        # ---------------------------------------------------------------
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO user_actions (user_id, action_text, system_response, timestamp)
                        VALUES (%s, %s, %s, NOW())
                    """, (g.current_user['user_id'], user_action, system_response))
                conn.commit()
        except Exception as db_err:
            app.logger.error(f"Failed to store user action in DB: {db_err}", exc_info=True)
            return render_template("action_confirmation.html", message="Action received but failed to store in database."), 500

        # ---------------------------------------------------------------
        # Return success response as a confirmation page.
        # ---------------------------------------------------------------
        return render_template("action_confirmation.html", message=f"Action '{user_action}' received and stored.")
    except Exception as e:
        app.logger.error(f"Error in /send_action: {e}", exc_info=True)
        return render_template("action_confirmation.html", message="An unexpected error occurred."), 500

################################################################################
# AGENTS & CHAT ENDPOINTS BEGIN
################################################################################
# ------------------------------------------------------------------------------
# Agent Orchestration Endpoint (Production-Ready)
# ------------------------------------------------------------------------------
@app.route("/agent_orchestration", methods=["POST"])
@token_required
def agent_orchestration():
    """
    Receives a user message, invokes our Agents pipeline, and returns the final agent response.
    Example JSON:
    {
      "message": "How can I improve sensor X's energy efficiency?"
    }
    By default, we now call the 'energy_efficiency_agent' agent.
    """
    user_id = g.current_user['user_id']
    data = request.get_json() or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    # 1) Store the user message in chat_logs
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO chat_logs (user_id, speaker, message, created_at)
                    VALUES (%s, %s, %s, NOW())
                """, (user_id, "You", user_message))
        conn.commit()
    except Exception as e:
        app.logger.warning(f"Error storing user message in chat_logs: {e}", exc_info=True)

    # 2) Invoke the Agents pipeline (defaults to "energy_efficiency_agent")
    try:
        final_response_text = run_sync_request(
            user_text=user_message,
            agent_name="energy_efficiency_agent"
        )
    except Exception as e:
        app.logger.error(f"Error during agent orchestration: {e}", exc_info=True)
        return jsonify({"error": f"Agent orchestration failed: {str(e)}"}), 500

    # 3) Store the agent's final reply in chat_logs
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO chat_logs (user_id, speaker, message, created_at)
                    VALUES (%s, %s, %s, NOW())
                """, (user_id, "System", final_response_text))
        conn.commit()
    except Exception as e:
        app.logger.warning(f"Error storing system reply in chat_logs: {e}", exc_info=True)

    # 4) Return the final agent response to the frontend
    return jsonify({
        "message": "Agent orchestration complete.",
        "reply": final_response_text
    }), 200

# ------------------------------------------------------------------------------
# Agent Interaction Endpoint
# ------------------------------------------------------------------------------
@app.route("/agent_interaction", methods=["POST"])
@token_required
def agent_interaction():
    """
    Accepts JSON:
      {
        "message": "User’s text or command",
        "agent_name": "energy_efficiency_agent"  # optional, defaults to "energy_efficiency_agent"
      }
    Uses run_sync_request() from agents_runner_module to process the user message with the specified agent.
    Returns a JSON with "agent_reply".
    """
    user_id = g.current_user["user_id"]
    data = request.get_json() or {}

    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    # Determine agent name; default to "energy_efficiency_agent" if not provided
    agent_name = data.get("agent_name", "energy_efficiency_agent")

    try:
        final_output = run_sync_request(user_message, agent_name=agent_name)
        app.logger.info(f"Agent '{agent_name}' processed request from user {user_id}.")
        return jsonify({"agent_reply": final_output}), 200
    except Exception as e:
        app.logger.error(f"Error in agent_interaction endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

################################################################################
# CHAT ENDPOINTS BEGIN
################################################################################
# ------------------------------------------------------------------------------
# Voice Command Endpoint (Personalized, Chat-Style)
# ------------------------------------------------------------------------------
@app.route("/voice_command", methods=["POST"])
@token_required
def voice_command():
    """
    Receives voice transcript data from the frontend.
    Example JSON:
    {
      "transcript": "#calibrate sensor X"
    }
    1) Insert the user's voice transcript into chat_logs (speaker="User [Voice]").
    2) If it starts with '#', log into user_actions (No response yet).
    3) If no '#' but GPT thinks it's definitely a command => prompt user to resend with '#'.
    4) If known keywords => short-circuit with that response; else fallback to the appropriate agent
       (use "visual_agent" if the transcript is visual-related, otherwise use "energy_efficiency_agent")
       with KPI context.
    5) If it was a command => update user_actions with final system_response.
    6) Insert final system_response into chat_logs (speaker="System").
    """
    app.logger.debug("[voice_command] Unique debug line 1234! Entering voice_command...")

    from B_Module_Files.natural_language_module import NaturalLanguageModule
    from B_Module_Files.anomaly_maintenance_helper_module import (
        fetch_recent_anomalies,
        fetch_predictive_maintenance_status
    )
    from B_Module_Files.agents_runner_module import run_sync_request
    from B_Module_Files.get_kpi_context_module import get_kpi_context
    from B_Module_Files.database_module import get_db_connection
    import asyncio

    user_id = g.current_user['user_id']
    data = request.get_json() or {}
    transcript = data.get('transcript', '').strip()

    app.logger.debug(f"[voice_command] user_id={user_id}, transcript='{transcript}'")

    if not transcript:
        return jsonify({"error": "No transcript provided."}), 400

    # 1) Insert the user's voice transcript into chat_logs (speaker="User [Voice]")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO chat_logs (user_id, speaker, message, created_at)
                    VALUES (%s, %s, %s, NOW())
                """, (user_id, "User [Voice]", transcript))
            conn.commit()
    except Exception as e:
        app.logger.error(f"[voice_command] Error storing voice transcript in chat_logs: {e}", exc_info=True)
        return jsonify({"error": "Failed to store voice transcript."}), 500

    # 2) Check if transcript is a '#' command => user_actions
    is_command = transcript.startswith("#")
    action_id = None
    if is_command:
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO user_actions (user_id, action_text, system_response, timestamp)
                        VALUES (%s, %s, %s, NOW())
                        RETURNING id
                    """, (user_id, f"[Voice] {transcript}", "No response yet"))
                    action_id = cursor.fetchone()["id"]
                conn.commit()
        except Exception as e:
            app.logger.error(f"[voice_command] Error storing voice command in user_actions: {e}", exc_info=True)
            return jsonify({"error": "Failed to store voice command."}), 500

    # 3) If not a command, use GPT to see if user might've forgotten '#'
    if not is_command:
        try:
            checker = NaturalLanguageModule(model_name="gpt-4o-mini", logger=app.logger)
            cmd_check = checker.check_if_command(transcript)
            if asyncio.iscoroutine(cmd_check):
                cmd_check = asyncio.run(cmd_check)
            if cmd_check:
                return jsonify({
                    "message": "Voice command processed.",
                    "reply": "Did you mean to execute that as a command? Please resend starting with '#'."
                }), 200
        except Exception as e:
            app.logger.warning(f"[voice_command] Error checking voice command with GPT: {e}")

    # 4) Check for known keywords/phrases => short-circuit
    system_response = None
    lower_transcript = transcript.lower()
    if "calibrate" in lower_transcript:
        system_response = "Sure, calibrating now."
    elif "status" in lower_transcript or "health" in lower_transcript:
        system_response = "System is functioning properly, no anomalies."
    elif "maintenance" in lower_transcript:
        pm_info = fetch_predictive_maintenance_status(user_id)
        system_response = pm_info or "No predictive maintenance items found."
    elif "anomalies" in lower_transcript:
        anomalies = fetch_recent_anomalies(user_id)
        if anomalies:
            system_response = f"Recent anomalies detected: {', '.join(anomalies)}"
        else:
            system_response = "No recent anomalies found."

    # 5) If no known phrase => fallback to appropriate agent with KPI context.
    if not system_response:
        # Check if the transcript contains visual-related keywords.
        visual_keywords = ["see", "look", "camera", "image", "visual"]
        if any(keyword in lower_transcript for keyword in visual_keywords):
            fallback_agent = "visual_agent"
        else:
            fallback_agent = "energy_efficiency_agent"
        try:
            kpi_context = get_kpi_context(user_id)
            fallback_prompt = (
                "SYSTEM NOTE: Please incorporate the following KPI data into your response:\n"
                f"{kpi_context}\n\nUser Query: {transcript}"
            )
            system_response = run_sync_request(fallback_prompt, fallback_agent)
        except Exception as e:
            app.logger.error(f"[voice_command] Error running {fallback_agent} with KPI context: {e}", exc_info=True)
            system_response = "I'm sorry, I encountered an error while processing your voice command."

    # 6) If it was a command => update user_actions row with final system_response
    if is_command and action_id:
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE user_actions
                        SET system_response = %s
                        WHERE id = %s
                    """, (system_response, action_id))
                conn.commit()
        except Exception as e:
            app.logger.error(f"[voice_command] Error updating user_actions: {e}", exc_info=True)

    # 7) Insert final system_response into chat_logs (speaker="System")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO chat_logs (user_id, speaker, message, created_at)
                    VALUES (%s, %s, %s, NOW())
                """, (user_id, "System", system_response))
            conn.commit()
    except Exception as e:
        app.logger.error(f"[voice_command] Error storing system response in chat_logs: {e}", exc_info=True)

    # 8) Return final JSON
    return jsonify({
        "message": "Voice command processed.",
        "reply": system_response
    }), 200

# ------------------------------------------------------------------------------
# Chat Endpoint (Text-based)
# ------------------------------------------------------------------------------
@app.route("/chat_endpoint", methods=["POST"])
@token_required
def chat_endpoint():
    """
    Handles text-based chat messages from the user.
    1) Insert user message into chat_logs (speaker="You").
    2) If it starts with '#', log in user_actions (No response yet).
    3) If no '#' but GPT thinks it's definitely a command => prompt user to resend with '#'.
    4) If known keywords => short-circuit response; else fallback to the appropriate agent 
       (use "visual_agent" if the query is visual-related, otherwise use "energy_efficiency_agent")
       with KPI context.
    5) If it was a command => update user_actions with final system_response.
    6) Insert system_response into chat_logs (speaker="System").
    """
    app.logger.debug("[chat_endpoint] Unique debug line 9999! Entering chat_endpoint...")

    from B_Module_Files.natural_language_module import NaturalLanguageModule
    from B_Module_Files.anomaly_maintenance_helper_module import (
        fetch_recent_anomalies,
        fetch_predictive_maintenance_status
    )
    from B_Module_Files.agents_runner_module import run_sync_request
    from B_Module_Files.get_kpi_context_module import get_kpi_context
    import asyncio

    user_id = g.current_user['user_id']
    data = request.get_json() or {}
    user_message = data.get("message", "").strip()
    app.logger.debug(f"[chat_endpoint] user_id={user_id}, user_message='{user_message}'")

    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    # 1) Insert user message into chat_logs (speaker="You")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO chat_logs (user_id, speaker, message, created_at)
                    VALUES (%s, %s, %s, NOW())
                """, (user_id, "You", user_message))
            conn.commit()
    except Exception as e:
        app.logger.error(f"[chat_endpoint] Error storing user message: {e}", exc_info=True)
        return jsonify({"error": "Failed to store user message."}), 500

    # 2) If it starts with '#', log in user_actions
    is_command = user_message.startswith("#")
    action_id = None
    if is_command:
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO user_actions (user_id, action_text, system_response, timestamp)
                        VALUES (%s, %s, %s, NOW())
                        RETURNING id
                    """, (user_id, user_message, "No response yet"))
                    action_id = cursor.fetchone()["id"]
                conn.commit()
        except Exception as e:
            app.logger.error(f"[chat_endpoint] Error logging user action: {e}", exc_info=True)

    # 2a) If not '#' prefix, do a GPT check to see if user forgot
    system_reply = None
    if not is_command:
        try:
            checker = NaturalLanguageModule(model_name="gpt-4o-mini", logger=app.logger)
            cmd_check = checker.check_if_command(user_message)
            if asyncio.iscoroutine(cmd_check):
                cmd_check = asyncio.run(cmd_check)
            if cmd_check:
                system_reply = "Did you mean to execute that as a command? (Please resend with '#')"
        except Exception as e:
            app.logger.warning(f"[chat_endpoint] Error checking command with GPT: {e}")

    # 3) If system_reply is still None => handle known keywords or fallback
    if system_reply is None:
        lower_msg = user_message.lower()
        if "calibrate" in lower_msg:
            system_reply = "Sure, calibrating now."
        elif "status" in lower_msg or "health" in lower_msg:
            system_reply = "System is functioning properly, no anomalies."
        elif "maintenance" in lower_msg:
            pm_info = fetch_predictive_maintenance_status(user_id)
            system_reply = pm_info or "No predictive maintenance items found."
        elif "anomalies" in lower_msg:
            anomalies = fetch_recent_anomalies(user_id)
            if anomalies:
                system_reply = f"Recent anomalies detected: {', '.join(anomalies)}"
            else:
                system_reply = "No recent anomalies found."
        else:
            # Determine fallback agent based on visual keywords
            visual_keywords = ["see", "look", "camera", "image", "visual"]
            if any(keyword in lower_msg for keyword in visual_keywords):
                fallback_agent = "visual_agent"
            else:
                fallback_agent = "energy_efficiency_agent"
            try:
                kpi_context = get_kpi_context(user_id)
                fallback_prompt = (
                    "SYSTEM NOTE: Please incorporate the following KPI data into your response:\n"
                    f"{kpi_context}\n\nUser Query: {user_message}"
                )
                system_reply = run_sync_request(fallback_prompt, fallback_agent)
            except Exception as e:
                app.logger.error(f"[chat_endpoint] Error running {fallback_agent} with KPI context: {e}", exc_info=True)
                system_reply = "I'm sorry, I encountered an error while processing your request."

    # 4) Insert system reply into chat_logs (speaker="System")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO chat_logs (user_id, speaker, message, created_at)
                    VALUES (%s, %s, %s, NOW())
                """, (user_id, "System", system_reply))
            conn.commit()
    except Exception as e:
        app.logger.error(f"[chat_endpoint] Error storing system chat message: {e}", exc_info=True)
        return jsonify({"reply": system_reply, "warning": "Failed to store system reply."}), 200

    # 5) If it was a command => update user_actions with final system_response
    if is_command and action_id:
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE user_actions
                        SET system_response = %s
                        WHERE id = %s
                    """, (system_reply, action_id))
                conn.commit()
        except Exception as e:
            app.logger.error(f"[chat_endpoint] Error updating user_actions: {e}", exc_info=True)

    return jsonify({"reply": system_reply}), 200

# ------------------------------------------------------------------------------
# Chat History Endpoint
# ------------------------------------------------------------------------------
@app.route("/chat_history", methods=["GET"])
@token_required
def chat_history():
    """
    Returns ALL chat messages for the current user in ascending order
    (oldest → newest). No speaker filters, no row limit.
    """
    user_id = g.current_user["user_id"]
    app.logger.debug(f"[chat_history] user_id={user_id}, fetching ALL chat logs in ascending order.")

    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT speaker, message, created_at
                    FROM chat_logs
                    WHERE user_id = %s
                    ORDER BY created_at ASC
                """, (user_id,))
                rows = cursor.fetchall()
                app.logger.debug(f"[chat_history] Fetched {len(rows)} rows for user_id={user_id}.")
        if not rows:
            app.logger.warning(f"[chat_history] No chat history found for user {user_id}.")
        history = []
        for row in rows:
            try:
                timestamp_str = row["created_at"].isoformat() if row["created_at"] else None
            except Exception as ts_err:
                app.logger.error(f"[chat_history] Error formatting timestamp: {ts_err}", exc_info=True)
                timestamp_str = None
            history.append({
                "speaker": row["speaker"],
                "message": row["message"],
                "timestamp": timestamp_str
            })
        app.logger.info(f"[chat_history] Returning {len(history)} total lines for user {user_id}.")
        return jsonify(history), 200
    except Exception as e:
        app.logger.error(f"[chat_history] Error fetching chat history: {e}", exc_info=True)
        return jsonify({"error": "Failed to fetch chat history"}), 500

################################################################################
# AGENTS & CHAT ENDPOINTS END
################################################################################

# ------------------------------------------------------------------------------
# Action Logs Endpoint (Personalized) - Now with Pagination & Search
# ------------------------------------------------------------------------------
@app.route('/action_logs', methods=['GET'])
@token_required
def action_logs():
    """
    Renders action_logs.html which displays past user actions along with
    any system-generated notes regarding the action's execution and results,
    limited to 15 entries per page. Also supports a simple 'search' query.
    """
    try:
        user_id = g.current_user['user_id']
        import math
        import pytz

        # 1) Parse query params
        page = request.args.get('page', 1, type=int)
        search_query = request.args.get('search', '').strip()

        # 2) Basic pagination setup
        page_size = 15
        offset = (page - 1) * page_size

        # 3) Build base query + optional search filter
        base_sql = """
            SELECT action_text, timestamp, system_response
            FROM user_actions
            WHERE user_id = %s
        """
        count_sql = """
            SELECT COUNT(*) AS total_count
            FROM user_actions
            WHERE user_id = %s
        """

        # If we have a search_query, apply it to both queries
        if search_query:
            base_sql += " AND (action_text ILIKE %s OR system_response ILIKE %s)"
            count_sql += " AND (action_text ILIKE %s OR system_response ILIKE %s)"
            base_sql += " ORDER BY timestamp DESC LIMIT %s OFFSET %s"
        else:
            base_sql += " ORDER BY timestamp DESC LIMIT %s OFFSET %s"

        # 4) Execute queries
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Count total rows first
                if search_query:
                    cursor.execute(count_sql, (user_id, f"%{search_query}%", f"%{search_query}%"))
                else:
                    cursor.execute(count_sql, (user_id,))
                row_count_res = cursor.fetchone()
                total_rows = row_count_res["total_count"] if row_count_res else 0

                # Now fetch the logs with limit/offset
                if search_query:
                    cursor.execute(base_sql, (
                        user_id,
                        f"%{search_query}%",
                        f"%{search_query}%",
                        page_size,
                        offset
                    ))
                else:
                    cursor.execute(base_sql, (
                        user_id,
                        page_size,
                        offset
                    ))
                logs = cursor.fetchall()

        # 5) Calculate total_pages
        total_pages = math.ceil(total_rows / page_size) if total_rows else 1
        if total_pages < 1:
            total_pages = 1

        # 6) Build pagination dict
        pagination = {
            "current_page": page,
            "total_pages": total_pages
        }

        # 7) Timezone for display
        local_tz = pytz.timezone("America/Chicago")

        return render_template(
            "action_logs.html",
            action_logs=logs,
            local_tz=local_tz,
            pagination=pagination,
            search_query=search_query
        )

    except Exception as e:
        app.logger.error(f"Error in /action_logs GET route: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred."}), 500
    
# ------------------------------------------------------------------------------
# KPI Charts Endpoint (Personalized - JSON version)
# Updated to show cumulative energy saved (using energy_saved_increment) over the last 7 days.
# ------------------------------------------------------------------------------
@app.route("/kpi_charts", methods=["GET"])
@token_required
def kpi_charts():
    """
    Fetches KPI data for the current user:
      - avg_efficiency and system_health_score from the most recent record.
      - total_energy_saved as a cumulative sum (of energy_saved_increment) over the last 7 days.
    """
    user_id = g.current_user['user_id']

    summary_metrics = {
        'avg_energy_efficiency': None,
        'total_energy_saved': None,
        'system_health_score': None
    }
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get the latest snapshot for avg_efficiency and system_health
                cursor.execute("""
                    SELECT avg_efficiency, system_health
                    FROM user_kpi_history
                    WHERE user_id = %s
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (user_id,))
                latest_kpi = cursor.fetchone()

                # Calculate cumulative energy saved over the last 7 days using energy_saved_increment
                cursor.execute("""
                    SELECT COALESCE(SUM(energy_saved_increment::float), 0) AS cumulative_energy_saved
                    FROM user_kpi_history
                    WHERE user_id = %s
                      AND timestamp >= NOW() - INTERVAL '7 days'
                """, (user_id,))
                energy_sum = cursor.fetchone()

        if latest_kpi and energy_sum:
            summary_metrics = {
                'avg_energy_efficiency': round(float(latest_kpi['avg_efficiency']), 1),
                'total_energy_saved': round(float(energy_sum['cumulative_energy_saved']), 1),
                'system_health_score': round(float(latest_kpi['system_health']), 1)
            }
    except Exception as e:
        app.logger.error(f"Error fetching KPI data: {e}", exc_info=True)
        return jsonify({"error": "Failed to fetch KPI data"}), 500

    # Build time series data for the last 7 days
    kpi_data = {
        "timestamps": [],
        "avg_eff_history": [],
        "energy_saved_history": [],
        "system_health_history": []
    }
    cumulative_energy_saved = 0.0
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT timestamp,
                           avg_efficiency,
                           energy_saved_increment,
                           system_health
                    FROM user_kpi_history
                    WHERE user_id = %s
                      AND timestamp >= NOW() - INTERVAL '7 days'
                    ORDER BY timestamp ASC
                """, (user_id,))
                kpi_rows = cursor.fetchall()

        for r in kpi_rows:
            kpi_data["timestamps"].append(r["timestamp"].strftime("%Y-%m-%d %H:%M:%S"))
            kpi_data["avg_eff_history"].append(float(r["avg_efficiency"]))
            cumulative_energy_saved += float(r["energy_saved_increment"])
            kpi_data["energy_saved_history"].append(cumulative_energy_saved)
            kpi_data["system_health_history"].append(float(r["system_health"]))
    except Exception as e:
        app.logger.error(f"Error fetching KPI time series data: {e}", exc_info=True)

    # Determine milestone text based on KPI values.
    milestone_text = None
    if summary_metrics['total_energy_saved'] and summary_metrics['total_energy_saved'] > 500:
        milestone_text = f"Over {summary_metrics['total_energy_saved']} units of energy saved—great job!"
    elif summary_metrics['avg_energy_efficiency'] and summary_metrics['avg_energy_efficiency'] >= 95.0:
        milestone_text = f"We just hit {summary_metrics['avg_energy_efficiency']:.1f}% efficiency—new record!"

    return jsonify({
        "kpi_data": kpi_data,
        "current_range": "7-day",
        "summary_metrics": summary_metrics,
        "milestone": milestone_text
    }), 200

# ------------------------------------------------------------------------------
# Real-Time Control & Anomaly Detection (Dynamic Use-Case with Severity, Authenticated)
# ------------------------------------------------------------------------------
from B_Module_Files.natural_language_module import NaturalLanguageModule  # fallback for dynamic messages
import json

@app.route("/real_time_control", methods=["POST"])
@token_required
def real_time_control():
    """
    1) Detect DOF count from pos/vel lengths (4=drone, 5=urban, 6=6dof, 8=warehouse, 9=9dof).
    2) If detection is "unknown", fallback to the USE_CASE environment variable (default: 'warehouse').
    3) For 'urban', skip anomaly detection but still run RL inference.
    4) If an anomaly is detected (medium/high), store in system_logs and optionally in chat_logs
       based on user alert preferences. If no direct logic matches, fallback to an AI-based
       message from the NaturalLanguageModule.
    """
    user_id = g.current_user['user_id']
    app.logger.info(f"[real_time_control] User {user_id} invoked endpoint.")

    data = request.get_json() or {}
    pos_array = data.get("pos", [])
    vel_array = data.get("vel", [])

    # Step 1: DOF-based detection
    pos_len = len(pos_array)
    vel_len = len(vel_array)
    dof_detected = "unknown"

    if pos_len == 9 and vel_len == 9:
        dof_detected = "9dof"
    elif pos_len == 6 and vel_len == 6:
        dof_detected = "6dof"
    elif pos_len == 5 and vel_len == 5:
        dof_detected = "urban"
    elif pos_len == 8 and vel_len == 8:
        dof_detected = "warehouse"
    elif pos_len == 4 and vel_len == 4:
        dof_detected = "drone"

    # Step 2: Fallback to environment variable if detection is "unknown"
    env_use_case = os.getenv("USE_CASE", "warehouse").lower()
    use_case = dof_detected if dof_detected != "unknown" else env_use_case

    # Label for logging
    display_use_case = {
        "9dof": "9DOF",
        "6dof": "6DOF",
        "urban": "Urban",
        "drone": "Drone"
    }.get(use_case, use_case.capitalize())

    app.logger.info(
        f"[real_time_control] pos_len={pos_len}, vel_len={vel_len}, "
        f"dof_detected={dof_detected}, env={env_use_case}, final_use_case={use_case}"
    )

    # Combine pos+vel into a single observation
    import numpy as np
    pos_np = np.array(pos_array, dtype=np.float32)
    vel_np = np.array(vel_array, dtype=np.float32)
    combined_obs = np.concatenate((pos_np, vel_np), axis=0)

    # Determine expected total features
    if use_case == "9dof":
        dof_count = 9
        expected_total = 2 * dof_count
    elif use_case == "6dof":
        dof_count = 6
        expected_total = 2 * dof_count
    elif use_case == "urban":
        # In code we see 5 DOFs + 5 velocities => 10, 
        # but the script sets 16 as a fallback for real_time_control. 
        # We'll align with the prior logic or correct it if needed:
        dof_count = 5
        # The old code had expected_total=16, but let's assume 10 is correct
        # to match the RL model. We'll keep it consistent with the script's comments.
        expected_total = 10
    elif use_case == "drone":
        dof_count = 4
        expected_total = 2 * dof_count  # 8
    else:  # fallback => warehouse
        dof_count = 8
        expected_total = 2 * dof_count

    # Align features (pad/truncate as needed)
    aligned_obs = preprocessor.align_features_for_inference(
        combined_obs.reshape(1, -1),
        required_features=expected_total,
        fill_value=0.0
    )
    app.logger.info(
        f"[real_time_control] aligned_obs.shape={aligned_obs.shape}, expected_total={expected_total}"
    )

    # For 'urban', skip anomaly detection but still run RL inference.
    anomaly_score = None
    anomaly_severity = "N/A"
    if use_case != "urban":
        global anomaly_detection_model
        if anomaly_detection_model is None:
            msg = f"{display_use_case} anomaly model not loaded."
            app.logger.error("[real_time_control] " + msg)
            return jsonify({"error": msg}), 500

        try:
            anomaly_score = anomaly_detection_model.predict(aligned_obs)
            anomaly_severity = get_anomaly_severity(anomaly_score[0], 0.5, 0.8)
            app.logger.info(
                f"[real_time_control] anomaly_score={anomaly_score[0]:.3f}, severity={anomaly_severity}"
            )
        except Exception as e:
            app.logger.error(
                f"[real_time_control] Error during anomaly detection: {e}",
                exc_info=True
            )
            return jsonify({"error": "An unexpected error occurred during anomaly detection."}), 500

    # RL inference for all use cases
    global rl_model
    if rl_model is None:
        msg = f"{display_use_case} RL model not loaded."
        app.logger.error("[real_time_control] " + msg)
        return jsonify({"error": msg}), 500

    try:
        action, _ = rl_model.predict(aligned_obs, deterministic=True)
        action_list = action.flatten().tolist()
        app.logger.info(f"[real_time_control] RL action={action_list}")
    except Exception as e:
        app.logger.error(f"[real_time_control] Error in RL inference: {e}", exc_info=True)
        return jsonify({"error": f"RL inference failed: {str(e)}"}), 500
    
    #####################
    # Step 3: If anomaly_severity is medium or high => log in system_logs, send email alert, and insert into chat_logs
    if anomaly_severity in ("medium", "high"):
        anomaly_msg = f"Anomaly detected (severity={anomaly_severity}). Score={anomaly_score[0]:.2f}"
        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Lock both tables to avoid concurrent duplicates
                    cursor.execute("LOCK TABLE chat_logs IN EXCLUSIVE MODE;")
                    cursor.execute("LOCK TABLE system_logs IN EXCLUSIVE MODE;")
                    
                    # Check for duplicates in both tables (last 5 minutes)
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT 1
                            FROM (
                                SELECT message, created_at AS ts
                                FROM chat_logs
                                WHERE user_id = %s
                                AND speaker = 'System [Anomaly]'
                                AND created_at >= NOW() - INTERVAL '5 minutes'
                                UNION ALL
                                SELECT message, timestamp AS ts
                                FROM system_logs
                                WHERE user_id = %s
                                AND log_type = 'Anomaly'
                                AND timestamp >= NOW() - INTERVAL '5 minutes'
                            ) AS recent_alerts
                            WHERE message = %s
                        ) AS alert_exists;
                    """, (user_id, user_id, anomaly_msg))
                    exists_result = cursor.fetchone()
                    
                    if not (exists_result and exists_result.get("alert_exists")):
                        # Insert into system_logs
                        cursor.execute("""
                            INSERT INTO system_logs
                                (user_id, log_type, message, timestamp)
                            VALUES
                                (%s, %s, %s, NOW())
                        """, (user_id, "Anomaly", anomaly_msg))
                        # Insert into chat_logs
                        chat_msg = f"System [Anomaly Alert]: {anomaly_msg}"
                        cursor.execute("""
                            INSERT INTO chat_logs (user_id, speaker, message, created_at)
                            VALUES (%s, %s, %s, NOW())
                        """, (user_id, "System", chat_msg))
                        insert_alert = True
                    else:
                        app.logger.info(f"Duplicate alert found for user {user_id}, skipping alert insertion and email.")
                        insert_alert = False
                conn.commit()
                
                # Retrieve the user's email
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT email
                        FROM users
                        WHERE id = %s
                        LIMIT 1
                    """, (user_id,))
                    user_row = cursor.fetchone()
                recipient = user_row["email"] if user_row and user_row.get("email") else "fallback@vestavio.com"
                
                # Retrieve the user's email alert preference
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT email_alerts
                        FROM alert_preferences
                        WHERE user_id = %s
                        LIMIT 1
                    """, (user_id,))
                    pref_row = cursor.fetchone()
                email_alerts_enabled = pref_row["email_alerts"] if pref_row and "email_alerts" in pref_row else True
                
                app.logger.info(f"Prepared to send anomaly alert email to {recipient} (Email alerts enabled: {email_alerts_enabled}).")
            
            if insert_alert and email_alerts_enabled:
                send_email(
                    recipient,
                    "Anomaly Alert",
                    f"Anomaly detected (severity={anomaly_severity}). Score={anomaly_score[0]:.2f}"
                )
        except Exception as e_alert:
            app.logger.error(f"[real_time_control] Error processing alert: {e_alert}", exc_info=True)

    else:
        # If no anomaly or low severity => optionally do a fallback chat message
        pass

    # Return final JSON
    return jsonify({
        "action": action_list,
        "anomaly_score": float(anomaly_score[0]) if anomaly_score is not None else None,
        "anomaly_severity": anomaly_severity,
        "message": f"Real-time {display_use_case} control inference success"
    }), 200

# ------------------------------------------------------------------------------
# Describe Image Endpoint (Using OpenAI Vision + Simple Local Color Detection)
# ------------------------------------------------------------------------------
from B_Module_Files.openai_vision_module import OpenAIVisionModule
import requests
import io
import base64
import numpy as np
from PIL import Image

@app.route("/describe_image", methods=["POST"])
@token_required
def describe_image():
    """
    Expects JSON with an "image_url" field.
    Uses OpenAI's vision-capable model (via OpenAIVisionModule) to analyze the image,
    plus a simple local color detection step to highlight the image’s dominant color.

    Example Request JSON:
    {
      "image_url": "https://example.com/path/to/image.jpg"
    }
    """
    data = request.get_json() or {}
    image_url = data.get("image_url")
    if not image_url:
        return jsonify({"error": "Missing 'image_url' field in request."}), 400

    try:
        user_id = g.current_user['user_id']
        app.logger.info(f"User {user_id} invoked describe_image with URL: {image_url}")

        # 1) Download the image for local color detection
        response = requests.get(image_url)
        response.raise_for_status()
        image_bytes = io.BytesIO(response.content)

        # 2) Convert to PIL for local color detection
        image = Image.open(image_bytes).convert("RGB")
        # Convert to numpy for a simple average color detection
        # (No need to fully resize for color detection; a small resize is enough)
        image_small = image.resize((224, 224))
        image_array = np.array(image_small, dtype=np.float32) / 255.0

        # Compute average color
        avg_color = np.mean(image_array, axis=(0, 1))
        # Identify which channel is largest
        channel_names = ["red", "green", "blue"]
        dominant_index = int(np.argmax(avg_color))
        dominant_color = channel_names[dominant_index]

        # 3) Use OpenAI Vision to describe the image
        # Create an instance of our hypothetical vision module
        vision = OpenAIVisionModule(model_name="gpt-4o-mini")
        # Provide a user prompt that references the color
        user_prompt = (
            f"Please describe this image. I believe it might have a dominant {dominant_color} color. "
            "What else do you see?"
        )

        # We can pass the image URL directly:
        openai_description = vision.analyze_image_url(
            image_url=image_url,
            user_prompt=user_prompt,
            detail="auto"  # or 'low'/'high'
        )

        # 4) Return final JSON response
        return jsonify({
            "dominant_color": dominant_color,
            "openai_vision_description": openai_description
        }), 200

    except Exception as e:
        app.logger.error(f"Error in /describe_image: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred while processing the image: {str(e)}"}), 500

################################################################################
# CAMERA FEEDS ENDPOINTS START
################################################################################

@app.route("/camera_feeds", methods=["GET"])
@token_required
def camera_feeds():
    """
    Fetches up to four camera feed URLs for the current user from the camera_feeds table.
    Renders them in camera_feeds.html, assigning the first as main_feed and others as preview_feeds.
    """
    user_id = g.current_user["user_id"]
    current_app.logger.debug(f"[camera_feeds] user_id={user_id}")

    username = g.current_user.get("username", "User")
    camera_feed_urls = []

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT feed_url
                    FROM camera_feeds
                    WHERE user_id = %s
                    ORDER BY id ASC
                    LIMIT 4
                    """,
                    (user_id,)
                )
                rows = cursor.fetchall()
                # Access feed_url by key
                camera_feed_urls = [row["feed_url"] for row in rows]
    except Exception as e:
        current_app.logger.error(f"Database error retrieving camera feeds: {e}", exc_info=True)
        return jsonify({"error": "Failed to fetch camera feeds"}), 500

    camera_count = len(camera_feed_urls)
    main_feed = camera_feed_urls[0] if camera_count > 0 else None
    preview_feeds = camera_feed_urls[1:] if camera_count > 1 else []

    return render_template(
        "camera_feeds.html",
        is_authenticated=True,
        username=username,
        main_feed=main_feed,
        preview_feeds=preview_feeds,
        camera_count=camera_count
    )


@app.route("/api/camera_list", methods=["GET"])
@token_required
def api_camera_list():
    """
    Returns up to four camera feeds for the current user as JSON.
    Example Response:
    {
      "cameras": [
        {
          "camera_id": 17,
          "url": "http://camerafeeds.vestavio.app/static/camera_feeds/video_feed_1",
          "thumbnail": null
        },
        ...
      ]
    }
    """
    user_id = g.current_user["user_id"]
    current_app.logger.debug(f"[api_camera_list] user_id={user_id}")

    base_camera_url = os.getenv("BASE_CAMERA_URL", "http://camerafeeds.vestavio.app/static/camera_feeds/")
    cameras_data = []

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, feed_url
                    FROM camera_feeds
                    WHERE user_id = %s
                    ORDER BY id ASC
                    LIMIT 4
                    """,
                    (user_id,)
                )
                rows = cursor.fetchall()
                for row in rows:
                    camera_id, raw_url = row
                    if not raw_url.lower().startswith("http"):
                        final_url = base_camera_url.rstrip("/") + "/" + raw_url.lstrip("/")
                    else:
                        final_url = raw_url
                    cameras_data.append({
                        "camera_id": camera_id,
                        "url": final_url,
                        "thumbnail": None
                    })
    except Exception as e:
        current_app.logger.error(f"[api_camera_list] Database error: {e}", exc_info=True)
        return jsonify({"error": "Failed to fetch camera feeds"}), 500

    return jsonify({"cameras": cameras_data}), 200


@app.route("/get_cameras", methods=["GET"])
@token_required
def get_cameras():
    """
    Returns JSON with the current camera count and a list of available cameras.
    Example response:
    {
      "cameraCount": 2,
      "cameras": [
        {
          "camera_id": 1,
          "name": "Main Entrance",
          "url": "http://camerafeeds.vestavio.app/static/camera_feeds/video_feed_1"
        },
        {
          "camera_id": 2,
          "name": "Back Door",
          "url": "http://camerafeeds.vestavio.app/static/camera_feeds/video_feed_2"
        }
      ]
    }
    """
    user_id = g.current_user["user_id"]
    current_app.logger.debug(f"[get_cameras] user_id={user_id}")

    base_camera_url = os.getenv("BASE_CAMERA_URL", "http://camerafeeds.vestavio.app/static/camera_feeds/")
    camera_list = []

    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT id, feed_url, name
                    FROM camera_feeds
                    WHERE user_id = %s
                    ORDER BY id ASC
                """, (user_id,))
                rows = cursor.fetchall()
                for row in rows:
                    url = row["feed_url"]
                    if not url.lower().startswith("http"):
                        url = base_camera_url.rstrip("/") + "/" + url.lstrip("/")
                    camera_list.append({
                        "camera_id": row["id"],
                        "name": row.get("name") or f"Camera {row['id']}",
                        "url": url
                    })
    except Exception as e:
        current_app.logger.error(f"[get_cameras] Database error: {e}", exc_info=True)
        return jsonify({"error": "Failed to fetch camera feeds"}), 500

    response = {
        "cameraCount": len(camera_list),
        "cameras": camera_list
    }
    return jsonify(response), 200


@app.route("/camera_config", methods=["GET"])
@token_required
def camera_config():
    """
    Returns additional camera configuration data:
    - For example: resolution, framerate, or any specialized settings.
    Example response:
      {
        "defaultResolution": "720p",
        "allowedResolutions": ["480p", "720p", "1080p"],
        "defaultFramerate": 30,
        "maxFramerate": 60
      }
    """
    user_id = g.current_user["user_id"]
    current_app.logger.debug(f"[camera_config] user_id={user_id}")

    # Hardcoded configuration data; in production, this may be stored in a config file or database.
    config_data = {
        "defaultResolution": "720p",
        "allowedResolutions": ["480p", "720p", "1080p"],
        "defaultFramerate": 30,
        "maxFramerate": 60
    }
    return jsonify(config_data), 200


@app.route("/camera_config", methods=["POST"])
@token_required
def save_camera_layout():
    """
    Persists the user's chosen camera layout to the database.
    Expects JSON like:
      {
        "layout": [
          { "camera_id": 1, "position": "top-left" },
          { "camera_id": 2, "position": "bottom-right" }
        ]
      }
    On success, returns 200 OK with a JSON message.
    """
    user_id = g.current_user["user_id"]
    data = request.get_json() or {}
    layout = data.get("layout", [])

    current_app.logger.debug(f"[save_camera_layout] user_id={user_id}, layout={layout}")

    if not isinstance(layout, list):
        return jsonify({"error": "Invalid layout format; expected a list."}), 400

    layout_json = json.dumps(layout)

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO camera_layouts (user_id, layout_json)
                    VALUES (%s, %s)
                    ON CONFLICT (user_id)
                    DO UPDATE SET layout_json = EXCLUDED.layout_json
                """, (user_id, layout_json))
            conn.commit()
        return jsonify({"message": "Camera layout saved successfully."}), 200
    except Exception as e:
        current_app.logger.error(f"[save_camera_layout] DB error: {e}", exc_info=True)
        return jsonify({"error": "Failed to save camera layout"}), 500

################################################################################
# CAMERA FEEDS ENDPOINTS END
################################################################################

# ------------------------------------------------------------------------------
# Subscription Endpoint
# ------------------------------------------------------------------------------
import stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

@app.route('/subscription', methods=['GET'])
def subscription():
    return render_template('subscription.html')

# ------------------------------------------------------------------------------
# Subscribe Endpoint
# ------------------------------------------------------------------------------
import stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

@app.route('/subscribe', methods=['POST'])
@token_required
def subscribe():
    user_id = g.current_user['user_id']

    # 1) Fetch the user and see if we already have a stripe_customer_id
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT stripe_customer_id FROM users WHERE id = %s", (user_id,))
            row = cursor.fetchone()
    
    stripe_customer_id = row['stripe_customer_id']
    if not stripe_customer_id:
        # 2) Create the Stripe customer
        customer = stripe.Customer.create(email="some@email.com")
        stripe_customer_id = customer.id
        # 3) Store it in DB
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE users
                    SET stripe_customer_id = %s
                    WHERE id = %s
                """, (stripe_customer_id, user_id))
            conn.commit()

    # 4) Now either create a Checkout Session, or do your subscription logic
    # ...
    return jsonify({"message": "Subscription started!"})

# ------------------------------------------------------------------------------
# Customer Portal Endpoint (No Login Required, Single Placeholder Customer)
# ------------------------------------------------------------------------------
import stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

@app.route('/customer_portal', methods=['GET'])
@token_required  # or however you enforce your login
def customer_portal():
    """
    Dynamically retrieve the user's Stripe customer ID from your DB
    and generate a Billing Portal session for them.
    """

    # 1) Identify the current user (already done by token_required).
    user_id = g.current_user.get('user_id')
    if not user_id:
        return jsonify({"error": "No user ID found in token"}), 400

    try:
        # 2) Query your 'users' table to get this user's stripe_customer_id
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT stripe_customer_id
                    FROM users
                    WHERE id = %s
                    LIMIT 1
                """, (user_id,))
                user_row = cursor.fetchone()

        if not user_row or not user_row.get('stripe_customer_id'):
            # If they don’t have a Stripe customer yet, either create one here
            # or show an error that they must sign up first, etc.
            return jsonify({"error": "No stripe_customer_id found for this user"}), 400

        stripe_customer_id = user_row['stripe_customer_id']

        # 3) Create the billing portal session for that specific customer
        portal_session = stripe.billing_portal.Session.create(
            customer=stripe_customer_id,
            return_url=url_for('dashboard', _external=True)
        )

        # 4) Redirect the user to the portal_session.url
        return redirect(portal_session.url)

    except Exception as e:
        app.logger.error(f"Error creating customer portal session: {e}", exc_info=True)
        return jsonify({"error": "Unable to create customer portal session."}), 500
  
# ------------------------------------------------------------------------------
# Logout Endpoint
# ------------------------------------------------------------------------------
@app.route('/logout', methods=['POST'])
def logout():
    # Clear session data if any
    session.clear()
    flash("Logged out successfully.", "success")
    response = redirect(url_for('login'))
    response.delete_cookie('token')
    return response

# ------------------------------------------------------------------------------
# Error Handling and Teardown Endpoint
# ------------------------------------------------------------------------------
@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({"error": "An unexpected error occurred."}), 500

SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.teardown_appcontext
def cleanup_models(exception=None):
    if hasattr(g, '__dict__'):
        for key in list(g.__dict__.keys()):
            if key.endswith('_model'):
                app.logger.info(f"Cleaning up {key}.")
                delattr(g, key)
    else:
        app.logger.warning("No attributes in g to clean up.")

################################################################################
# MOBILE APP ENDPOINTS
################################################################################
# ------------------------------------------------------------------------------
# Mobile Initialization Endpoint
# ------------------------------------------------------------------------------
@app.route('/mobile_init', methods=['GET'])
@token_required
def mobile_init():
    """
    Aggregates mobile-specific data in a single call.
    Returns current user info, camera feed URLs, and the user's chat logs.
    """
    user_id = g.current_user.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID not found in token"}), 400

    # 1) Retrieve user info
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT id, username, email FROM users WHERE id = %s", (user_id,))
                user_info = cursor.fetchone()
    except Exception as e:
        app.logger.error(f"Error fetching user info: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve user information"}), 500

    # 2) Define camera feeds (static for now, as in /camera_feeds)
    camera_feeds = [
        {
            "camera_id": 1,
            "name": "Main Entrance",
            "url": "http://camerafeeds.vestavio.app/video_feed_1"
        },
        {
            "camera_id": 2,
            "name": "Back Door",
            "url": "http://camerafeeds.vestavio.app/video_feed_2"
        }
    ]

    # 3) Retrieve chat logs for the current user
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT speaker, message, created_at
                    FROM chat_logs
                    WHERE user_id = %s
                    ORDER BY created_at ASC
                """, (user_id,))
                chat_logs = cursor.fetchall()
    except Exception as e:
        app.logger.error(f"Error fetching chat logs: {e}", exc_info=True)
        chat_logs = []

    return jsonify({
        "user": user_info,
        "cameraFeeds": camera_feeds,
        "chatLogs": chat_logs
    }), 200

# ------------------------------------------------------------------------------
# Push Notification Registration Endpoint
# ------------------------------------------------------------------------------
@app.route('/push_notification', methods=['POST'])
@token_required
def push_notification():
    """
    Registers or updates the device token for push notifications.
    Expects JSON:
    {
      "deviceToken": "the_device_token_string",
      "platform": "android"   // optional, default is 'unknown'
    }
    Stores the token in the device_tokens table.
    """
    data = request.get_json() or {}
    device_token = data.get("deviceToken")
    platform = data.get("platform", "unknown")

    if not device_token:
        return jsonify({"error": "Missing device token"}), 400

    user_id = g.current_user.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID not found in token"}), 400

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Insert or update the device token for this user.
                # Assumes a table "device_tokens" with columns:
                # id, user_id, device_token, platform, created_at
                cursor.execute("""
                    INSERT INTO device_tokens (user_id, device_token, platform, created_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (user_id)
                    DO UPDATE SET device_token = EXCLUDED.device_token,
                                  platform = EXCLUDED.platform,
                                  created_at = NOW()
                """, (user_id, device_token, platform))
            conn.commit()
        return jsonify({"message": "Device token registered successfully."}), 200
    except Exception as e:
        app.logger.error(f"Error storing device token: {e}", exc_info=True)
        return jsonify({"error": "Failed to store device token."}), 500

################################################################################
# TESTING ENDPOINTS
################################################################################
# ------------------------------------------------------------------------------
# TEMPORARY TEST PROMPTS Endpoint
# ------------------------------------------------------------------------------
@app.route("/test_prompts", methods=["GET"])
@token_required
def test_prompts():
    """
    A temporary endpoint that runs a small set of predefined test prompts
    against our NaturalLanguageModule for demonstration or debugging.
    Returns a JSON array of {id, input, output}.
    """
    from B_Module_Files.natural_language_module import NaturalLanguageModule

    # Define some sample prompts (no placeholders)
    test_set = [
        {"id": 1, "input": "Hello, how are you today?"},
        {"id": 2, "input": "#start simulation with parameters alpha=0.1 beta=0.9"},
        {"id": 3, "input": "Could you summarize the concept of sensor fusion?"},
        {"id": 4, "input": "What's the status of my system health right now?"},
        {"id": 5, "input": "#run predictive maintenance check"}
    ]

    # Initialize our language module (using the same model as the rest of the app)
    nlm = NaturalLanguageModule(model_name="gpt-4o-mini", logger=app.logger)

    results = []
    for test_item in test_set:
        prompt_text = test_item["input"]
        # Generate a quick reply (no special system/developer instructions here)
        try:
            output_text = nlm.generate_reply(prompt_text)
        except Exception as e:
            app.logger.error(f"Error in test_prompts for prompt '{prompt_text}': {e}", exc_info=True)
            output_text = f"Error: {str(e)}"

        results.append({
            "id": test_item["id"],
            "input": prompt_text,
            "output": output_text
        })

    return jsonify(results), 200

# ------------------------------------------------------------------------------
# Updated /run_genesis_simulation Endpoint (testing)(for future use in Tier 2 deployment, see file in A_Core_App_Files)
# ------------------------------------------------------------------------------
'''@app.route('/run_genesis_simulation', methods=['POST'])
@token_required
def run_genesis_simulation_endpoint():
    return jsonify({
        "status": "error",
        "message": "Simulation functionality is not available in this release."
    }), 501
'''

# ------------------------------------------------------------------------------
# Trigger Genesis Sim Endpoint (testing)(for future use in Tier 2 deployment, see file in A_Core_App_Files)
# ------------------------------------------------------------------------------
'''@app.route('/trigger_genesis_sim', methods=['POST'])
@token_required
def trigger_genesis_sim():
    return jsonify({
        "status": "error",
        "message": "Simulation functionality is not available in this release."
    }), 501
'''

# ------------------------------------------------------------------------------
# Testing Endpoint
# ------------------------------------------------------------------------------
@app.route('/example_endpoint', methods=['POST'])
def example():
    data = request.get_json() or {}
    # Just return a dummy response for testing
    return jsonify({"message": "Example endpoint received data", "data": data}), 200

print("Gunicorn or systemd will run this in production; no Waitress needed.")

if __name__ == '__main__':
    # In dev mode, you might still want to run Flask directly:
    app.logger.info("Running Flask dev server (not production).")
    app.run(host='0.0.0.0', port=5003, debug=True)