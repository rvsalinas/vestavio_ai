Energy Optimization Project (Tier 1 Demo)
[NOTE: The project folder should be renamed to energy_optimization_project—remove the _github from the name.]

Welcome to the Tier 1 demonstration of our Energy Optimization Project! This repository showcases how Genesis-based GPU simulations can integrate with ML models (anomaly detection, RL, energy efficiency, etc.) via a Flask backend—all in a headless environment. Gensis files have been purposely removed, but can be retrieved from github, search: genesis physics engine. 

Project Structure
bash
Copy
energy_optimization_project/ 
├─ A_Core_App_Files/ 
│    └─ app.py           # Flask backend
├─ B_Module_Files/       # Supporting modules (preprocessing, registry, etc.)
├─ C_Model_Files/        # Trained models (joblib, Keras, RL checkpoints)
├─ config/               # model_registry.json, my_benchmark_results.json, performance_history.json 
├─ templates/            # HTML templates for the UI
├─ READMEmain.md        # This Tier 1 README
├─ gunicorn.service     # Gunicorn service file (top-level)
Prerequisites
Anaconda Environment:
Use the fresh_env environment containing all required packages: TensorFlow, scikit-learn, stable-baselines3, psycopg2, etc.

Genesis v0.2.1:
Ensure you have Genesis installed (or available as an external dependency) with headless mode enabled for GPU simulations.

PostgreSQL:
Used for sensor_data storage and user management.

Flask/Waitress:
For serving the backend on port 5002.

1. Environment Setup
SSH into your EC2 or local machine.

Activate your environment:

bash
Copy
conda activate fresh_env
Configure Environment Variables:
Create or update your .env file (which should not be open-sourced) with DB credentials, MODEL_DIR, log paths, etc. Example:

bash
Copy
DB_USER=example
DB_PASSWORD=example
DB_NAME=example
DB_HOST=example
DB_PORT=example
MODEL_DIR=/path/to/models
LOG_DIR=/path/to/logs
Launch the Flask App:

bash
Copy
cd A_Core_App_Files
python app.py
This starts the backend on http://0.0.0.0:5002. You should see logs indicating that models have loaded and the database connection is established.

Obtain a Token:

bash
Copy
curl -X POST http://<YOUR_SERVER_IP>:5002/login \
  -H "Content-Type: application/json" \
  -d '{"username":"example","password":"example"}'
Copy the token from the JSON response. This token is required for endpoints protected by @token_required.

Run a Demo Script (Tier 1):
Inside the G_Genesis_Files/experiments/ folder, you'll find:

demo_9dof_scenario.py (short 50-step demo)

test_final_integration.py (200-step integration test)

test_stress_scenario.py (1,000-step multi-robot stress test)

For example, to run the 50-step demo:

bash
Copy
cd G_Genesis_Files/experiments
xvfb-run -a -s "-screen 0 1024x768x24 +extension GLX +render -noreset" \
python demo_9dof_scenario.py
What Happens:

Genesis runs ~50 steps in a GPU-based headless environment.

At each step, it collects DOF/velocity data and posts it to /send_sensor_data.

Flask logs will display anomaly detection, RL actions, or energy predictions.

Checking Logs & Dashboard:

Logs:
Check F_Log_Files/flask.log for runtime logs (these files are not open-sourced).

Dashboard:
Open http://<YOUR_SERVER_IP>:5002/ to view a minimal “System Health” snapshot. (If you have additional frontends like dash_app.py, you can run those as well.)

Next Steps
Multi-Robot Stress:
Try test_stress_scenario.py for 1,000 steps with multiple fallback scenarios.

Integration Testing:
Run test_final_integration.py (200-step run) to verify anomaly detection and RL integration.

Docstrings & Refinement:
Review the provided docstrings in major modules to understand function purposes. Explore Tier 1 to Tier 2 transitions for advanced sensor data or real hardware integration.

Contributing
For minor changes, PRs are welcome. Please update .env.example if you add new environment variables, and maintain consistency in docstrings and coding style.

License
Vestavio Proprietary Commercial License

This Software and its associated documentation ("Software") is confidential and proprietary to Vestavio ("Company") and is licensed under the terms of the Proprietary Commercial License ("License"). By using the Software, you agree to the terms of this License. If you do not agree, do not install or use the Software.

Grant of License:
Vestavio grants a limited, non-exclusive, non-transferable license to use the Software for internal business purposes only.

You may not distribute, modify, reverse-engineer, or create derivative works unless explicitly authorized by Vestavio.

Restrictions:

Do not remove or alter any proprietary notices.

Do not use the Software for any unauthorized purpose.

Do not disclose any portion of the Software to third parties without prior written consent.

Intellectual Property Rights:
All intellectual property rights in the Software remain with Vestavio.

Warranty Disclaimer:
The Software is provided "as is" without any warranty, either express or implied.

Limitation of Liability:
Vestavio is not liable for any indirect, incidental, or consequential damages arising from the use of the Software.

Termination:
This License may be terminated immediately for any breach. Upon termination, cease all use of the Software.

Governing Law:
This License is governed by the laws of Texas, United States.

Contact Information:
For any questions, please contact:
Vestavio
Email: rsalinas@vestavio.com
Address: 4017 Prescott Ave Apt A, Dallas, Texas 75219

© 2024 Vestavio. All Rights Reserved.

Enjoy your Tier 1 demonstration of GPU-based Genesis simulations integrated with multiple ML models—headless yet fully functional!