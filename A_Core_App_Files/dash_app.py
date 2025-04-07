import dash
from dash import dcc, html, dash_table, Input, Output
import requests
import pandas as pd
import datetime

app = dash.Dash(__name__)
server = app.server  # For production deployment with e.g. gunicorn

# =============================================================================
# Layout
# =============================================================================

app.layout = html.Div([
    # 1) Hidden interval component for auto-refresh every 60s
    dcc.Interval(
        id="auto-refresh",
        interval=60*1000,  # 60 seconds
        n_intervals=0
    ),

    # 2) Milestone Banner (positive reinforcement)
    #    We fill this via callback if server returns a "milestone".
    html.Div(
        id="milestone-banner",
        style={
            "margin": "10px 0",
            "padding": "10px",
            "textAlign": "center",
            "fontWeight": "bold",
            "fontSize": "1.1rem",
            "border": "2px dashed #ccc",
            "display": "none",  # hidden by default
        }
    ),

    # 3) KPI Row for immediate value
    html.Div(
        id="kpi-container",
        style={
            "display": "flex",
            "justifyContent": "space-around",
            "marginBottom": "20px"
        }
    ),

    # 4) System Health Banner
    html.Div(
        id="system-health-banner",
        style={
            "width": "100%",
            "padding": "20px",
            "color": "white",
            "textAlign": "center",
            "fontWeight": "bold",
            "fontSize": "1.2rem",
            "marginBottom": "20px"
        }
    ),

    # 5) Last Updated & Refresh Row
    html.Div(
        style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "marginBottom": "10px"
        },
        children=[
            html.Div(
                "Auto-updating every 60s",
                style={"fontStyle": "italic", "color": "#666"}
            ),
            html.Button(
                "Refresh Data Manually",
                id="refresh-button",
                n_clicks=0,
                style={"fontSize": "1rem"}
            ),
            html.Div(
                "Last updated: ",
                style={"marginRight": "5px"}
            ),
            html.Div(
                id="last-updated-div",
                style={"fontWeight": "bold"}
            )
        ]
    ),

    # 6) Sensor Table Container
    html.Div(id="sensor-table-container", style={"marginTop": "15px"}),

    # 7) Autonomous Optimization Message
    html.Div(
        "System continuously optimizes in the background—no manual intervention required.",
        id="optimization-message",
        style={
            "marginTop": "30px",
            "fontStyle": "italic",
            "textAlign": "center",
            "fontSize": "1rem",
            "color": "#555"
        }
    )
], style={"maxWidth": "900px", "margin": "0 auto", "padding": "20px"})

# =============================================================================
# Callback
# =============================================================================
@app.callback(
    # 1) Banner Text, Banner Style
    Output("system-health-banner", "children"),
    Output("system-health-banner", "style"),
    # 2) Sensor Table
    Output("sensor-table-container", "children"),
    # 3) KPI Container
    Output("kpi-container", "children"),
    # 4) Milestone Banner
    Output("milestone-banner", "children"),
    Output("milestone-banner", "style"),
    # 5) Last Updated
    Output("last-updated-div", "children"),

    # Inputs: Refresh button + auto-refresh Interval
    Input("refresh-button", "n_clicks"),
    Input("auto-refresh", "n_intervals")
)
def update_sensor_data(n_clicks, n_intervals):
    """
    Fetch data from /snapshot every time user clicks "Refresh Data" or
    auto-refresh triggers (once per 60s).

    Expects JSON from /snapshot like:
    {
      "sensors": [...],
      "summary_metrics": {
         "avg_energy_efficiency": 85.5,
         "total_energy_saved": 120,
         "system_health_score": 45.5
      },
      "milestone": "Record efficiency reached!"
    }
    """
    try:
        # Updated to use port 5003
        response = requests.get("http://127.0.0.1:5003/snapshot", timeout=5)
        if response.status_code != 200:
            # HTTP error
            return (
                "Server Error",  # banner text
                {"backgroundColor": "gray"},
                html.Div(f"Error fetching data: {response.status_code}", style={"color": "red"}),
                _build_kpi_div(None),
                "",  # milestone banner text
                {"display": "none"},
                _get_time_str()
            )

        data = response.json()
        sensor_list = data.get("sensors", [])
        summary_metrics = data.get("summary_metrics", {})
        milestone_text = data.get("milestone", "")

        if not sensor_list:
            return (
                "No sensor data returned.",
                {"backgroundColor": "gray"},
                html.Div("No sensor data available.", style={"color": "red"}),
                _build_kpi_div(summary_metrics),
                milestone_text,
                {"display": "none"},
                _get_time_str()
            )

        # Build the sensor table
        df = pd.DataFrame(sensor_list)
        if "sensor_name" not in df.columns or "sensor_output" not in df.columns:
            # Unexpected format
            return (
                "Unexpected Data Format",
                {"backgroundColor": "gray"},
                html.Div("Missing columns in sensor data", style={"color": "red"}),
                _build_kpi_div(summary_metrics),
                milestone_text,
                {"display": "none"},
                _get_time_str()
            )

        # Determine health
        statuses = df.get("status", pd.Series(["Operational"]*len(df))).fillna("Operational")
        lower_statuses = statuses.str.lower().values
        any_issue = any(s not in ["operational"] for s in lower_statuses)

        if any_issue:
            banner_text = "ISSUE DETECTED: System requires attention."
            banner_style = {"backgroundColor": "red", "width": "100%", "padding": "20px", "color": "white",
                            "textAlign": "center", "fontWeight": "bold", "fontSize": "1.2rem",
                            "marginBottom": "20px"}
        else:
            banner_text = "SYSTEM FUNCTIONING PROPERLY"
            banner_style = {"backgroundColor": "green", "width": "100%", "padding": "20px", "color": "white",
                            "textAlign": "center", "fontWeight": "bold", "fontSize": "1.2rem",
                            "marginBottom": "20px"}

        # Build table
        def status_to_color_icon(st):
            st_lower = str(st).lower()
            if st_lower == "operational":
                return "🔵"
            else:
                return "🔴"

        df["Status Icon"] = df["status"].apply(status_to_color_icon)

        show_cols = ["Status Icon", "sensor_name", "sensor_output", "status"]
        rename_map = {
            "Status Icon": " ",
            "sensor_name": "Sensor",
            "sensor_output": "Output",
            "status": "Status"
        }
        df_display = df[show_cols].rename(columns=rename_map)

        table = dash_table.DataTable(
            data=df_display.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df_display.columns],
            page_size=15,
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#222", "color": "#fff"},
            style_data={
                "backgroundColor": "#f9f9f9",
                "color": "#111",
                "border": "1px solid #ccc"
            },
            style_cell={
                "fontFamily": "Arial",
                "fontSize": "14px",
                "padding": "8px"
            }
        )

        table_div = html.Div(table, style={"marginTop": "20px"})

        # Build KPI row
        kpi_div = _build_kpi_div(summary_metrics)

        # Build milestone banner
        if milestone_text:
            milestone_banner_text = f"🎉 {milestone_text} 🎉"
            milestone_style = {
                "margin": "10px 0",
                "padding": "10px",
                "textAlign": "center",
                "fontWeight": "bold",
                "fontSize": "1.1rem",
                "border": "2px dashed #ccc",
                "backgroundColor": "#ffe",
                "display": "block"
            }
        else:
            milestone_banner_text = ""
            milestone_style = {"display": "none"}

        return (
            banner_text,
            banner_style,
            table_div,
            kpi_div,
            milestone_banner_text,
            milestone_style,
            _get_time_str()
        )

    except Exception as e:
        return (
            "Exception encountered",
            {"backgroundColor": "gray"},
            html.Div(f"Exception: {str(e)}", style={"color": "red"}),
            _build_kpi_div(None),
            "",
            {"display": "none"},
            _get_time_str()
        )

# =============================================================================
# Helpers
# =============================================================================
def _build_kpi_div(metrics):
    """
    Builds a row of 3 KPI 'cards' for immediate value:
      - avg_energy_efficiency
      - total_energy_saved
      - system_health_score (displayed as a fraction out of 100)
    If metrics is None or missing keys, falls back to 'N/A'.
    """
    if not metrics:
        metrics = {}

    # Retrieve values (or default to "N/A")
    eff = metrics.get("avg_energy_efficiency", "N/A")
    saved = metrics.get("total_energy_saved", "N/A")
    health = metrics.get("system_health_score", "N/A")

    # Convert and format the average efficiency as a percentage string
    try:
        eff = f"{float(eff):.1f}%"
    except (TypeError, ValueError):
        eff = str(eff)

    # Convert and format the energy saved with units
    try:
        saved = f"{float(saved):.0f} units"
    except (TypeError, ValueError):
        saved = str(saved)

    # Force conversion of health score to a float and format as fraction e.g., "35.5/100"
    try:
        health_val = float(health)
        health = f"{health_val:.1f}/100"
    except (TypeError, ValueError):
        health = str(health)

    style_card = {
        "flex": "1",
        "margin": "0 10px",
        "backgroundColor": "white",
        "borderRadius": "8px",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        "padding": "15px",
        "textAlign": "center"
    }
    style_val = {"fontSize": "1.5rem", "fontWeight": "bold"}
    style_label = {"marginTop": "5px", "color": "#666"}

    return html.Div([
        html.Div([
            html.Div(eff, style=style_val),
            html.Div("Avg Efficiency", style=style_label)
        ], style=style_card),
        html.Div([
            html.Div(saved, style=style_val),
            html.Div("Energy Saved", style=style_label)
        ], style=style_card),
        html.Div([
            html.Div(health, style=style_val),
            html.Div("System Health Score", style=style_label)
        ], style=style_card)
    ], style={"display": "flex", "width": "100%"})

def _get_time_str():
    """Returns a short string for the current time, e.g. '19:44:05'."""
    now = datetime.datetime.now().strftime("%H:%M:%S")
    return f"{now}"

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)