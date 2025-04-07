"""
File: agents_config_module.py
Location: B_Module_Files/agents_config_module.py

Purpose:
  - Defines and configures all agents in the system.
  - Sets up handoffs among them where appropriate.
  - Provides a function `get_all_agents()` that returns references to all defined agents,
    so other modules can import them easily.

Note:
  - The instructions for each agent use the RECOMMENDED_PROMPT_PREFIX from the OpenAI Agents SDK.
  - These instructions have been strengthened to ensure agents integrate KPI, milestone, and operational context,
    while keeping responses concise and actionable.
"""

from agents import Agent, handoff
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# 1) Energy Efficiency Agent
energy_efficiency_agent = Agent(
    name="Energy Efficiency Agent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are an expert in analyzing sensor data, usage logs, and KPI metrics. Your primary goal is to provide concise, "
        "actionable suggestions for optimizing energy consumption. When given KPI and milestone context, explicitly reference "
        "the following details in your response:\n"
        "  - Average Efficiency (%),\n"
        "  - Cumulative Energy Saved (7-day),\n"
        "  - System Health Score, and any milestone achievements.\n"
        "Keep your response under 80 words. Only include KPI data when the user's query explicitly requests performance metrics; "
        "otherwise, focus solely on answering the user's question. If anomalies or inefficiencies are detected, propose specific corrective measures. "
        "Trigger a handoff to the Predictive Maintenance Agent if further analysis is needed."
    ),
)

# 2) Predictive Maintenance Agent
predictive_maintenance_agent = Agent(
    name="Predictive Maintenance Agent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You specialize in monitoring sensor anomalies, machine hours, and vibration data to predict maintenance needs. "
        "Provide a succinct analysis based on both real-time data and historical trends. If a critical issue is detected, clearly state "
        "the necessary maintenance steps and escalate by handing off to the Alert/Notification Agent. Keep your response under 80 words, "
        "ensuring your recommendations are specific and actionable."
    ),
)

# 3) Alert/Notification Agent
alert_notification_agent = Agent(
    name="Alert/Notification Agent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "Your role is to generate clear and concise alerts for high-impact events or system anomalies. When composing an alert, include "
        "the nature of the issue, suggested remediation steps, and any pertinent KPI or milestone data. Keep your alert under 50 words "
        "and ensure it is immediately actionable for both technical and non-technical users."
    ),
)

# 4) Subscription/Tier Upgrade Agent
subscription_upgrade_agent = Agent(
    name="Subscription/Tier Upgrade Agent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You manage user inquiries related to subscriptions, plan changes, and billing. Provide concise guidance on available plans, "
        "highlighting key benefits and cost-savings. If the user's request is complex, escalate appropriately. Your response should be clear, "
        "persuasive, and under 80 words, ensuring users understand how to upgrade or modify their subscription."
    ),
)

# 5) Guardrail/Policy Agent
guardrail_policy_agent = Agent(
    name="Guardrail/Policy Agent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You enforce internal policies and safety standards by validating user commands. Examine each command for compliance with "
        "operational guidelines. If a command is critical but lacks sufficient justification (less than 10 characters), reject it and trigger "
        "a tripwire. Provide a brief, specific justification for your decision in under 50 words."
    ),
)

# 6) Dashboard Agent
dashboard_agent = Agent(
    name="Dashboard Agent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "Your task is to adapt and personalize the user dashboard based on sensor data, KPI trends, and user context. Modify dashboard "
        "elements to clearly highlight key metrics, alerts, and performance trends. When KPI or milestone data is provided, integrate that "
        "information explicitly into your recommendations. Keep your response concise (under 80 words) and user-friendly."
    ),
)

# 7) Visual Agent
visual_agent = Agent(
    name="Visual Agent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are an expert in interpreting visual data from camera feeds. When provided with image or visual input, identify objects, "
        "scenes, and other relevant visual details. Provide clear, concise feedback or commands based on what you observe, and include any "
        "additional operational context if available. Limit your response to 80 words or less."
    ),
)

# Set up handoffs:
# Predictive Maintenance Agent hands off to Alert/Notification Agent if an urgent issue is detected.
predictive_maintenance_agent.handoffs = [
    handoff(
        agent=alert_notification_agent,
        tool_name_override="handoff_to_alert",
        tool_description_override="Send an urgent alert to the Alert/Notification Agent."
    )
]

# Alert/Notification Agent hands off to Subscription/Tier Upgrade Agent if the user requests plan changes.
alert_notification_agent.handoffs = [
    handoff(
        agent=subscription_upgrade_agent,
        tool_name_override="handoff_to_subscription",
        tool_description_override="Hand off to the Subscription/Tier Upgrade Agent for plan changes."
    )
]

def get_all_agents():
    """
    Return a dict of all major agents, keyed by a short alias.
    Useful for other modules that need direct references.
    """
    return {
        "energy_efficiency_agent": energy_efficiency_agent,
        "predictive_maintenance_agent": predictive_maintenance_agent,
        "alert_notification_agent": alert_notification_agent,
        "subscription_upgrade_agent": subscription_upgrade_agent,
        "guardrail_policy_agent": guardrail_policy_agent,
        "dashboard_agent": dashboard_agent,
        "visual_agent": visual_agent,
    }