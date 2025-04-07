# ------------------------------------------------------------------------------
# File: function_schemas_module.py
# Purpose:
#   - Defines a list of function schemas (tools) in the format required by
#     openai.chat.completions.create(..., functions=[...]).
#   - Each schema must include "type": "function", "name", "description",
#     and "parameters".
#
#   This script now includes function schemas relevant to both:
#     - Basic examples (weather, email, knowledge base)
#     - New agent-driven actions (dashboard updates, alerts, maintenance, subscriptions)
#
#   The model can call these functions by name, passing JSON arguments
#   that match the 'parameters' JSON schema.
#
#   Additionally, we provide a 'handle_function_call()' function at the end,
#   which acts as a centralized dispatcher for these function calls.
# ------------------------------------------------------------------------------

import logging

def get_all_tools():
    """
    Returns a list of dictionaries, each describing a function:
      - "type": "function"
      - "name": unique function name
      - "description": short explanation of what it does
      - "parameters": JSON schema describing the arguments

    This format is required by openai.chat.completions.create(..., functions=[...])
    so that the model can call these functions with valid arguments.

    Note: The model (or agent) can choose from any of these function calls
    when generating a response, if it deems them relevant.
    """
    return [
        # -----------------------
        # 1) Example Tools
        # -----------------------
        {
            "type": "function",
            "name": "get_weather",
            "description": "Retrieve the current temperature for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country, e.g. 'Bogotá, Colombia' or 'Paris, France'"
                    }
                },
                "required": ["location"],
                "additionalProperties": False
            }
        },
        {
            "type": "function",
            "name": "send_email",
            "description": "Send an email using our SendGrid/Twilio integration to a specified recipient with a subject and message. This function is used to dispatch alert notifications and other communications in accordance with user alert preferences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "The recipient email address."
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line."
                    },
                    "body": {
                        "type": "string",
                        "description": "Body of the email message."
                    }
                },
                "required": ["to", "subject", "body"],
                "additionalProperties": False
            }
        },
        {
            "type": "function",
            "name": "search_knowledge_base",
            "description": "Query a knowledge base to retrieve relevant info on a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user question or search query."
                    },
                    "options": {
                        "type": "object",
                        "properties": {
                            "num_results": {
                                "type": "number",
                                "description": "Number of top results to return."
                            },
                            "domain_filter": {
                                "type": ["string", "null"],
                                "description": "Optional domain to narrow the search (e.g. 'finance', 'medical'). Pass null if not needed."
                            },
                            "sort_by": {
                                "type": ["string", "null"],
                                "enum": ["relevance", "date", "popularity", "alphabetical"],
                                "description": "How to sort results. Pass null if not needed."
                            }
                        },
                        "required": ["num_results", "domain_filter", "sort_by"],
                        "additionalProperties": False
                    }
                },
                "required": ["query", "options"],
                "additionalProperties": False
            }
        },

        # -----------------------
        # 2) Tools for New Agents
        # -----------------------

        # (A) Dashboard Agent => update dashboard layout or highlight features
        {
            "type": "function",
            "name": "update_dashboard_layout",
            "description": "Allows the Dashboard Agent to modify or highlight parts of the user dashboard interface.",
            "parameters": {
                "type": "object",
                "properties": {
                    "layout_type": {
                        "type": "string",
                        "description": "Type of layout to apply (e.g. 'split', 'single-column', 'custom')."
                    },
                    "highlight_feature": {
                        "type": "string",
                        "description": "Optional feature name to highlight (e.g. 'kpi_charts', 'camera_feed')."
                    }
                },
                "required": ["layout_type"],
                "additionalProperties": False
            }
        },

        # (B) Alert/Notification Agent => issue alerts
        {
            "type": "function",
            "name": "issue_alert_notification",
            "description": "Used by the Alert/Notification Agent to send an alert to a user or system channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "alert_level": {
                        "type": "string",
                        "description": "Severity of the alert (e.g. 'info', 'warning', 'critical')."
                    },
                    "message": {
                        "type": "string",
                        "description": "Main alert message or content."
                    }
                },
                "required": ["alert_level", "message"],
                "additionalProperties": False
            }
        },

        # (C) Predictive Maintenance Agent => schedule maintenance
        {
            "type": "function",
            "name": "schedule_maintenance",
            "description": "Used by the Predictive Maintenance Agent to schedule or request maintenance for equipment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "equipment_id": {
                        "type": "string",
                        "description": "Unique ID or name of the equipment needing maintenance."
                    },
                    "urgency": {
                        "type": "string",
                        "description": "Urgency level (e.g. 'low', 'medium', 'high')."
                    },
                    "proposed_date": {
                        "type": "string",
                        "description": "Proposed date/time for maintenance (e.g. '2025-03-20 10:00')."
                    }
                },
                "required": ["equipment_id", "urgency"],
                "additionalProperties": False
            }
        },

        # (D) Subscription/Tier Upgrade Agent => change user subscription
        {
            "type": "function",
            "name": "upgrade_subscription_plan",
            "description": "Used by the Subscription/Tier Upgrade Agent to modify a user's subscription plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "Unique user ID or email to identify the user."
                    },
                    "new_plan": {
                        "type": "string",
                        "description": "The new plan or tier (e.g. 'premium', 'enterprise')."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional note or reason for the upgrade."
                    }
                },
                "required": ["user_id", "new_plan"],
                "additionalProperties": False
            }
        }
    ]


def handle_function_call(func_name: str, func_args: dict) -> str:
    """
    A centralized dispatcher for function calls invoked by the LLM or agent.
    This ensures the model can call 'get_weather', 'send_email', 'search_knowledge_base',
    or any new agent-driven actions, and we handle the logic here.

    :param func_name: Name of the function the LLM wants to call.
    :param func_args: Dictionary of arguments for that function.
    :return: String result of calling the function logic.
    """
    logger = logging.getLogger("FunctionCallHandler")

    try:
        # ------------------------------------------------------------------
        # 1) Example Tools
        # ------------------------------------------------------------------
        if func_name == "get_weather":
            location = func_args.get("location", "Unknown")
            # In production, integrate with a weather API
            return f"The current weather in {location} is ~72°F and sunny."

        elif func_name == "send_email":
            to = func_args.get("to", "")
            subject = func_args.get("subject", "")
            body = func_args.get("body", "")
            # In production, this would call our SendGrid/Twilio integration
            logger.info(f"Sending email to {to} with subject '{subject}'. Body: {body[:60]}...")
            return f"Email to {to} with subject '{subject}' was sent successfully."

        elif func_name == "search_knowledge_base":
            query = func_args.get("query", "")
            options = func_args.get("options", {})
            num_results = options.get("num_results", 3)
            domain_filter = options.get("domain_filter", None)
            sort_by = options.get("sort_by", None)
            # In production, integrate with your vector database or RAG pipeline
            logger.info(f"KB Search for '{query}' [domain={domain_filter}, sort_by={sort_by}, limit={num_results}]")
            return f"Found {num_results} results for '{query}' in knowledge base. Domain={domain_filter}, sorted by={sort_by}."

        # ------------------------------------------------------------------
        # 2) Tools for New Agents
        # ------------------------------------------------------------------
        elif func_name == "update_dashboard_layout":
            layout_type = func_args.get("layout_type", "default")
            highlight_feature = func_args.get("highlight_feature", None)
            msg = f"Dashboard layout updated to '{layout_type}' layout."
            if highlight_feature:
                msg += f" Highlighting feature: {highlight_feature}."
            logger.info(f"[DashboardAgent] {msg}")
            return msg

        elif func_name == "issue_alert_notification":
            alert_level = func_args.get("alert_level", "info")
            message = func_args.get("message", "")
            logger.info(f"[AlertAgent] Alert level={alert_level} message={message}")
            return f"Issued {alert_level.upper()} alert: {message}"

        elif func_name == "schedule_maintenance":
            equipment_id = func_args.get("equipment_id", "")
            urgency = func_args.get("urgency", "low")
            proposed_date = func_args.get("proposed_date", "not specified")
            logger.info(f"[MaintenanceAgent] Scheduling maintenance for {equipment_id} (urgency={urgency}) on {proposed_date}")
            return f"Scheduled maintenance for equipment '{equipment_id}' with urgency '{urgency}' on {proposed_date}."

        elif func_name == "upgrade_subscription_plan":
            user_id = func_args.get("user_id", "")
            new_plan = func_args.get("new_plan", "")
            reason = func_args.get("reason", "")
            logger.info(f"[SubscriptionAgent] Upgrading user {user_id} to plan '{new_plan}'. Reason: {reason}")
            return f"User '{user_id}' upgraded to plan '{new_plan}'. Reason: {reason or 'No specific reason provided.'}"

        # ------------------------------------------------------------------
        # Unknown function
        # ------------------------------------------------------------------
        else:
            logger.warning(f"Unknown function call: '{func_name}'")
            return f"[Error] Unknown function call: '{func_name}'"

    except Exception as e:
        logger.error(f"Exception in handle_function_call('{func_name}'): {e}", exc_info=True)
        return f"[Error] Exception in '{func_name}' function: {str(e)}"