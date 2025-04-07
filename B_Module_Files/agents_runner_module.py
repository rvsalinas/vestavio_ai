"""
File: agents_runner_module.py
Location: B_Module_Files/agents_runner_module.py

Purpose:
  - Orchestrates agent usage at a higher level.
  - Demonstrates a sample function process_user_request() that selects an agent
    (defaulting to "energy_efficiency_agent"), applies global guardrails, and returns the final output.
  - Intended for integration with Flask endpoints or other production code.
  - Note: The instructions for energy_efficiency_agent (and visual_agent, if used) in agents_config_module.py
    have been updated to incorporate KPI, milestone, and operational context as required.
    To use the visual agent, pass agent_name="visual_agent" to run_sync_request().
"""

import asyncio
import logging

from agents import Runner, GuardrailFunctionOutput
from B_Module_Files.agents_config_module import get_all_agents
from B_Module_Files.agent_guardrails_module import AgentGuardrails, apply_guardrails

# Retrieve the dictionary of all agents from the configuration
all_agents = get_all_agents()

# Create an instance of AgentGuardrails for global guardrail checks
global_guardrails = AgentGuardrails(name="Global Guardrails")

def get_agent_by_name(agent_name: str):
    """
    Retrieve an agent by its name from our configuration.
    
    :param agent_name: The name of the agent.
    :return: The agent object if found, otherwise None.
    """
    return all_agents.get(agent_name)

async def process_user_request(user_text: str, agent_name: str = "energy_efficiency_agent") -> str:
    """
    Asynchronously processes a user request by applying global guardrails and then running the specified agent.
    The agent's instructions are designed to integrate KPI and milestone context when provided.
    
    :param user_text: The raw input text from the user.
    :param agent_name: The name of the agent to run (defaults to "energy_efficiency_agent").
                       To use the visual agent, pass agent_name="visual_agent".
    :return: The final output from the agent as a string.
    """
    agent = get_agent_by_name(agent_name)
    if not agent:
        return f"[Error] No agent found with name: {agent_name}"

    # Apply global guardrails on the input text.
    try:
        guardrail_result: GuardrailFunctionOutput = await apply_guardrails(user_text, global_guardrails)
        if guardrail_result.tripwire_triggered:
            return "[Error] Guardrail check failed: insufficient justification for a critical command."
    except Exception as gr_e:
        logging.error(f"Guardrail application error: {gr_e}", exc_info=True)
        return "[Error] Guardrail processing encountered an exception."

    # Run the agent with the provided user text.
    try:
        result = await Runner.run(agent, user_text)
        output = result.final_output
        if not isinstance(output, str):
            output = str(output)
        return output
    except Exception as e:
        logging.error(f"Error in process_user_request with agent '{agent_name}': {e}", exc_info=True)
        return "[Error] Agent processing encountered an exception."

def run_sync_request(user_text: str, agent_name: str = "energy_efficiency_agent") -> str:
    """
    Synchronous wrapper around process_user_request for convenience.
    
    :param user_text: The raw input text from the user.
    :param agent_name: The name of the agent to run (defaults to "energy_efficiency_agent").
                       To use the visual agent, pass agent_name="visual_agent".
    :return: The final output from the agent.
    """
    return asyncio.run(process_user_request(user_text, agent_name))

# Example usage for quick testing:
if __name__ == "__main__":
    test_input = "Hello, I'd like some suggestions on reducing energy consumption."
    print("User Input:", test_input)
    response = run_sync_request(test_input)
    print("Energy Efficiency Agent Output:", response)
    
    # Example test for the visual agent (if defined in your configuration)
    visual_test_input = "What objects do you see in the current camera feed?"
    visual_response = run_sync_request(visual_test_input, agent_name="visual_agent")
    print("Visual Agent Output:", visual_response)