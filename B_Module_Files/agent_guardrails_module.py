"""
File: agent_guardrails_module.py
Location: B_Module_Files/agent_guardrails_module.py

Purpose:
  - Provides guardrail logic (input validations/short-circuit checks) for agent operations.
  - Illustrates how to run a guardrail agent or custom function to ensure operations meet safety/policy guidelines.
  - Uses GuardrailFunctionOutput to indicate whether a guardrail's tripwire was triggered.
"""

import asyncio
from typing import Optional

from pydantic import BaseModel
from agents import Agent, Runner, GuardrailFunctionOutput, RunContextWrapper

# ------------------------------------------------------------------------------
# Data Model for a critical command check
# ------------------------------------------------------------------------------
class CriticalCommandCheck(BaseModel):
    """
    Represents the result of a critical command check.
    
    Attributes:
        is_critical (bool): True if the command is considered critical.
        justification (str): A brief explanation why the command is or isn't critical.
    """
    is_critical: bool
    justification: str

# ------------------------------------------------------------------------------
# Specialized Guardrail Agent
# ------------------------------------------------------------------------------
guardrail_agent = Agent(
    name="Critical Command Checker",
    instructions=(
        "You are to check if the user's input command is critical. "
        "Return a JSON object with 'is_critical' set to True if it is a critical command, "
        "and include a 'justification' field. If the command is not critical, return is_critical: False. "
        "If the command is critical but the justification is insufficient (less than 10 characters), "
        "the guardrail should trigger a tripwire."
    ),
    output_type=CriticalCommandCheck
)

# ------------------------------------------------------------------------------
# AgentGuardrails Class
# ------------------------------------------------------------------------------
class AgentGuardrails:
    """
    Encapsulates one or more guardrail checks for agent input.
    
    This class currently provides a method to verify if a command is critical and whether adequate
    justification is provided.
    """
    def __init__(self, name: str = "Global Guardrails"):
        self.name = name

    async def check_critical_command(
        self,
        input_data: str,
        context: Optional[RunContextWrapper] = None
    ) -> GuardrailFunctionOutput:
        """
        Executes the critical command check using the specialized guardrail agent.
        
        :param input_data: The user input command to validate.
        :param context: Optional RunContextWrapper containing additional context.
        :return: A GuardrailFunctionOutput with the check result and a tripwire flag.
        """
        result = await Runner.run(
            guardrail_agent,
            input_data,
            context=context.context if context else None
        )
        final_output = result.final_output_as(CriticalCommandCheck)
        
        # Trigger tripwire if the command is critical but the justification is too short.
        if final_output.is_critical and len(final_output.justification.strip()) < 10:
            return GuardrailFunctionOutput(
                output_info=final_output,
                tripwire_triggered=True
            )
        return GuardrailFunctionOutput(
            output_info=final_output,
            tripwire_triggered=False
        )

# ------------------------------------------------------------------------------
# Function to Apply Guardrails
# ------------------------------------------------------------------------------
async def apply_guardrails(
    input_data: str,
    guardrails_obj: AgentGuardrails,
    context: Optional[RunContextWrapper] = None
) -> GuardrailFunctionOutput:
    """
    Applies guardrail checks to the given input using the provided AgentGuardrails instance.
    
    :param input_data: The input command text to validate.
    :param guardrails_obj: An instance of AgentGuardrails containing guardrail check methods.
    :param context: Optional RunContextWrapper with additional context.
    :return: A GuardrailFunctionOutput where 'tripwire_triggered' is True if the check fails.
    """
    return await guardrails_obj.check_critical_command(input_data, context)
