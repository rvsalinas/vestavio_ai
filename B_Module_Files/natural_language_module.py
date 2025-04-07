#!/usr/bin/env python
"""
Module: natural_language_module.py
Purpose:
  - Handles natural language tasks such as summarization, command parsing, text generation,
    and anomaly explanation.
  - Integrates with OpenAI’s gpt-4o-mini (or similar) model for text generation.
  - Incorporates advanced prompt-engineering strategies for clarity, context, and maintainability.

Usage Example:
  python natural_language_module.py
"""

import os
import logging
import re
from typing import Dict, Any, Optional, List

import openai


class NaturalLanguageModule:
    """
    A class for natural language tasks: summarization, command parsing, text generation,
    anomaly explanation, and advanced prompt-engineering with roles (system, developer, user).
    """

    def __init__(self, model_name: str = "gpt-4o-mini", logger: Optional[logging.Logger] = None):
        """
        :param model_name: The OpenAI model name (default: "gpt-4o-mini")
        :param logger: Optional logger instance.
        """
        if logger is None:
            self.logger = logging.getLogger("NaturalLanguageModule")
            if not self.logger.handlers:
                console = logging.StreamHandler()
                console.setLevel(logging.INFO)
                self.logger.addHandler(console)
                self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger

        self.logger.info(f"Initializing NaturalLanguageModule with OpenAI model '{model_name}'.")

        # Retrieve API key from environment (or rely on environment-based usage).
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            self.logger.warning("No OPENAI_API_KEY found in environment! Calls to OpenAI will fail.")

        self.model_name = model_name

    def _call_chat_api(
        self,
        system_message: str,
        user_message: str,
        developer_message: Optional[str] = None,
        max_tokens: int = 150,
        temperature: float = 0.3
    ) -> str:
        """
        Internal helper that sends a list of messages (system + user) to openai.ChatCompletion.create().
        
        :param system_message: The system message (overall style/tone).
        :param user_message: The user’s query or request.
        :param developer_message: (Optional) extra instructions merged into system context.
        :param max_tokens: The maximum tokens in the response.
        :param temperature: The sampling temperature (higher => more creative).
        :return: The model’s generated text (assistant role).
        """
        if not self.api_key:
            self.logger.error("No API key set. Returning user_message unmodified.")
            return user_message

        combined_system_msg = system_message or ""
        if developer_message:
            combined_system_msg += f"\n{developer_message}"

        messages = [
            {"role": "system", "content": combined_system_msg},
            {"role": "user", "content": user_message}
        ]

        try:
            response = openai.ChatCompletion.create(
                api_key=self.api_key,   # Pass the API key here
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error calling openai.chat_completions: {e}")
            return f"Error generating text: {str(e)}"

    def summarize_text(self, text: str) -> str:
        """
        Summarize the given text using the model, with advanced prompt-engineering.

        :param text: The text to summarize.
        :return: Summarized text.
        """
        self.logger.info("Summarizing text...")

        if len(text) < 50:
            self.logger.info("Text is quite short; returning as-is.")
            return text

        system_message = (
            "You are a concise, detail-oriented summarizer. "
            "Your goal is to produce short, clear summaries."
        )
        developer_message = (
            "Steps to summarize:\n"
            "1) Read the user’s text.\n"
            "2) Provide a concise summary in 1-2 sentences.\n"
            "3) Do not add extra commentary or disclaimers."
        )
        user_message = f"Please summarize the following text:\n\n{text}"

        return self._call_chat_api(
            system_message=system_message,
            user_message=user_message,
            developer_message=developer_message,
            max_tokens=120
        )

    def parse_command(self, command: str) -> Dict[str, Any]:
        """
        Parse a user command in plain English. Return a dict describing the action.
        Example: "Please increase temperature to 26.5 C" => {"action": "increase_temperature", "value": 26.5}

        :param command: The user’s textual command.
        :return: A dict with "action" and "value".
        """
        self.logger.info(f"Parsing command: {command}")
        cmd_lower = command.lower()

        # Simple rule-based approach:
        if "increase temperature" in cmd_lower:
            match = re.search(r"(\d+(\.\d+)?)", cmd_lower)
            temp_value = float(match.group(1)) if match else 25.0
            return {"action": "increase_temperature", "value": temp_value}
        elif "decrease temperature" in cmd_lower:
            match = re.search(r"(\d+(\.\d+)?)", cmd_lower)
            temp_value = float(match.group(1)) if match else 20.0
            return {"action": "decrease_temperature", "value": temp_value}

        return {"action": "unknown", "value": None}

    def generate_reply(
        self,
        user_text: str,
        system_persona: Optional[str] = None,
        developer_instructions: Optional[str] = None
    ) -> str:
        """
        Generate text from a user_text, with optional system persona & developer instructions.

        :param user_text: The user’s prompt.
        :param system_persona: High-level system message describing style/tone. (Optional)
        :param developer_instructions: Additional instructions that override user’s message. (Optional)
        :return: Generated reply from the model.
        """
        self.logger.info(f"Generating reply for user text: {user_text}")

        if not system_persona:
            system_persona = (
                "You are a helpful AI assistant. "
                "Provide answers in a clear, polite, and concise manner."
            )
        if not developer_instructions:
            developer_instructions = (
                "Please follow the user’s instructions carefully. If uncertain, ask clarifying questions."
            )

        return self._call_chat_api(
            system_message=system_persona,
            user_message=user_text,
            developer_message=developer_instructions,
            max_tokens=150
        )

    def check_if_command(self, user_message: str) -> bool:
        """
        Decide if user_message is definitely a command that should have '#' prefix.

        :param user_message: The raw text from the user.
        :return: True if it's likely a command, else False.
        """
        if not self.api_key:
            self.logger.warning("No API key found; defaulting to False.")
            return False

        developer_message = (
            "Classify if the user’s statement is definitely a command requiring a '#' prefix.\n"
            "Answer only with 'YES' or 'NO'."
        )
        system_msg = "You are a classification assistant."

        reply = self._call_chat_api(
            system_message=system_msg,
            user_message=f'User wrote: "{user_message}"',
            developer_message=developer_message,
            max_tokens=40,
            temperature=0.0
        ).strip().lower()

        return "yes" in reply

    def explain_anomaly(self, context: Dict[str, Any]) -> str:
        """
        Generate a short explanation for an anomaly, plus troubleshooting steps.

        :param context: Dict with keys like:
          - sensor: e.g. "dof_3"
          - reading: numeric or string reading
          - expected_range: e.g. "0.0 to 0.05"
          - anomaly_score (optional): float
        :return: Explanation string
        """
        self.logger.info("Generating anomaly explanation...")

        sensor = context.get("sensor", "unknown sensor")
        reading = context.get("reading", "N/A")
        expected_range = context.get("expected_range", "N/A")
        anomaly_score = context.get("anomaly_score", None)

        system_message = (
            "You are an expert sensor diagnostic assistant. "
            "You provide short, factual explanations and troubleshooting steps."
        )
        developer_message = (
            "1) Acknowledge the sensor reading.\n"
            "2) Explain why it might be outside the normal range.\n"
            "3) Suggest 1-2 short troubleshooting steps.\n"
            "Keep it concise."
        )

        user_msg = f"Sensor {sensor} reading is {reading}, expected range is {expected_range}."
        if anomaly_score is not None:
            user_msg += f"\nAnomaly score: {anomaly_score:.2f}"

        return self._call_chat_api(
            system_message=system_message,
            user_message=user_msg,
            developer_message=developer_message,
            max_tokens=120
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    nlm = NaturalLanguageModule()

    # Summarize example
    text_to_summarize = (
        "Here is a very long text that we want to summarize for the user to see. "
        "This text goes on and on, providing context for how the system might handle summarization. "
        "In a real scenario, we'd be using a powerful model to generate concise summaries."
    )
    print("Summary:", nlm.summarize_text(text_to_summarize), "\n")

    # Parse command example
    command = "Please Increase temperature to 26.5 C"
    print("Parsed command:", nlm.parse_command(command), "\n")

    # Generate a simple reply example
    user_prompt = "Explain how sensor anomalies can be addressed."
    print("Generated reply:", nlm.generate_reply(user_prompt), "\n")

    # Example check_if_command usage
    test_message = "Should I start the calibration now?"
    print("Is command?", nlm.check_if_command(test_message), "\n")

    # Explain anomaly example
    anomaly_context = {
        "sensor": "dof_3",
        "reading": 0.12,
        "expected_range": "0.0 to 0.05",
        "anomaly_score": 0.87
    }
    print("Anomaly explanation:", nlm.explain_anomaly(anomaly_context), "\n")