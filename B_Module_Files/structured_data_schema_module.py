#!/usr/bin/env python
"""
structured_data_schema.py

Purpose:
  This module defines Pydantic models (and corresponding JSON Schemas) for use with
  OpenAI’s “Structured Outputs” feature. Each model enforces a strict schema, ensuring
  consistent JSON responses that our system can reliably parse.

Usage:
  1. Import the models in your codebase:
        from structured_data_schema import MathReasoningSchema, UIElement, ...
     2. Provide these models to the OpenAI SDK (Python or JS) as your `response_format`.
     3. The model will then output JSON that strictly follows the schema(s) below.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# ------------------------------------------------------------------------------
# 1) Example: Step-by-Step Math Reasoning Schema
# ------------------------------------------------------------------------------
class Step(BaseModel):
    """
    Represents a single step in a chain-of-thought style math solution.
    """
    explanation: str = Field(
        ...,
        description="A textual explanation for the step, e.g., 'Subtract 7 from both sides.'"
    )
    output: str = Field(
        ...,
        description="The expression or partial result at this step, e.g., '8x = -30'."
    )


class MathReasoningSchema(BaseModel):
    """
    Represents a structured output for a math problem solution,
    including step-by-step reasoning plus a final answer.
    """
    steps: List[Step] = Field(
        ...,
        description="List of step-by-step reasoning steps in chronological order."
    )
    final_answer: str = Field(
        ...,
        description="The final computed answer after all steps."
    )


# ------------------------------------------------------------------------------
# 2) Example: Simple UI Schema (Recursive)
# ------------------------------------------------------------------------------
class UIType(str, Enum):
    """
    Allowed UI component types for demonstration.
    """
    DIV = "div"
    BUTTON = "button"
    HEADER = "header"
    SECTION = "section"
    FIELD = "field"
    FORM = "form"


class UIAttribute(BaseModel):
    """
    Arbitrary attribute for a UI element (e.g., onClick, className, placeholder, etc.).
    """
    name: str = Field(..., description="The name of the attribute, e.g., 'onClick'.")
    value: str = Field(..., description="The value assigned to that attribute.")


class UIElement(BaseModel):
    """
    Represents a single UI element in a hierarchical structure (e.g., a form, a button, etc.).
    """
    type: UIType = Field(..., description="The type of this UI component.")
    label: str = Field(..., description="A label for the component, or an empty string if not applicable.")
    children: List["UIElement"] = Field(
        default_factory=list,
        description="Child UI elements nested under this component."
    )
    attributes: List[UIAttribute] = Field(
        default_factory=list,
        description="List of arbitrary attributes for customization."
    )

    class Config:
        # This is required for recursive models in Pydantic:
        orm_mode = True
        arbitrary_types_allowed = True


# Because of the recursive reference, we need to update forward references:
UIElement.update_forward_refs()


# ------------------------------------------------------------------------------
# 3) Example: Content Moderation Schema
# ------------------------------------------------------------------------------
class ModerationCategory(str, Enum):
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    SELF_HARM = "self_harm"


class ModerationSchema(BaseModel):
    """
    Example structured output for a moderation system. The model
    either flags the content or deems it non-violating.
    """
    is_violating: bool = Field(
        ...,
        description="Indicates if the content is violating guidelines."
    )
    category: Optional[ModerationCategory] = Field(
        None,
        description="Type of violation if the content is violating guidelines; null if not violating."
    )
    explanation_if_violating: Optional[str] = Field(
        None,
        description="Explanation if the content is violating. Null if not violating."
    )


# ------------------------------------------------------------------------------
# 4) Example: Minimal “Action Log” entry
# ------------------------------------------------------------------------------
class ActionLogEntry(BaseModel):
    """
    Represents a single action taken by a user, or system response to that action,
    in strictly defined fields.
    """
    user_action: str = Field(..., description="Text describing the user's command or action.")
    timestamp: str = Field(..., description="Timestamp in ISO format when the action occurred.")
    system_response: Optional[str] = Field(
        None,
        description="The system's official response or outcome for this action, if any."
    )
    # Additional fields can be added here as needed (e.g., status, severity, etc.)


# ------------------------------------------------------------------------------
# Example of how to create a single "master" schema or multiple specialized ones.
# Here, we just define them separately so you can pick and choose as needed.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick self-test: Generate JSON schema for each model
    import json

    print("=== MathReasoningSchema JSON Schema ===")
    print(json.dumps(MathReasoningSchema.schema(), indent=2))

    print("\n=== UIElement JSON Schema ===")
    print(json.dumps(UIElement.schema(), indent=2))

    print("\n=== ModerationSchema JSON Schema ===")
    print(json.dumps(ModerationSchema.schema(), indent=2))

    print("\n=== ActionLogEntry JSON Schema ===")
    print(json.dumps(ActionLogEntry.schema(), indent=2))