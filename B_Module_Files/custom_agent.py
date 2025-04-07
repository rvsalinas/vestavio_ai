# custom_agent.py
Agent as BaseAgent

class Agent(BaseAgent):
    def __init__(self, *args, handoffs=None, **kwargs):
        if handoffs is None:
            handoffs = []  # Ensure a default empty list
        super().__init__(*args, handoffs=handoffs, **kwargs)