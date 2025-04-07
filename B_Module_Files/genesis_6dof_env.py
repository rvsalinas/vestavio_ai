# File: /home/ec2-user/energy_optimization_project/B_Module_Files/genesis_6dof_env.py

import numpy as np
import gym
from gym import spaces

class Genesis6DofEnv(gym.Env):
    """
    A minimal Gym environment for a 6-DOF manipulator.
    Observations: concatenation of positions (6) and velocities (6) -> 12 features.
    Actions: continuous changes for each of the 6 DOFs.
    """
    def __init__(self):
        super(Genesis6DofEnv, self).__init__()
        # Define action space: each joint can change between -0.05 and 0.05
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(6,), dtype=np.float32)
        # Define observation space: positions (6) and velocities (6)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.state = np.zeros(12, dtype=np.float32)
        self.step_count = 0
        self.max_steps = 1000  # Arbitrary maximum number of steps per episode

    def reset(self):
        self.state = np.zeros(12, dtype=np.float32)
        self.step_count = 0
        return self.state

    def step(self, action):
        # Update positions and velocities in a simple manner for demonstration.
        positions = self.state[:6] + action
        velocities = action  # In this dummy environment, we set velocities equal to the action.
        self.state = np.concatenate([positions, velocities])
        # Reward: negative L2 norm of the positions
        reward = -np.linalg.norm(positions)
        self.step_count += 1
        done = self.step_count >= self.max_steps
        info = {}
        return self.state, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass