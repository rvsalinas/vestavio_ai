#!/usr/bin/env python
import gym
import numpy as np
from gym import spaces

class GenesisDroneEnv(gym.Env):
    """
    A custom Gym environment for controlling a Drone with 4 DOF in a simplified simulation.
    
    Observations:
      - 8 dimensions: 4 positions and 4 velocities.
    
    Actions:
      - 4-dimensional continuous action controlling changes in positions.
    
    Rewards:
      - Negative L2 norm of the current joint positions (to encourage the drone to remain near the origin).
    
    Episodes end after a fixed number of timesteps.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(GenesisDroneEnv, self).__init__()
        
        obs_dim = 8  # 4 positions + 4 velocities
        self.observation_space = spaces.Box(low=-1e3, high=1e3, shape=(obs_dim,), dtype=np.float32)
        # Action space: changes in positions (4 DOF)
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(4,), dtype=np.float32)
        
        self.max_steps = 200
        self.current_step = 0
        self.dt = 1.0
        
        # Initialize state: positions and velocities.
        self.pos = np.zeros(4, dtype=np.float32)
        self.vel = np.zeros(4, dtype=np.float32)
    
    def _get_obs(self):
        return np.concatenate([self.pos, self.vel]).astype(np.float32)
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.pos = np.zeros(4, dtype=np.float32)
        self.vel = np.zeros(4, dtype=np.float32)
        return self._get_obs(), {}
    
    def step(self, action):
        self.current_step += 1
        action = np.clip(np.array(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        
        prev_pos = self.pos.copy()
        # Simple Euler integration to update positions.
        self.pos = self.pos + action
        self.vel = (self.pos - prev_pos) / self.dt
        
        # Reward: negative L2 norm of the positions.
        reward = -np.linalg.norm(self.pos)
        done = self.current_step >= self.max_steps
        info = {}
        return self._get_obs(), reward, done, False, info
    
    def render(self, mode='human'):
        print(f"Step: {self.current_step}, pos: {self.pos}, vel: {self.vel}")
    
    def close(self):
        pass