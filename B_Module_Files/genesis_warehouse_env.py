# File: /home/ec2-user/energy_optimization_project/B_Module_Files/genesis_warehouse_env.py

import gym
import numpy as np
import genesis as gs

class GenesisWarehouseEnv(gym.Env):
    """
    A custom Gym environment for controlling an Autonomous Warehouse Robot in Genesis.
    Observations:
      - Concatenated DOF positions and velocities (e.g., 8 + 8 = 16D) from wheels and manipulator.
    Actions:
      - Control signals for 8 actuators (2 wheels + 6 manipulator joints), shape (8,).
    Rewards:
      - Negative sum of absolute torques to encourage energy efficiency.
    """

    def __init__(self):
        super(GenesisWarehouseEnv, self).__init__()

        # Observation space: 16 dimensions (8 positions + 8 velocities)
        obs_dim = 16
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space: 8 dimensions (2 wheels + 6 manipulator joints)
        act_dim = 8
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

        # Initialize Genesis with CPU backend
        gs.init(backend=gs.cpu)
        self.scene = gs.Scene(show_viewer=False)
        self.scene.add_entity(gs.morphs.Plane())

        # Use the absolute EC2 path for the MJCF file
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file="/home/ec2-user/energy_optimization_project/G_Genesis_Files/xml/warehouse_robot.xml",
                collision=True
            )
        )
        self.scene.build()

        self.max_steps = 200
        self.current_step = 0

    def _get_obs(self):
        """
        Returns the current observation (concatenated positions and velocities)
        with NaNs replaced and then clipped to a reasonable range.
        """
        dof_pos = np.array(self.robot.get_dofs_position(), dtype=np.float32)
        dof_vel = np.array(self.robot.get_dofs_velocity(), dtype=np.float32)
        # Replace NaNs and infinite values
        dof_pos = np.nan_to_num(dof_pos, nan=0.0, posinf=1e6, neginf=-1e6)
        dof_vel = np.nan_to_num(dof_vel, nan=0.0, posinf=1e6, neginf=-1e6)
        obs = np.concatenate([dof_pos, dof_vel], axis=0)
        # Clip observation to avoid extreme values
        obs = np.clip(obs, -1e3, 1e3)
        return obs.astype(np.float32)

    def reset(self):
        """
        Resets the environment and returns the initial observation.
        """
        self.current_step = 0
        zero_positions = np.zeros(8, dtype=np.float32)
        self.robot.set_dofs_position(zero_positions)
        self.scene.step()
        return self._get_obs()

    def step(self, action):
        """
        Applies the action, steps the simulation, and returns:
          - observation (sanitized and clipped),
          - reward (negative sum of absolute torques),
          - done flag,
          - info dict.
        """
        self.current_step += 1

        # Get current positions and sanitize them
        current_positions = np.array(self.robot.get_dofs_position(), dtype=np.float32)
        current_positions = np.nan_to_num(current_positions, nan=0.0, posinf=1e6, neginf=-1e6)
        current_positions = np.clip(current_positions, -1e3, 1e3)

        # Ensure action is a numpy array and compute new positions
        action = np.array(action, dtype=np.float32)
        new_positions = current_positions + action
        new_positions = np.nan_to_num(new_positions, nan=0.0, posinf=1e6, neginf=-1e6)
        new_positions = np.clip(new_positions, -1e3, 1e3)

        # Apply control and step the simulation
        self.robot.control_dofs_position(new_positions)
        self.scene.step()

        # Obtain and sanitize observation
        obs = self._get_obs()

        # Compute reward from torques
        torques = np.array(self.robot.get_dofs_force(), dtype=np.float32)
        torques = np.nan_to_num(torques, nan=0.0, posinf=1e6, neginf=-1e6)
        reward = -float(np.sum(np.abs(torques)))

        done = (self.current_step >= self.max_steps)
        info = {}

        return obs, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        gs.release()