#!/usr/bin/env python

import gym
import numpy as np
import genesis as gs

class GenesisUrbanEnv(gym.Env):
    """
    A custom Gym environment for controlling an Autonomous Delivery Robot for Urban Last-Mile Logistics in Genesis.
    
    Observations:
      - Concatenated actuated DOF positions and velocities (5 positions + 5 velocities = 10 dimensions)
    
    Actions:
      - A 5-dimensional vector representing control adjustments for each DOF.
    
    Rewards:
      - The reward is computed as the difference in distance from the target configuration before and after taking an action,
        minus a penalty for large actions. A bonus is given if the robot's configuration is within a small threshold of the target.
    """

    def __init__(self):
        super(GenesisUrbanEnv, self).__init__()

        # Define observation space (5 positions + 5 velocities = 10 dimensions)
        obs_dim = 10
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Define action space (5 dimensions)
        act_dim = 5
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32
        )

        # Initialize Genesis using the CPU backend.
        gs.init(backend=gs.cpu)
        self.scene = gs.Scene(show_viewer=False)

        # Add a ground plane.
        plane = self.scene.add_entity(gs.morphs.Plane())

        # Add the Urban Delivery Robot using the absolute path to urban_robot.xml.
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file="/home/ec2-user/energy_optimization_project/G_Genesis_Files/xml/urban_robot.xml",
                collision=True
            )
        )

        self.scene.build()

        self.max_steps = 200
        self.current_step = 0

        # Set the target configuration for the robot's DOFs (desired 5-dimensional configuration)
        self.target_position = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        # Initialize last error to track progress.
        self.last_error = None

    def _get_obs(self):
        # Obtain DOF positions and velocities (each expected as a (5,) array)
        dof_pos = self.robot.get_dofs_position()
        dof_vel = self.robot.get_dofs_velocity()
        obs = np.concatenate([dof_pos, dof_vel], axis=0)
        return obs.astype(np.float32)

    def reset(self):
        """
        Resets the environment by setting the robot's DOF positions to zeros and computing the initial error.
        Returns the initial observation.
        """
        self.current_step = 0
        zero_positions = np.zeros(5, dtype=np.float32)
        self.robot.set_dofs_position(zero_positions)
        self.scene.step()
        # Compute initial error from target.
        current_positions = self.robot.get_dofs_position()
        self.last_error = np.linalg.norm(current_positions - self.target_position)
        return self._get_obs()

    def step(self, action):
        """
        Applies the given action, steps the simulation, and computes the reward.
        
        Args:
            action (np.array): 5-dimensional control input.
            
        Returns:
            obs (np.array): Next observation (10D).
            reward (float): Reward computed as improvement in error minus an action penalty.
            done (bool): Whether the episode has ended.
            info (dict): Additional information (e.g., current position error).
        """
        self.current_step += 1

        current_positions = self.robot.get_dofs_position()
        # Apply action (simple additive control)
        new_positions = current_positions + action
        self.robot.control_dofs_position(new_positions)
        self.scene.step()

        obs = self._get_obs()

        # Compute the new error relative to the target configuration.
        current_error = np.linalg.norm(new_positions - self.target_position)
        # Reward is the improvement in error minus an action penalty.
        reward = (self.last_error - current_error) - 0.1 * np.linalg.norm(action)
        # Bonus if the robot reaches close to the target.
        if current_error < 0.1:
            reward += 10.0

        self.last_error = current_error

        done = self.current_step >= self.max_steps
        info = {"position_error": current_error}

        return obs, reward, done, info

    def render(self, mode='human'):
        # Add rendering logic if necessary.
        pass

    def close(self):
        gs.release()