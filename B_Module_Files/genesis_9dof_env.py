# File: /Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/genesis_9dof_env.py

import gym
import numpy as np
import genesis as gs

class Genesis9DofEnv(gym.Env):
    """
    A custom Gym environment for controlling a 9-DOF manipulator in Genesis.
    Observations: 
      (pos + vel) => 9 + 9 = 18D
    Actions:
      Could be position deltas or torque values => shape (9,)
    Rewards:
      Example: negative sum of torques to minimize energy usage, plus a success condition?
    """

    def __init__(self):
        super(Genesis9DofEnv, self).__init__()

        # 1) Set up observation space (18D)
        obs_dim = 18
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # 2) Action space (9D)
        act_dim = 9
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32
        )

        # 3) Initialize Genesis
        gs.init(backend=gs.cpu)
        self.scene = gs.Scene(show_viewer=False)

        # Plane
        plane = self.scene.add_entity(gs.morphs.Plane())

        # 9-DOF manipulator (placeholder: Franka is 7 DOFs, but we assume 9 for demonstration)
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
                collision=True
            )
        )

        self.scene.build()

        self.max_steps = 200
        self.current_step = 0

    def _get_obs(self):
        dof_pos = self.robot.get_dofs_position()  # shape (9,) if your URDF has 9 dofs
        dof_vel = self.robot.get_dofs_velocity()  # shape (9,)
        obs = np.concatenate([dof_pos, dof_vel], axis=0)  # shape=(18,)
        return obs.astype(np.float32)

    def reset(self):
        """
        Reset the environment (random or fixed init).
        Return the initial observation.
        """
        self.current_step = 0
        zero_positions = np.zeros(9, dtype=np.float32)
        self.robot.set_dofs_position(zero_positions)
        self.scene.step()
        return self._get_obs()

    def step(self, action):
        """
        action is shape (9,) - e.g., delta pos
        We'll add it to current positions for position control
        """
        self.current_step += 1

        current_positions = self.robot.get_dofs_position()
        new_positions = current_positions + action  # naive approach
        self.robot.control_dofs_position(new_positions)
        self.scene.step()  # step the simulation

        obs = self._get_obs()

        # Convert torques to np array before summing
        torques = self.robot.get_dofs_force()
        torques_np = np.array(torques, dtype=np.float32)
        torque_cost = float(np.sum(np.abs(torques_np)))

        # Negative reward for torque usage
        reward = -torque_cost

        done = (self.current_step >= self.max_steps)
        info = {}

        return obs, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        gs.release()