import gymnasium as gym
from gymnasium import spaces
from envs.shaping_wrappers.pbrs_common import PotentialBasedShapingWrapper
import numpy as np

class PointMazeNotGoalDirectedWrapper(gym.Wrapper):
    def __init__(self, env, on_step_rewards, reward_value=-1):
        super().__init__(env)

        self.on_step_rewards = on_step_rewards
        self.reward_value = reward_value

        # NOT USED: current step in observation to be markovian w.r.t truncation
        self.current_step = 0
        self.max_steps = 300

        obs_shape = self.env.unwrapped.point_env.observation_space.shape
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_shape[0] + 2,), dtype="float64")
        #self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_shape[0] + 3,), dtype="float64")

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.on_step_rewards:
            reward = self.reward_value
            if terminated:
                reward = 0

        self.current_step += 1

        return np.concatenate((obs['desired_goal'], obs['observation'])), reward, terminated, truncated, info
        #return np.concatenate((obs['desired_goal'], obs['observation'], [self.current_step / self.max_steps])), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_step = 0

        return np.concatenate((obs['desired_goal'], obs['observation'])), info
        #return np.concatenate((obs['desired_goal'], obs['observation'], [self.current_step / self.max_steps])), info


class PointMazeShapingWrapper(PotentialBasedShapingWrapper):
    def __init__(self, env, gamma, exponential, base, potential_scaling, potential_shift, reset_to_bias, reset_value):
        super().__init__(env, gamma, exponential, base, potential_scaling, potential_shift, reset_to_bias, reset_value)

        self.max_potential = (self.env.unwrapped.maze.map_width + self.env.unwrapped.maze.map_length) * self.env.unwrapped.maze.maze_size_scaling
    
    #outputs potential of current state (= distance to goal)
    def potential(self, state):
        current = state[:2]
        pot = np.linalg.norm(self.env.unwrapped.goal - current)

        return -pot

    def rescale_potential(self, potential):
        #by default is in (-max_dist, 0)
        #should be in (0, 1)
        return potential/self.max_potential + 1

    def get_obs(self):
        # ugly copy of _get_obs
        position = self.env.unwrapped.point_env.data.qpos.flatten()
        velocity = self.env.unwrapped.point_env.data.qvel.flatten()

        return np.concatenate((position, velocity))
