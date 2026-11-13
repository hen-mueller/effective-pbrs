import gymnasium as gym
from envs.shaping_wrappers.pbrs_common import PotentialBasedShapingWrapper

class VStarGridworldShapingWrapper(PotentialBasedShapingWrapper):
    def __init__(self, env, gamma, exponential, base, potential_scaling, potential_shift, reset_to_bias, reset_value):#, max_potential):
        super().__init__(env, gamma, exponential, base, potential_scaling, potential_shift, reset_to_bias, reset_value)
        self.max_potential = self.env.unwrapped.goal_pos[0] + self.env.unwrapped.goal_pos[1]

    #outputs potential of current state (= distance to x=0)
    #potential for terminal states is always 0
    #gamma
        #gamma**d(goal) - 1
    def potential(self, state):
        pot = self.gamma**(abs(state[0] - self.env.unwrapped.goal_pos[0]) + abs(state[1] - self.env.unwrapped.goal_pos[1]) - 1)
        return pot

    def rescale_potential(self, potential):
        #by default is in (-max_potential, 0)
        #should be in (0, 1)
        return potential
        #return (potential/self.max_potential) + 1

    def get_obs(self):
        return self.env.unwrapped.agent_pos
