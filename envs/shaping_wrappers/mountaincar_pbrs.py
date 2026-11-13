import gymnasium as gym
from envs.shaping_wrappers.pbrs_common import PotentialBasedShapingWrapper

class MountainCarShapingWrapper(PotentialBasedShapingWrapper):
    def __init__(self, env, gamma, exponential, base, potential_scaling, potential_shift, reset_to_bias, reset_value):
        super().__init__(env, gamma, exponential, base, potential_scaling, potential_shift, reset_to_bias, reset_value)
        self.max_potential = 0.07 #for velocity

    #outputs potential of current state (absolute velocity)
    def potential(self, state):
        pot = abs(state[1]) - self.max_potential
        return pot

    def rescale_potential(self, potential):
        #by default is in (0.07, 0)
        #should be in (-1, 0)
        return potential/self.max_potential

    def get_obs(self):
        return self.env.unwrapped.state

    #ADDED FOR NEGATIVE POTENTIAL
    def exponential_potential(self, potential):
        #potential_normalization = self.base
        return self.base**(potential)