import gymnasium as gym
from envs.shaping_wrappers.pbrs_common import PotentialBasedShapingWrapper

class CartPoleShapingWrapper(PotentialBasedShapingWrapper):
    def __init__(self, env, gamma, exponential, base, potential_scaling, potential_shift, reset_to_bias, reset_value):#, max_potential):
        super().__init__(env, gamma, exponential, base, potential_scaling, potential_shift, reset_to_bias, reset_value)
        #self.max_potential = 4.8 #for x position in cartpole
        self.max_potential = 0.418 #for pole angle in cartpole

    #outputs potential of current state (= distance to x=0)
    def potential(self, state):
        pot = -abs(state[2]) #/ self.max_potential
        return pot

    def rescale_potential(self, potential):
        #by default is in (-4.8, 0)
        #should be in (0, 1)
        return (potential/self.max_potential) + 1

    def get_obs(self):
        return self.env.unwrapped.state

