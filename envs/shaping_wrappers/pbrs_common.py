from abc import abstractmethod
import gymnasium as gym

class PotentialBasedShapingWrapper(gym.Wrapper):
    def __init__(self, env, gamma, exponential, base, potential_scaling, potential_shift, reset_to_bias, reset_value):#, max_potential):
        super().__init__(env)
        self.gamma = gamma
        self.exponential = exponential
        self.base = base
        self.potential_scaling = potential_scaling
        self.potential_shift = potential_shift
        self.reset_to_bias = reset_to_bias
        self.reset_value = reset_value
        self.max_potential = None

    #outputs potential of current state (= distance to x=0)
    #in []
    @abstractmethod
    def potential(self, state):
        pass

    # for positive potential functions
    # from [0, max_potential] to [0, 1]
    def rescale_potential(self, potential):
        return potential/self.max_potential

    #output in [1/b, 1]
    #TODO check normalization
    def exponential_potential(self, potential):
        #potential_normalization = self.base
        return self.base**(potential-1)

    # shifts shaping by constant self.potential_shift
    # self.potential_shift should be equal to difference of Q_init and R
    def shift_potential(self, potential):
        return potential + self.potential_shift / (self.gamma - 1)

    def is_terminal(self, potential, done):
        if done:
            if not self.reset_to_bias:
                return 0
            if self.reset_value is not None:
                return self.reset_value
            return self.shift_potential(0.0)
        else:
            return potential

    @abstractmethod
    def get_obs(self):
        pass

    def step(self, action):
        prev_obs = self.get_obs()
        obs, reward, terminated, truncated, info = self.env.step(action)

        curr_obs = self.get_obs()
        done = terminated or truncated
        cur_pot, prev_pot = self.potential(curr_obs), self.potential(prev_obs)
        cur_pot, prev_pot = self.rescale_potential(cur_pot), self.rescale_potential(prev_pot)
        if self.exponential:
            cur_pot, prev_pot = self.exponential_potential(cur_pot), self.exponential_potential(prev_pot)
        reward = reward + (self.gamma * self.is_terminal(self.shift_potential(self.potential_scaling * cur_pot), done)
                            - self.is_terminal(self.shift_potential(self.potential_scaling * prev_pot), False))
        return obs, reward, terminated, truncated, info
