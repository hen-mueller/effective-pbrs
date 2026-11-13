from collections import deque

import numpy as np
from addict import addict

from stable_baselines3.common import utils
#from utils.default_addict import Default_Addict #=addit.Dict()

#TODO credit ASGRL implementation
class Q_Learning:
    def __init__(self, env, tensorboard_log, gamma=.95, epsilon=0.05, q_init=0, max_episode_len=250, lr=0.1, n_eval_episodes=10):#config, agent_id=0, logger=None, agent_name=''
        class Default_Addict(addict.Dict):
            def __missing__(self, key):
                return q_init
        self.q_table = Default_Addict()
        self.env = env
        self.logger = utils.configure_logger(0, tensorboard_log, tb_log_name="Q-Learning", reset_num_timesteps=True)
        #self.agent_id = agent_id
        #self.agent_name = agent_name
        self.action_dim = self.env.action_space.n
        #print(f'[INFO:{self.agent_name}:{self.agent_id}] action space: {self.action_dim}')

        self.total_step = 0
        self.i_episode = 0      # current episode id
        self.episode_step = 0       # number of steps in curr episode
        self.is_greedy = False      # whether to use greedy action selection
        self.is_eval = False
        self.eval_freq = 1000
        self.n_eval_episodes = n_eval_episodes

        ##################
        ##### Config #####
        ##################
        #self.config = config
        #self.n_episode = config.n_episode
        self.max_episode_len = max_episode_len#config.max_episode_len      # maximum step in an episode
        self.gamma = gamma #config.gamma       # default: 0.99
        self.lr = lr #if not config.lr_decay else config.max_epsilon
        #self.max_eps = config.max_epsilon
        #self.min_eps = config.min_epsilon
        #self.epsilon_decay_start_episode = -1   # to store the episode id when exploration-factor decay starts
        #self.epsilon_decay = config.epsilon_decay   # exploration factor decay rate
        #self._curr_eps = self.max_eps  # this variable is only used with decay_when_success
        #self.epsilon_success_decay = config.decay_when_success
        self.EPSILON = epsilon
        self.train_freq = 20

        ###################
        ##### Logging #####
        ###################
        self.eval_success_moving_avg = deque(maxlen=5)
        self.eval_score_moving_avg = deque(maxlen=5)
        self.score_moving_avg = deque(maxlen=5)
        self.max_score = -np.inf
        self.min_score = np.inf


    def _get_epsilon(self, i_episode, is_eval=False):
        """ Compute current exploration factor (epsilon) """
        #if self.total_step < self.update_start_from:
        #    return 1.0
        if is_eval:
            return 0.05
        return self.EPSILON
        # if following normal exploration factor decay
        #else:
        #    if self.epsilon_decay_start_episode < 0:
        #        self.epsilon_decay_start_episode = i_episode
        #    return max(self.min_eps, self.max_eps * (self.epsilon_decay ** (i_episode - self.epsilon_decay_start_episode)))

    def select_action(self, state):
        q_values = np.array([self.q_table[(*state, a)] for a in range(self.action_dim)])
        eps = self._get_epsilon(self.i_episode, is_eval=self.is_eval)
        if self.is_greedy or np.random.random() > eps:
            action = np.argmax(q_values)
        else:
            action = np.array(self.env.action_space.sample())
        return action.item() #TODO added .item() to avoid getting 0-dim np.array

    def q_update(self, experience):
        self.lr = self.lr #if not self.config.lr_decay else max(self.config.lr, self._get_epsilon(self.i_episode))
        state, action, reward, next_state, done, info = experience[:6]
        q_value = self.q_table[(*state, action)]
        next_value = np.max([self.q_table[(*next_state, a)] for a in range(self.action_dim)])
        # compute new estimation of q value
        not_done = int(not done)
        new_q_value = (1 - self.lr) * q_value + self.lr * (reward + not_done * self.gamma * next_value)
        self.q_table[(*state, action)] = new_q_value

    def _render(self):
        self.env.render()

    def post_episode_update(self, episode_done=True, episode_score=0):
        """ this function is called whenever an episode ends """
        self.i_episode += 1
        #if self.epsilon_success_decay and episode_done and episode_score > 0:
        #    self._curr_eps = max(self.min_eps, self._curr_eps * self.epsilon_decay)

    def train_episode(self):
        self.is_eval = False

        done = False
        score = 0
        self.episode_step = 0
        state, _ = self.env.reset()
        #curr_traj = []

        while not done: #and self.episode_step < self.max_episode_len:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
           
            #curr_traj.append((state, action, reward, next_state, done, info))
            # Q update
            self.q_update(experience=(state, action, reward, next_state, done, info))

            # post update
            score += reward
            state = next_state
            self.episode_step += 1
            self.total_step += 1

            # EVALUATION
            if self.total_step % self.eval_freq == 0:
                # indicated eval run => epsilon=0.05 fixed
                self.is_eval = True
                eval_rewards = []
                eval_lengths = []
                for _ in range(self.n_eval_episodes):
                    self.is_eval = True
                    eval_state, _ = self.eval_env.reset()
                    eval_reward = 0
                    i = 0
                    eval_done = False
                    while not eval_done and i < self.max_episode_len:
                        action = self.select_action(eval_state)
                        eval_next_state, reward, terminated, truncated, info = self.eval_env.step(action)
                        #TODO keep track of discounted rewards?
                        eval_reward += reward # * self.gamma**i
                        eval_done = terminated or truncated
                        eval_state = eval_next_state
                        i += 1
                    eval_rewards.append(eval_reward)
                    eval_lengths.append(i)
                    #log eval run length and total reward
                mean_reward = np.mean(eval_rewards)
                mean_ep_length = np.mean(eval_lengths)

                self.logger.record("eval/mean_reward", float(mean_reward))
                self.logger.record("eval/mean_ep_length", mean_ep_length)
                self.logger.dump(self.total_step)
            self.is_eval = False

        self.post_episode_update(episode_done=done, episode_score=score)

        self.max_score = max(self.max_score, score)
        self.min_score = min(self.min_score, score)

        #ADDED flags for evaluation logging
        self.episode_score = score

        self.logger.record("rollout/ep_rew_mean", self.episode_score)
        self.logger.record("rollout/ep_len_mean", self.episode_step)
        self.logger.dump(self.total_step)

    def learn(self, total_timesteps, callback):
        self.eval_env = callback

        while True:
            if self.total_step > total_timesteps:
                print(f'\n[INFO] Maximum number of steps reached: {self.total_step} > {total_timesteps}\n')
                break

            self.episode_score = 0

            self.train_episode()
