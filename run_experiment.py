import os
import numpy as np
import argparse
import gymnasium as gym
from stable_baselines3 import DQN, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback

from envs.shaping_wrappers.gridworld_pbrs import GridworldShapingWrapper
from envs.shaping_wrappers.vstar_gridworld_pbrs import VStarGridworldShapingWrapper
from envs.shaping_wrappers.cartpole_pbrs import CartPoleShapingWrapper
from envs.shaping_wrappers.mountaincar_pbrs import MountainCarShapingWrapper
from envs.shaping_wrappers.pointmaze_pbrs import PointMazeShapingWrapper, PointMazeNotGoalDirectedWrapper
from envs.gridworld import SquareGridworld
from minigrid.wrappers import  FullyObsWrapper
from q_learning.q_learning import Q_Learning
from utils.ddqn import attach_double_dqn_train

from gymnasium.wrappers import TransformObservation

import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

ENV_CHOICES = ['gridworld', 'vstar-gridworld', 'cartpole', 'mountaincar', 'pointmaze', 'pointmaze-medium', 'pointmaze-large', 'pointmaze-giant']
AGENT_CHOICES = ['dqn', 'ddqn', 'sac', 'td3']

parser = argparse.ArgumentParser()
parser.add_argument('env', choices=ENV_CHOICES)
parser.add_argument('log_dir')
parser.add_argument('-a', '--agent', choices=AGENT_CHOICES, default='sac')
parser.add_argument('--device', type=str, default='cuda') # cuda (gpu) or cpu, only for deep RL
parser.add_argument('-steps', '--maxsteps', type=int, default=250_000)
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3) #pointmaze only
parser.add_argument('-g', '--gamma', type=float, default=0.95)#TODO default was 0.99
parser.add_argument('-s', '--activate_shaping', action='store_true')
parser.add_argument('-e', '--exponential', action='store_true')
parser.add_argument('-b', '--base', type=float, default=2)
parser.add_argument('-p', '--potential_scaling', type=float, default=1)
parser.add_argument('--potential_shift', type=float, default=0, help='Constant shift to potential function (rescaled with 1/(gamma - 1))')
parser.add_argument('-q', '--q_init', type=float, default=0) #gridworld only
parser.add_argument('--gridworld_on_step', action='store_true') #gridworld, pointmaze only
parser.add_argument('-rs', '--on_step_reward_value', type=float, default=-1)
parser.add_argument('-rg', '--goal_reward_value', type=float, default=-1)
parser.add_argument('--reset_to_bias', action='store_true', help='Sets potential of terminal states to the potential-shift bias instead of zero')
parser.add_argument('-rv', '--reset_value', type=float, default=None, help='Explicit terminal potential value used when --reset_to_bias is enabled; defaults to the potential-shift bias')
parser.add_argument('--double_dqn', action='store_true')
args = parser.parse_args()

max_steps = args.maxsteps

def make_gridworld(size):
    base_env = SquareGridworld(
        size=size,
        on_step_rewards=args.gridworld_on_step,
        on_step_reward_value=args.on_step_reward_value,
        goal_reward_value=args.goal_reward_value,
        max_steps=250,
    )
    return TransformObservation(
        base_env,
        lambda obs, base_env=base_env: np.array(base_env.unwrapped.agent_pos),
        observation_space=gym.spaces.MultiDiscrete(np.array([size, size])),
    )

#TODO wrap evaluation environment in MonitorWrapper to remove error msg
if args.env == 'gridworld':
    grid_size = 27
    env = make_gridworld(grid_size)

    eval_callback = make_gridworld(grid_size)
    if args.activate_shaping:
        env = GridworldShapingWrapper(env, args.gamma, args.exponential, args.base, args.potential_scaling, args.potential_shift, args.reset_to_bias, args.reset_value)
    agent = Q_Learning(env, gamma=args.gamma, tensorboard_log=args.log_dir, q_init=args.q_init)
elif args.env == 'vstar-gridworld':
    grid_size = 27
    env = make_gridworld(grid_size)

    eval_callback = make_gridworld(grid_size)
    if args.activate_shaping:
        env = VStarGridworldShapingWrapper(env, args.gamma, args.exponential, args.base, args.potential_scaling, args.potential_shift, args.reset_to_bias, args.reset_value)
    agent = Q_Learning(env, gamma=args.gamma, tensorboard_log=args.log_dir, q_init=args.q_init)
elif args.env == 'cartpole':
    env = gym.make('CartPole-v1')
    eval_callback = EvalCallback(env, log_path=args.log_dir, eval_freq=500,
                        deterministic=True, render=False, verbose=0)
    if args.activate_shaping:
        env = CartPoleShapingWrapper(env, args.gamma, args.exponential, args.base, args.potential_scaling, args.potential_shift, args.reset_to_bias, args.reset_value)
    #https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml#L21
    #agent = DQN("MlpPolicy", env, gamma=args.gamma, exploration_fraction=1000/max_steps , learning_starts=500, tensorboard_log=args.log_dir, device='cpu', buffer_size=50000)#verbose=1
    if args.agent == 'dqn':
        agent = DQN("MlpPolicy", env, gamma=args.gamma, exploration_fraction=10000/max_steps , learning_starts=1000, tensorboard_log=args.log_dir, device=args.device, buffer_size=50000, learning_rate=args.learning_rate)#verbose=1
        if args.double_dqn:
            agent = attach_double_dqn_train(agent)
    elif args.agent == 'ddqn':
        agent = DQN("MlpPolicy", env, gamma=args.gamma, exploration_fraction=10000/max_steps , learning_starts=1000, tensorboard_log=args.log_dir, device=args.device, buffer_size=50000, learning_rate=args.learning_rate)#verbose=1
        agent = attach_double_dqn_train(agent)
    else:
        raise ValueError("Unsupported agent for cartpole: " + args.agent)
elif args.env == 'mountaincar':
    env = gym.make('MountainCar-v0')
    eval_callback = EvalCallback(env, log_path=args.log_dir, eval_freq=500, n_eval_episodes=5,
                        deterministic=True, render=False, verbose=0)
    if args.activate_shaping:
        env = MountainCarShapingWrapper(env, args.gamma, args.exponential, args.base, args.potential_scaling, args.potential_shift, args.reset_to_bias, args.reset_value)
    #https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml#L37
    if args.agent == 'dqn':
        agent = DQN("MlpPolicy", env, gamma=args.gamma, exploration_fraction=10000/max_steps , learning_starts=1000, tensorboard_log=args.log_dir, device=args.device, buffer_size=25000, learning_rate=args.learning_rate, target_update_interval=1000, batch_size=128, gradient_steps=2)
        if args.double_dqn:
            agent = attach_double_dqn_train(agent)
    elif args.agent == 'ddqn':
        agent = DQN("MlpPolicy", env, gamma=args.gamma, exploration_fraction=10000/max_steps , learning_starts=1000, tensorboard_log=args.log_dir, device=args.device, buffer_size=25000, learning_rate=args.learning_rate, target_update_interval=1000, batch_size=128, gradient_steps=2)
        agent = attach_double_dqn_train(agent)
    else:
        raise ValueError("Unsupported agent for mountaincar: " + args.agent)

elif args.env == 'pointmaze':
    FIXED_MAZE = [[1, 1, 1, 1, 1],
        [1, 'g', 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 'r', 1],
        [1, 1, 1, 1, 1]]

    env = gym.make('PointMaze_UMaze-v3', continuing_task=False, maze_map=FIXED_MAZE)
    # combine current goal progress with rest of observation for new observatons, remove current goal from obs
    env = PointMazeNotGoalDirectedWrapper(env, args.gridworld_on_step, args.on_step_reward_value)
    eval_callback = EvalCallback(env, log_path=args.log_dir, eval_freq=2000,
                        deterministic=True, render=False, verbose=0)
    if args.activate_shaping:
        env = PointMazeShapingWrapper(env, args.gamma, args.exponential, args.base, args.potential_scaling, args.potential_shift, args.reset_to_bias, args.reset_value)

    if args.agent == 'sac':
        agent = SAC("MlpPolicy", env, gamma=args.gamma, tensorboard_log=args.log_dir, device=args.device, learning_starts=1000, learning_rate=args.learning_rate)#, buffer_size=50000), learning_starts=1000
    elif args.agent == 'td3':
        agent = TD3("MlpPolicy", env, gamma=args.gamma, tensorboard_log=args.log_dir, device=args.device, learning_rate=args.learning_rate)#, replay_buffer_kwargs={"handle_timeout_termination":False})
    else:
        raise ValueError("Unsupported agent for pointmaze: " + args.agent)
elif args.env == 'pointmaze-medium':
    FIXED_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1],
            [1, 'g', 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 'r', 1],
            [1, 1, 1, 1, 1, 1, 1, 1]]

    env = gym.make('PointMaze_Medium-v3', continuing_task=False, maze_map=FIXED_MAZE)
    # combines current goal progress with rest of observation for new observatons, remove current goal from obs
    env = PointMazeNotGoalDirectedWrapper(env, args.gridworld_on_step, args.on_step_reward_value)
    eval_callback = EvalCallback(env, log_path=args.log_dir, eval_freq=2000,
                        deterministic=True, render=False, verbose=0)
    if args.activate_shaping:
        env = PointMazeShapingWrapper(env, args.gamma, args.exponential, args.base, args.potential_scaling, args.potential_shift, args.reset_to_bias, args.reset_value)

    if args.agent == 'sac':
        agent = SAC("MlpPolicy", env, gamma=args.gamma, tensorboard_log=args.log_dir, device=args.device, learning_starts=1000, learning_rate=args.learning_rate)#, buffer_size=50000), learning_starts=1000
    elif args.agent == 'td3':
        agent = TD3("MlpPolicy", env, gamma=args.gamma, tensorboard_log=args.log_dir, device=args.device, learning_rate=args.learning_rate)#, replay_buffer_kwargs={"handle_timeout_termination":False})
    else:
        raise ValueError("Unsupported agent for pointmaze-medium: " + args.agent)
elif args.env == 'pointmaze-large':
    FIXED_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1,'g',0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 'r', 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

    env = gym.make('PointMaze_Large-v3', continuing_task=False, maze_map=FIXED_MAZE)
    # combines current goal progress with rest of observation for new observatons, remove current goal from obs
    env = PointMazeNotGoalDirectedWrapper(env, args.gridworld_on_step, args.on_step_reward_value)
    eval_callback = EvalCallback(env, log_path=args.log_dir, eval_freq=2000,
                        deterministic=True, render=False, verbose=0)
    if args.activate_shaping:
        env = PointMazeShapingWrapper(env, args.gamma, args.exponential, args.base, args.potential_scaling, args.potential_shift, args.reset_to_bias, args.reset_value)

    if args.agent == 'sac':
        agent = SAC("MlpPolicy", env, gamma=args.gamma, tensorboard_log=args.log_dir, device=args.device, learning_starts=1000, learning_rate=args.learning_rate, )#, buffer_size=50000), learning_starts=1000
    elif args.agent == 'td3':
        agent = TD3("MlpPolicy", env, gamma=args.gamma, tensorboard_log=args.log_dir, device=args.device, learning_rate=args.learning_rate, learning_starts=10000) #, replay_buffer_kwargs={"handle_timeout_termination":False})
    else:
        raise ValueError("Unsupported agent for pointmaze-large: " + args.agent)
elif args.env == 'pointmaze-giant':
    GIANT_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 'g', 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
                [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 'r', 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

    env = gym.make('PointMaze_Large-v3', continuing_task=False, maze_map=GIANT_MAZE)
    # combines current goal progress with rest of observation for new observatons, remove current goal from obs
    env = PointMazeNotGoalDirectedWrapper(env, args.gridworld_on_step, args.on_step_reward_value)
    eval_callback = EvalCallback(env, log_path=args.log_dir, eval_freq=2000,
                        deterministic=True, render=False, verbose=0)
    if args.activate_shaping:
        env = PointMazeShapingWrapper(env, args.gamma, args.exponential, args.base, args.potential_scaling, args.potential_shift, args.reset_to_bias, args.reset_value)

    if args.agent == 'sac':
        agent = SAC("MlpPolicy", env, gamma=args.gamma, tensorboard_log=args.log_dir, device=args.device, learning_starts=1000, learning_rate=args.learning_rate, )#, buffer_size=50000), learning_starts=1000
    elif args.agent == 'td3':
        agent = TD3("MlpPolicy", env, gamma=args.gamma, tensorboard_log=args.log_dir, device=args.device, learning_rate=args.learning_rate, learning_starts=10000) #, replay_buffer_kwargs={"handle_timeout_termination":False})
    else:
        raise ValueError("Unsupported agent for pointmaze-giant: " + args.agent)

else:
    raise "Environment name not recognized: " + args.env

callbacks = eval_callback

# unique logging directory to avoid concurrency issues when running multiple experiments in parallel on a cluster
#job_id = os.environ.get("SLURM_JOB_ID", "default_local")

if isinstance(agent, Q_Learning):
    agent.learn(total_timesteps=max_steps, callback=callbacks)
else:
    agent.learn(total_timesteps=max_steps, callback=callbacks, tb_log_name=job_id)
