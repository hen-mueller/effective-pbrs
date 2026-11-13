import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class ValueStatisticsCallback(BaseCallback):
    """Log scalar statistics of value estimates for a fixed state grid.

    The callback is intentionally isolated so it can be removed later without
    changing the rest of the training loop.
    """

    def __init__(self, state_sampler, value_estimator, log_freq=1000, log_name="value_stats"):
        super().__init__()
        self.state_sampler = state_sampler
        self.value_estimator = value_estimator
        self.log_freq = log_freq
        self.log_name = log_name
        self.previous_values = None

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True

        states = self.state_sampler(self.model)
        values = self.value_estimator(self.model, states)

        if values.ndim != 1:
            values = values.reshape(-1)

        stats = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "max": float(np.max(values)),
            "min": float(np.min(values)),
            "q25": float(np.percentile(values, 25)),
            "q50": float(np.percentile(values, 50)),
            "q75": float(np.percentile(values, 75)),
        }

        if self.previous_values is not None:
            stats["mean_abs_change"] = float(np.mean(np.abs(values - self.previous_values)))

        for name, value in stats.items():
            self.logger.record_mean(f"{self.log_name}/{name}", value)

        self.logger.dump(self.num_timesteps)
        self.previous_values = values
        return True


def make_pointmaze_state_sampler(grid_size=11, velocity=0.0):
    """Create a fixed grid of states for the PointMaze environment."""

    def sampler(model):
        env = model.get_env()
        env = env.envs[0]

#        if hasattr(env, "unwrapped"):
        env = env.unwrapped

        #if hasattr(env, "maze"):
        maze = env.maze
        width = maze.map_width
        length = maze.map_length
        maze_size = maze.maze_size_scaling

        goal = env.goal.copy()
       
        xs = np.linspace(0, width, grid_size, endpoint=False)
        ys = np.linspace(0, length, grid_size, endpoint=False)

        # velocity bound in [-5, 5] for PointMaze
        velocities = np.linspace(-3, 3, 3)

        maze_map = maze.maze_map

        states = []
        for x in xs:
            for y in ys:
                for dx in velocities:
                    for dy in velocities:
                        row = int(np.floor(x))
                        col = int(np.floor(y))
                        if maze_map[row][col] == 1:
                            # wall cell, skip this state
                            continue
                        coords = maze.cell_rowcol_to_xy(np.array([x, y]))
                        (sim_x, sim_y) = coords
                        state = np.array([sim_x, sim_y, dx, dy], dtype=np.float32)
                        # match observation of NotGoalDirected wrapper, concatenate desired goal and observation
                        state = np.concatenate((goal, state))
                        states.append(state)

        return np.array(states, dtype=np.float32)

    return sampler


def make_q_value_estimator(use_critic=False):
    """Estimate values from the model for a batch of states.

    For SAC/TD3 the critic is the most direct quantity to inspect. The callback
    defaults to the critic's Q estimate when available; otherwise it falls back
    to the policy's action-value estimate.
    """

    def estimator(model, states):
        if use_critic:
            critic = getattr(model, "critic", None)
            if critic is None:
                critic = getattr(model, "q_net", None)
            if critic is not None:
                obs = states.astype(np.float32)
                values = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
                actions = np.stack(np.meshgrid(values, values, indexing="ij"), axis=-1).reshape(-1, 2)

                repeated_obs = np.repeat(obs, len(actions), axis=0)
                repeated_actions = np.tile(actions, (len(obs), 1))

                device = getattr(model, "device", None)
                repeated_obs = torch.as_tensor(repeated_obs, dtype=torch.float32)
                repeated_actions = torch.as_tensor(repeated_actions, dtype=torch.float32)
                if device is not None:
                    repeated_obs = repeated_obs.to(device)
                    repeated_actions = repeated_actions.to(device)

                with torch.no_grad():
                    q_values = critic(repeated_obs, repeated_actions)
                if isinstance(q_values, (tuple, list)):
                    q_values = q_values[0]
                if isinstance(q_values, torch.Tensor):
                    q_values = q_values.detach().cpu().numpy()
                return np.asarray(q_values, dtype=np.float32).reshape(-1)

        # not tested, probably not working
        #obs = states.astype(np.float32)
        #actions, _ = model.predict(obs, deterministic=True)
        #if actions.ndim == 2:
        #    actions = actions[:, 0]
        #return np.asarray(actions, dtype=np.float32).reshape(-1)
        raise NotImplementedError("Value estimation not implemented for this model type.")

    return estimator
