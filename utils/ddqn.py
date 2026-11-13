import types
import torch as th


def double_dqn_train(self, gradient_steps: int, batch_size: int = 100) -> None:
    self.policy.set_training_mode(True)
    self._update_learning_rate(self.policy.optimizer)
    losses = []

    for _ in range(gradient_steps):
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

        with th.no_grad():
            # Double DQN: select actions with online net, evaluate with target net.
            next_q_online = self.q_net(replay_data.next_observations)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.q_net_target(replay_data.next_observations)
            next_q_values = th.gather(next_q_target, dim=1, index=next_actions).squeeze(1)
            target_q_values = replay_data.rewards.flatten() + (1 - replay_data.dones.flatten()) * self.gamma * next_q_values

        current_q_values = self.q_net(replay_data.observations)
        current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long()).squeeze(1)

        loss = th.nn.functional.smooth_l1_loss(current_q_values, target_q_values)
        losses.append(loss.item())

        self.policy.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

    self._n_updates += gradient_steps
    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    self.logger.record("train/loss", sum(losses) / len(losses))


def attach_double_dqn_train(model):
    model.train = types.MethodType(double_dqn_train, model)
    return model
