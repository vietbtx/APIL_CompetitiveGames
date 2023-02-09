import numpy as np
from trainer.utils import swap_and_flatten


class Buffer:

    def __init__(self, states, n_steps, gamma, lam):
        self.obs, self.rewards, self.actions, self.values, self.dones, self.neglogpacs = [], [], [], [], [], []
        self.states = states
        self.infos = []
        self.n_steps = n_steps
        self.gamma = gamma
        self.lam = lam
        self.last_actions = None
        self.last_values = None
        self.last_dones = None
    
    def add(self, obs, actions, values, neglogpacs, dones):
        self.add_value(obs, values, dones)
        self.actions.append(actions)
        self.neglogpacs.append(neglogpacs)
    
    def add_value(self, obs, values, dones):
        self.obs.append(np.array(obs))
        self.values.append(values)
        self.dones.append(np.array(dones))
    
    def add_info(self, rewards, dones, info):
        self.rewards.append(rewards)
        self.last_dones = np.array(dones)
        self.infos.append(info)
    
    def _convert_to_numpy(self):
        obs          = np.asarray(self.obs, dtype=np.float32)               # (n_steps, n_envs, observation_space)
        rewards      = np.asarray(self.rewards, dtype=np.float32)           # (n_steps, n_envs)
        values       = np.asarray(self.values, dtype=np.float32)            # (n_steps, n_envs)
        dones        = np.asarray(self.dones, dtype=np.bool)                # (n_steps, n_envs)
        advs         = np.zeros_like(self.rewards)
        true_reward  = np.copy(self.rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.last_dones
                nextvalues = self.last_values
            else:
                nextnonterminal = 1.0 - dones[step + 1]
                nextvalues = values[step + 1]
            delta = rewards[step] + self.gamma * nextvalues * nextnonterminal - values[step]
            advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        returns = advs + values
        obs, returns, dones, values, true_reward = map(swap_and_flatten, (obs, returns, dones, values, true_reward))
        infos = [i for info in self.infos for i in info]
        return obs, returns, dones, values, self.states, infos, true_reward

    def convert_to_numpy(self):
        obs, returns, dones, values, self.states, infos, true_reward = self._convert_to_numpy()
        actions             = np.asarray(self.actions)                           # (n_steps, n_envs, action_space)
        neglogpacs          = np.asarray(self.neglogpacs, dtype=np.float32)      # (n_steps, n_envs)
        actions, neglogpacs = map(swap_and_flatten, (actions, neglogpacs))
        return obs, returns, dones, actions, values, neglogpacs, self.states, infos, true_reward
