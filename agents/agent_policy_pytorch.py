import math
import torch
import numpy as np
import torch.nn as nn
import torch.jit as jit
from torch import Tensor
import torch.nn.functional as F
from typing import Any, List, Tuple


class Linear(nn.Linear):
    
    def __init__(self, in_features: int, out_features: int, init_scale=1, init_bias=0.0, bias: bool = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        nn.init.orthogonal_(self.weight, gain=init_scale)
        self.bias.data.fill_(init_bias)


class RunningMeanStd(jit.ScriptModule):
    
    def __init__(self, shape=(), epsilon=1e-2):
        super().__init__()
        self.sum = nn.Parameter(torch.zeros(shape))
        self.sumsq = nn.Parameter(torch.ones(shape)*epsilon)
        self.count = nn.Parameter(torch.ones(())*epsilon)
    
    @jit.script_method
    def forward(self) -> Tuple[Tensor, Tensor]:
        mean = self.sum / self.count
        var_est = self.sumsq / self.count - torch.square(mean)
        std = torch.sqrt(torch.clamp_min(var_est, 1e-2))
        return std, mean


class Critic(jit.ScriptModule):

    def __init__(self, normalize: bool):
        super().__init__()
        self.ret_rms = RunningMeanStd()
        self.normalize = normalize

    @jit.script_method
    def _normalize_reward(self, value: Tensor) -> Tensor:
        if self.normalize:
            ret_std, ret_mean = self.ret_rms()
            value = value * ret_std + ret_mean
        return value


class CriticMLP(Critic):

    def __init__(self, ob_dim, normalize=False, hiddens=None):
        super().__init__(normalize)
        self.net = dense_net(ob_dim, 1, hiddens)

    @jit.script_method
    def forward(self, obs: Tensor) -> Tensor:
        value_flat = self.net(obs)
        value_flat = self._normalize_reward(value_flat)
        value_flat = torch.squeeze(value_flat, -1)
        return value_flat


class CriticLSTM(Critic):

    def __init__(self, ob_dim, num_lstm, n_envs, n_steps, normalize=False, hiddens=None):
        super().__init__(normalize)
        self.net = LSTMLayer(ob_dim, 1, num_lstm, n_envs, n_steps, hiddens)

    @jit.script_method
    def forward(self, obs: Tensor, states: Tensor, masks: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        value_flat, states = self.net(obs, states, masks)
        value_flat = self._normalize_reward(value_flat)
        value_flat = torch.squeeze(value_flat, -1)
        value_flat = torch.squeeze(value_flat, 0)
        return value_flat, states


class Actor(jit.ScriptModule):

    def __init__(self, ac_dim: int):
        super().__init__()
        self.logstd = nn.Parameter(torch.zeros((ac_dim, 1), requires_grad=True))
        self.ff_out: Tensor = torch.tensor(0.0)

    def sample(self, means: Tensor, deterministic: bool) -> Tensor:
        if not deterministic:
            eps = torch.empty(means.shape).normal_().to(means.device)
            std = self.logstd.T.exp()
            actions = means + eps * std
        else:
            actions = means
        return actions
    
    @jit.script_method
    def neglogp(self, actions: Tensor, means: Tensor) -> Tensor:
        std = self.logstd.T.exp() ** 2
        neglogp = ((actions - means) ** 2) / (2 * std) + self.logstd.T + math.log(math.sqrt(2 * math.pi))
        return neglogp.sum(-1)
    
    @jit.script_method
    def entropy(self) -> Tensor:
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + self.logstd.squeeze(-1)
        return entropy.sum(-1).mean()


class ActorMLP(Actor):

    def __init__(self, ob_dim, ac_dim, hiddens=None):
        super().__init__(ac_dim)
        self.net = dense_net(ob_dim, ac_dim, hiddens)
        self.ac_dim = ac_dim
    
    @jit.script_method
    def forward(self, obs: Tensor, deterministic: bool) -> Tuple[Tensor, Tensor]:
        means = self.net(obs)
        if self.net.save_ff_out:
            self.ff_out = self.net.ff_out[:-self.ac_dim]
        actions = self.sample(means, deterministic)
        neglogps = self.neglogp(actions, means)
        return actions, neglogps
    
    def set_ff_out(self, value=True):
        self.net.save_ff_out = value
    

class ActorLSTM(Actor):

    def __init__(self, ob_dim, ac_dim, num_lstm, n_envs, n_steps, hiddens):
        super().__init__(ac_dim)
        self.net = LSTMLayer(ob_dim, ac_dim, num_lstm, n_envs, n_steps, hiddens)
    
    @jit.script_method
    def forward(self, obs: Tensor, states: Tensor, masks: Tensor, deterministic: bool) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
        means, states = self.net(obs, states, masks)
        if self.net.fc_in.save_ff_out:
            self.ff_out = self.net.fc_in.ff_out
        means = torch.reshape(means, (-1, means.shape[-1]))
        actions = self.sample(means, deterministic)
        neglogps = self.neglogp(actions, means)
        return actions, neglogps, states
    
    def set_ff_out(self, value=True):
        self.net.fc_in.save_ff_out = value
    

class PolicyValue(jit.ScriptModule):

    def __init__(self, ob_dim, ac_dim, normalize=True):
        super().__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.device = "cpu"
        self.initial_state = None
        self.ob_rms = RunningMeanStd((self.ob_dim,))
        self.use_norm_ob = True if normalize else False
        self.use_norm_ret = True if normalize and normalize != "ob" else False
        self.is_mlp = True
        self.clip_obs = 5.0

    def to(self, device):
        self.device = device
        return super().to(device)

    @jit.script_method
    def _normalize_obs(self, obs: Tensor) -> Tensor:
        if self.use_norm_ob:
            ob_std, ob_mean = self.ob_rms()
            obs = torch.clamp((obs - ob_mean)/ob_std, -self.clip_obs, self.clip_obs)
        return obs

    def flatten(self, tensor):
        return tensor.squeeze(0).cpu().detach().numpy()
    
    def convert_to_tensor(self, data: Any) -> Tensor:
        if data is not None:
            if not torch.is_tensor(data):
                data = torch.FloatTensor(data)
            data = data.to(self.device)
        return data

    @jit.script_method
    def compute_value_loss(self, vpred, old_values, returns, cliprange) -> Tensor:
        vpred_clipped = old_values + torch.clamp(vpred-old_values, -cliprange, cliprange)
        vf_losses1 = torch.square(vpred - returns)
        vf_losses2 = torch.square(vpred_clipped - returns)
        vf_loss = 0.5 * torch.maximum(vf_losses1, vf_losses2).mean()
        return vf_loss
    
    @jit.script_method
    def compute_pg_loss(self, advs, old_neglogpacs, neglogps, cliprange) -> Tuple[Tensor, Tensor, Tensor]:
        ratio = torch.exp(old_neglogpacs - neglogps)
        pg_losses = - advs * ratio
        pg_losses2 = - advs * torch.clamp(ratio, 1.0-cliprange, 1.0+cliprange)
        pg_loss = torch.maximum(pg_losses, pg_losses2)
        pg_loss = pg_loss.mean()
        with torch.no_grad():
            approxkl = 0.5 * torch.square(torch.log(ratio).abs()).mean()
            clipfrac = torch.greater(torch.abs(ratio - 1.0), cliprange).float().mean()
        return pg_loss, approxkl, clipfrac


class MlpPolicyValue(PolicyValue):

    def __init__(self, ob_dim, ac_dim, *, hiddens=None, normalize=True, **kwargs):
        super().__init__(ob_dim, ac_dim, normalize)
        if hiddens is None:
            hiddens = [64, 64]
        self.actor = ActorMLP(self.ob_dim, self.ac_dim, hiddens)
        self.critic = CriticMLP(self.ob_dim, self.use_norm_ret, hiddens)

    @jit.script_method
    def forward(self, obs: Tensor, deterministic: bool) -> Tuple[Tensor, Tensor, Tensor]:
        obs = self._normalize_obs(obs)
        values = self.critic(obs)
        actions, neglogps = self.actor(obs, deterministic)
        return actions, values, neglogps

    def step(self, obs, states=None, masks=None, deterministic=False):
        with torch.no_grad():
            obs = self.convert_to_tensor(np.array(obs))
            actions, values, neglogpacs = self.forward(obs, deterministic)
            actions = actions.cpu().detach().numpy()
            values = values.cpu().detach().numpy()
            neglogpacs = neglogpacs.cpu().detach().numpy()
        return actions, values, None, neglogpacs
    
    @jit.script_method
    def compute_loss(self, obs, returns, actions, old_values, old_neglogpacs, advs, cliprange, ent_coef: float, vf_coef: float):
        obs = self._normalize_obs(obs)
        vpred = self.critic(obs)
        means = self.actor.net(obs)
        neglogps = self.actor.neglogp(actions, means)
        entropy = self.actor.entropy()
        vf_loss = self.compute_value_loss(vpred, old_values, returns, cliprange)
        pg_loss, approxkl, clipfrac = self.compute_pg_loss(advs, old_neglogpacs, neglogps, cliprange)
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        loss.backward()
        return loss, pg_loss, entropy, vf_loss, approxkl, clipfrac

    @jit.script_method
    def compute_vf_loss(self, obs, returns, old_values, cliprange, vf_coef: float):
        obs = self._normalize_obs(obs)
        vpred = self.critic(obs)
        vf_loss = self.compute_value_loss(vpred, old_values, returns, cliprange)
        loss = vf_loss * vf_coef
        loss.backward()
        return vf_loss
    

class LSTMLayer(jit.ScriptModule):

    def __init__(self, in_dim: int, out_dim: int, h_dim: int, n_envs: int, n_steps: int, hiddens: List):
        super().__init__()
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.fc_in = dense_net(in_dim, h_dim, hiddens[:-2])
        self.rnn_cell = nn.LSTMCell(h_dim, h_dim)
        self.fc_out = dense_net(h_dim, out_dim, hiddens[:-2])
    
    @jit.script_method
    def forward(self, obs: Tensor, state: Tensor, mask: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        ff_out = F.relu(self.fc_in(obs))
        input_seq = torch.reshape(ff_out, (self.n_envs, self.n_steps, -1))
        input_seq = torch.permute(input_seq, (1, 0, 2))
        masks = torch.reshape(mask, (self.n_envs, self.n_steps, 1))
        c, h = state[0], state[1]
        output = []
        for i in range(input_seq.shape[0]):
            h = h * (1 - masks[:,i,:])
            c = c * (1 - masks[:,i,:])
            h, c = self.rnn_cell(input_seq[i], (h, c))
            output.append(h)
        output = torch.stack(output)
        output = self.fc_out(output)
        return output, (c, h)


class LSTMPolicy(PolicyValue):

    def __init__(self, ob_dim, ac_dim, n_envs=1, n_steps=1, *, hiddens=None, normalize=False, **kwargs):
        super().__init__(ob_dim, ac_dim, normalize)
        if hiddens is None:
            hiddens = [128, 128]
        num_lstm = hiddens[-1]
        self.is_mlp = False
        self._n_envs = n_envs
        self._n_steps = n_steps
        self._zero_state = np.zeros((4, num_lstm), dtype=np.float32)
        self.initial_state = np.tile(self._zero_state, (n_envs, 1, 1))
        self.actor = ActorLSTM(self.ob_dim, self.ac_dim, num_lstm, n_envs, n_steps, hiddens)
        self.critic = CriticLSTM(self.ob_dim, num_lstm, n_envs, n_steps, self.use_norm_ret, hiddens)
    
    def setup_env(self, n_envs=None, n_steps=None):
        if n_envs is None: n_envs = self._n_envs
        if n_steps is None: n_steps = self._n_steps
        self.initial_state = np.tile(self._zero_state, (n_envs, 1, 1))
        self.actor.net.n_steps = n_steps
        self.actor.net.n_envs = n_envs
        self.critic.net.n_steps = n_steps
        self.critic.net.n_envs = n_envs
        
    @jit.script_method
    def forward(self, obs: Tensor, states: Tensor, masks: Tensor, deterministic: bool) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        obs = self._normalize_obs(obs)
        states = torch.permute(states, (1, 0, 2))
        values, v_states = self.critic(obs, states[:2], masks)
        actions, neglogps, p_states = self.actor(obs, states[2:], masks, deterministic)
        with torch.no_grad():
            states = torch.stack(v_states + p_states)
            states = torch.permute(states, (1, 0, 2))
        return actions, values, neglogps, states.detach()

    def step(self, obs, states, masks, deterministic=False):
        with torch.no_grad():
            obs = self.convert_to_tensor(np.array(obs))
            states = self.convert_to_tensor(states)
            masks = self.convert_to_tensor(masks)
            actions, values, neglogpacs, states = self.forward(obs, states, masks, deterministic)
            actions = actions.cpu().detach().numpy()
            values = values.cpu().detach().numpy()
            neglogpacs = neglogpacs.cpu().detach().numpy()
            states = states.cpu().detach().numpy()
        return actions, values, states, neglogpacs
    
    @jit.script_method
    def compute_loss(self, obs, returns, masks, actions, old_values, old_neglogpacs, advs, states, cliprange, ent_coef: float, vf_coef: float):
        obs = self._normalize_obs(obs)
        states = torch.permute(states, (1, 0, 2))
        vpred, v_states = self.critic(obs, states[:2], masks)
        means, p_states = self.actor.net(obs, states[2:], masks)
        means = torch.permute(means, (1, 0, 2))
        neglogps = self.actor.neglogp(actions, means)
        entropy = self.actor.entropy()
        with torch.no_grad():
            states = torch.stack(v_states + p_states)
            states = torch.permute(states, (1, 0, 2))
        vf_loss = self.compute_value_loss(vpred.T, old_values, returns, cliprange)
        pg_loss, approxkl, clipfrac = self.compute_pg_loss(advs, old_neglogpacs, neglogps, cliprange)
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        loss.backward()
        return loss, pg_loss, entropy, vf_loss, approxkl, clipfrac, states.detach()

    @jit.script_method
    def compute_vf_loss(self, obs, returns, masks, old_values, states, cliprange, vf_coef: float):
        obs = self._normalize_obs(obs)
        states = torch.permute(states, (1, 0, 2))
        vpred, v_states = self.critic(obs, states[:2], masks)
        with torch.no_grad():
            states = torch.stack(v_states + (states[2], states[3]))
            states = torch.permute(states, (1, 0, 2))
        vf_loss = self.compute_value_loss(vpred, old_values, returns, cliprange)
        loss = vf_loss * vf_coef
        loss.backward()
        return vf_loss, states.detach()


class Sequential(nn.Sequential):

    def __init__(self, *layers) -> None:
        super().__init__(*layers)
        self.ff_out = torch.tensor(0.0)
        self.linear_pos = [isinstance(l, nn.Linear) for l in layers]
        self.save_ff_out = False
    
    def forward(self, input):
        ff_out = []
        k = 0
        for module in self:
            input = module(input)
            if self.linear_pos[k] and self.save_ff_out:
                ff_out.append(input)
            k += 1
        if len(ff_out) > 0:
            self.ff_out = torch.cat(ff_out)
        return input


def dense_net(in_dim, out_dim, hiddens=[], activation=None):
    layers = []
    for h_dim in hiddens:
        layers.append(Linear(in_dim, h_dim))
        layers.append(nn.Tanh())
        in_dim = h_dim
    layers.append(Linear(in_dim, out_dim, 0.01))
    if activation is not None:
        layers.append(activation())
    return Sequential(*layers)


def load_policy(ob_dim, ac_dim, n_envs=None, n_steps=1, zoo_path=None, normalize="ob", use_lstm=True):
    if zoo_path is not None:
        use_lstm = "KickAndDefend" in zoo_path or "Sumo" in zoo_path
    if use_lstm:
        policy = LSTMPolicy(ob_dim, ac_dim, n_envs, n_steps, normalize=normalize)
    else:
        policy = MlpPolicyValue(ob_dim, ac_dim, normalize=normalize)
    if zoo_path is not None:
        policy.load_state_dict(torch.load(f"agents/{zoo_path}"))
        print(f"Loaded params from: {zoo_path}")
    return policy