import time
import torch
import numpy as np
import torch.nn as nn
import torch.jit as jit
from torch.optim import AdamW
import torch.nn.functional as F
from agents.agent_policy_pytorch import MlpPolicyValue, dense_net
from trainer.utils import explained_variance, get_schedule_fn, read_ep_infos
from torch.utils.tensorboard import SummaryWriter


class PPO2:

    def __init__(self, policy: MlpPolicyValue, n_envs, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.2, device="cpu",
                 max_grad_norm=0.5, lam=0.95, nminibatches=4, n_timesteps=20e6, use_imitation=False, use_l2t=False,
                 noptepochs=4, cliprange=0.2, lr_schedule='const', log_dir="logs", **kwargs):
        self.n_steps = n_steps
        self.gamma = gamma
        self.lam = lam
        self.n_envs = n_envs
        self.nminibatches = nminibatches
        self.n_timesteps = n_timesteps
        self.noptepochs = noptepochs
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.n_batch = self.n_envs * self.n_steps
        self.num_timesteps = 0
        self.num_trainsteps = 0
        self.policy = policy
        self.device = device
        self.use_imitation = use_imitation
        self.use_l2t = use_l2t

        self.lr = learning_rate
        self.lr_schedule = lr_schedule
        self.learning_rate = get_schedule_fn(self.lr, schedule=lr_schedule)
        self.cliprange = get_schedule_fn(cliprange, schedule='const')
        if self.policy.training:
            self.writer = SummaryWriter(log_dir)
        self.policy.to(self.device)
        self.optim_policy = AdamW(self.policy.parameters(), lr=self.lr, eps=1e-5)

    def _convert_to_tensor(self, data):
        if data is not None:
            if not torch.is_tensor(data):
                data = torch.FloatTensor(data)
            data = data.to(self.device)
        return data
    
    def _compute_ppo_loss(self, policy: MlpPolicyValue, obs, returns, masks, actions, old_values, old_neglogpacs, advs, cliprange, states=None):
        if policy.is_mlp:
            losses = policy.compute_loss(obs, returns, actions, old_values, old_neglogpacs, advs, cliprange, self.ent_coef, self.vf_coef)
            return losses + (None,)
        else:
            return policy.compute_loss(obs, returns, masks, actions, old_values, old_neglogpacs, advs, states, cliprange, self.ent_coef, self.vf_coef)
    
    def _compute_advantage(self, returns, values):
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        return advs

    def _write_loss_logs(self, loss, pg_loss, entropy, vf_loss, approxkl, clipfrac, others={}):
        self.writer.add_scalar("losses/total_loss", loss, self.num_trainsteps)
        self.writer.add_scalar("losses/pg_loss", pg_loss, self.num_trainsteps)
        self.writer.add_scalar("losses/entropy", entropy, self.num_trainsteps)
        self.writer.add_scalar("losses/vf_loss", vf_loss, self.num_trainsteps)
        self.writer.add_scalar("losses/approxkl", approxkl, self.num_trainsteps)
        self.writer.add_scalar("losses/clipfrac", clipfrac, self.num_trainsteps)
        for name, value in others.items():
            self.writer.add_scalar(f"losses/{name}", value, self.num_trainsteps)
        self.num_trainsteps += 1
    
    def _update_ppo(self, policy: MlpPolicyValue, optim: AdamW, obs, returns, masks, actions, values, neglogpacs, advs, lr, cliprange, states=None):
        obs, returns, masks, actions, values, neglogpacs, advs = map(self._convert_to_tensor, (obs, returns, masks, actions, values, neglogpacs, advs))
        for g in optim.param_groups:
            g['lr'] = lr
        optim.zero_grad()
        loss, pg_loss, entropy, vf_loss, approxkl, clipfrac, states = self._compute_ppo_loss(policy, obs, returns, masks, actions, values, neglogpacs, advs, cliprange, states)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
        optim.step()
        if not policy.is_mlp:
            states = states.detach()
        lossvals = [loss, pg_loss, entropy, vf_loss, approxkl, clipfrac]
        return lossvals, states
    
    def _train_step(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
        advs = self._compute_advantage(returns, values)
        lossvals, states = self._update_ppo(self.policy, self.optim_policy, obs, returns, masks, actions, values, neglogpacs, advs, lr, cliprange, states)
        lossvals = [_loss.item() for _loss in lossvals]
        return lossvals, states
    
    def schedule_value(self, schedule, update, nupdates, mode='const'):
        if mode == 'const':
            value = schedule(0)
        elif mode == 'linear':
            value = schedule(update, nupdates)
        elif mode == 'step':
            value = schedule(update)
        return value
    
    def update(self, update, data, states):
        nupdates = self.n_timesteps // self.n_batch
        lr_now = self.schedule_value(self.learning_rate, update, nupdates, self.lr_schedule)
        cliprangenow = self.cliprange(0)
        self.writer.add_scalar("settings/learning_rate", lr_now, self.num_trainsteps)
        self.writer.add_scalar("settings/cliprange", cliprangenow, self.num_trainsteps)
        lr_now = torch.tensor(lr_now).to(self.device)
        cliprangenow = torch.tensor(cliprangenow).to(self.device)
        loss_vals = []
        if self.policy.is_mlp:
            batch_size = self.n_steps // self.nminibatches
            inds = np.arange(self.n_batch)
            for _ in range(self.noptepochs):
                np.random.shuffle(inds)
                for start in range(0, self.n_batch, batch_size):
                    end = start + batch_size
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in data)
                    loss_vals.append(self._train_step(lr_now, cliprangenow, *slices)[0])
        else:
            envsperbatch = self.n_envs // self.nminibatches
            envinds = np.arange(self.n_envs)
            flatinds = np.arange(self.n_envs * self.n_steps).reshape(self.n_envs, self.n_steps)
            
            if isinstance(states, list):
                states = [self._convert_to_tensor(s) for s in states]
            else:
                states = self._convert_to_tensor(states)
            
            for _ in range(self.noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, self.n_envs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds]
                    if isinstance(states, list):
                        mb_states = [_states[mbenvinds] for _states in states]
                    else:
                        mb_states = states[mbenvinds]
                    slices = (arr[mbflatinds] for arr in data)
                    losses, mb_states = self._train_step(lr_now, cliprangenow, *slices, states=mb_states)
                    loss_vals.append(losses)
        return loss_vals
    
    def write_log(self, update, loss_vals, fps, explained_var, ep_infos, true_reward):
        self.num_timesteps += len(ep_infos)
        global_step = self.num_timesteps + self.num_trainsteps
        self._write_loss_logs(*loss_vals)
        self.writer.add_scalar("status/serial_timesteps", update * self.n_steps, global_step)
        self.writer.add_scalar("status/nupdates", update, global_step)
        self.writer.add_scalar("status/total_timesteps", global_step, global_step)
        self.writer.add_scalar("status/fps", fps, global_step)
        self.writer.add_scalar("rewards/adv_reward", np.mean(true_reward), global_step)
        self.writer.add_scalar("rewards/explained_variance", float(explained_var), global_step)
        
        rewards, n_games, rewards_ori, n_games_ori = read_ep_infos(ep_infos)
        keys = ["win", "lose", "tie"]
        if n_games > 0:
            for key, value in zip(keys, rewards):
                self.writer.add_scalar(f'game/{key}', value, global_step)
            self.writer.add_scalar(f'game/count', n_games, global_step)
        if n_games > 0:
            for key, value in zip(keys, rewards_ori):
                self.writer.add_scalar(f'game_ori/{key}', value, global_step)
            self.writer.add_scalar(f'game_ori/count', n_games_ori, global_step)
        
        other_losses = None
        if isinstance(loss_vals[-1], dict):
            other_losses = [f"{v:.3f}" for v in loss_vals[-1].values()]
            other_losses = other_losses[:2]
            other_losses = ", ".join(other_losses)
        loss_vals = ", ".join(f"{x:.3f}" for x in loss_vals[1:4])
        rewards = ", ".join(f"{np.mean(x):.4f}" for x in rewards) if len(rewards) > 0 else ""
        if other_losses:
            loss_vals = f"{loss_vals} - {other_losses}"
        print(f"Step: {global_step} - lossvals: [{loss_vals}] - rewards: [{rewards}] - fps: {fps}")

    def train(self, update, t_start, data, *kwargs):
        self.policy.train()
        batch_size = self.n_batch // self.nminibatches
        envs_per_batch = batch_size // self.n_steps
        if not self.policy.is_mlp:
            self.policy.setup_env(n_envs=envs_per_batch, n_steps=self.n_steps)
        obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = data
        loss_vals = self.update(update, [obs, returns, masks, actions, values, neglogpacs], states)
        loss_vals = np.mean(loss_vals, axis=0)
        loss_vals = list(loss_vals)
        explained_var = explained_variance(values, returns)
        t_now = time.time()
        fps = int(self.n_batch / (t_now - t_start))
        self.write_log(update, loss_vals, fps, explained_var, ep_infos, true_reward)
        if not self.policy.is_mlp:
            self.policy.setup_env()
        global_step = self.num_timesteps + self.num_trainsteps
        return global_step < self.n_timesteps


class OppPPO2(PPO2):

    def __init__(self, adv_policy: MlpPolicyValue, opp_policy: MlpPolicyValue, **kwargs):
        super().__init__(adv_policy, **kwargs)
        self.adv_policy = self.policy
        self.opp_policy = opp_policy
        self.adv_optim = self.optim_policy
        self.opp_policy.to(self.device)
        self.opp_optim = AdamW(self.opp_policy.parameters(), lr=self.lr, eps=1e-5)
    
    def _compute_vf_loss(self, policy: MlpPolicyValue, obs, old_values, returns, masks, states, cliprange):
        if policy.is_mlp:
            vf_loss = policy.compute_vf_loss(obs, returns, old_values, cliprange, self.vf_coef)
            return vf_loss, None
        else:
            return policy.compute_vf_loss(obs, returns, masks, old_values, states, cliprange, self.vf_coef)

    def _update_vf(self, policy: MlpPolicyValue, optim: AdamW, obs, values, returns, masks, lr, cliprange, states):
        obs, values, returns, masks, states = map(self._convert_to_tensor, (obs, values, returns, masks, states))
        for g in optim.param_groups:
            g['lr'] = lr
        optim.zero_grad()
        vf_loss, states = self._compute_vf_loss(policy, obs, values, returns, masks, states, cliprange)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
        optim.step()
        if not policy.is_mlp:
            states = states.detach()
        return [vf_loss], states

    def _train_step(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, opp_obs, opp_returns, opp_values, states=None):
        adv_states, opp_states = states if states else (None, None)
        
        advs = self._compute_advantage(returns, values)
        opp_advs = self._compute_advantage(opp_returns, opp_values)
        advs = advs - opp_advs

        adv_lossvals, adv_states = self._update_ppo(self.adv_policy, self.adv_optim, obs, returns, masks, actions, values, neglogpacs, advs, lr, cliprange, adv_states)
        opp_lossvals, opp_states = self._update_vf(self.opp_policy, self.opp_optim, opp_obs, opp_values, opp_returns, masks, lr, cliprange, opp_states)
        
        lossvals = adv_lossvals + opp_lossvals
        lossvals = [_loss.item() for _loss in lossvals]
        states = [adv_states, opp_states]
        return lossvals, states

    def train(self, update, t_start, adv_data, opp_data, *args):
        self.adv_policy.train()
        self.opp_policy.train()
        batch_size = self.n_batch // self.nminibatches
        envs_per_batch = batch_size // self.n_steps
        if not self.adv_policy.is_mlp:
            self.adv_policy.setup_env(n_envs=envs_per_batch, n_steps=self.n_steps)
        if not self.opp_policy.is_mlp:
            self.opp_policy.setup_env(n_envs=envs_per_batch, n_steps=self.n_steps)
        obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = adv_data
        opp_obs, opp_returns, _, _, opp_values, _, opp_states, _, _ = opp_data
        loss_vals = self.update(update, [obs, returns, masks, actions, values, neglogpacs, opp_obs, opp_returns, opp_values], [states, opp_states])
        loss_vals = np.mean(loss_vals, axis=0)
        loss_vals = list(loss_vals)[:6] + [{
            "opp_vf_loss": loss_vals[6],
        }]
        explained_var = explained_variance(values, returns)
        t_now = time.time()
        fps = int(self.n_batch / (t_now - t_start))
        self.write_log(update, loss_vals, fps, explained_var, ep_infos, true_reward)
        if not self.adv_policy.is_mlp:
            self.adv_policy.setup_env()
        if not self.opp_policy.is_mlp:
            self.opp_policy.setup_env()
        global_step = self.num_timesteps + self.num_trainsteps
        return global_step < self.n_timesteps


class Discriminator(jit.ScriptModule):

    def __init__(self, in_dim, out_dim, hiddens=[128, 128]):
        super().__init__()
        self.net = dense_net(in_dim, out_dim, hiddens, nn.Tanh)
        self.device = "cpu"
    
    def to(self, device):
        self.device = device
        return super().to(device)
    
    @jit.script_method
    def forward(self, obs, actions):
        return self.net(torch.cat([obs, actions], dim=-1)).squeeze(-1)
    
    @jit.script_method
    def _calculate_reward(self, obs, actions):
        with torch.no_grad():
            return -F.logsigmoid(-self.forward(obs, actions))
    
    def calculate_reward(self, obs, actions):
        obs = torch.FloatTensor(np.array(obs)).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = self._calculate_reward(obs, actions)
        rewards = rewards.detach().cpu().numpy()
        return rewards
    
    @jit.script_method
    def compute_loss(self, obs, actions, exp_actions):
        logits_pi = self.forward(obs, actions)
        logits_exp = self.forward(obs, exp_actions)
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss = loss_pi + loss_exp
        loss.backward()
        with torch.no_grad():
            acc_pi = (logits_pi < 0).float().mean()
            acc_exp = (logits_exp > 0).float().mean()
        return loss, acc_pi, acc_exp


class ImitationPPO2(OppPPO2):

    def __init__(self, adv_policy: MlpPolicyValue, opp_policy: MlpPolicyValue, use_opp=False, **kwargs):
        super().__init__(adv_policy, opp_policy, **kwargs)
        self.disc = Discriminator(opp_policy.ob_dim + opp_policy.ac_dim, 1)
        self.disc.to(self.device)
        self.disc_optim = AdamW(self.disc.parameters(), lr=self.lr, eps=1e-5)
        self.use_opp = use_opp
    
    def _update_disc(self, disc: Discriminator, optim: AdamW, obs, actions, exp_actions, lr):
        obs, actions, exp_actions = map(self._convert_to_tensor, (obs, actions, exp_actions))
        for g in optim.param_groups:
            g['lr'] = lr
        optim.zero_grad()
        disc_loss, acc_pi, acc_exp = disc.compute_loss(obs, actions, exp_actions)
        lossvals = [disc_loss, acc_pi, acc_exp]
        optim.step()
        return lossvals

    def _train_step(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, opp_obs, opp_returns, opp_values, opp_neglogpacs, opp_actions, exp_actions, states=None):
        adv_states, opp_states = states if states else (None, None)
        
        advs = self._compute_advantage(returns, values)
        opp_advs = self._compute_advantage(opp_returns, opp_values)
        if self.use_l2t: advs = - advs

        adv_lossvals, adv_states = self._update_ppo(self.adv_policy, self.adv_optim, obs, returns, masks, actions, values, neglogpacs, advs, lr, cliprange, adv_states)
        opp_lossvals, opp_states = self._update_ppo(self.opp_policy, self.opp_optim, opp_obs, opp_returns, masks, opp_actions, opp_values, opp_neglogpacs, opp_advs, lr, cliprange, opp_states)
        disc_lossvals = self._update_disc(self.disc, self.disc_optim, opp_obs, opp_actions, exp_actions, lr)
        
        lossvals = adv_lossvals + [opp_lossvals[3]] + disc_lossvals
        lossvals = [_loss.item() for _loss in lossvals]
        states = [adv_states, opp_states]
        return lossvals, states

    def train(self, update, t_start, adv_data, opp_data, vic_data, *args):
        self.adv_policy.train()
        self.opp_policy.train()
        self.disc.train()
        batch_size = self.n_batch // self.nminibatches
        envs_per_batch = batch_size // self.n_steps
        if not self.adv_policy.is_mlp:
            self.adv_policy.setup_env(n_envs=envs_per_batch, n_steps=self.n_steps)
        if not self.opp_policy.is_mlp:
            self.opp_policy.setup_env(n_envs=envs_per_batch, n_steps=self.n_steps)
        obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = adv_data
        opp_obs, opp_returns, _, opp_actions, opp_values, opp_neglogpacs, opp_states, _, _ = opp_data
        exp_actions = vic_data[3]
        loss_vals = self.update(update, [obs, returns, masks, actions, values, neglogpacs, opp_obs, opp_returns, opp_values, opp_neglogpacs, opp_actions, exp_actions], [states, opp_states])
        loss_vals = np.mean(loss_vals, axis=0)
        loss_vals = list(loss_vals)[:6] + [{
            "opp_vf_loss": loss_vals[6],
            "disc_loss": loss_vals[7],
            "acc_pi": loss_vals[8],
            "acc_exp": loss_vals[9],
        }]
        explained_var = explained_variance(values, returns)
        t_now = time.time()
        fps = int(self.n_batch / (t_now - t_start))
        self.write_log(update, loss_vals, fps, explained_var, ep_infos, true_reward)
        if not self.adv_policy.is_mlp:
            self.adv_policy.setup_env()
        if not self.opp_policy.is_mlp:
            self.opp_policy.setup_env()
        global_step = self.num_timesteps + self.num_trainsteps
        return global_step < self.n_timesteps