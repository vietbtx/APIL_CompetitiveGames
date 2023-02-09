import gym
import random
import numpy as np
from tqdm import tqdm
from trainer.buffer import Buffer
from env.env import VectorizedEnv
from trainer.ppo import Discriminator
from agents.agent_policy_pytorch import MlpPolicyValue, PolicyValue


class Agent:

    def __init__(self, policy: MlpPolicyValue, env: VectorizedEnv, n_steps, n_envs, gamma, lam, **kwargs) -> None:
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.gamma = gamma
        self.lam = lam
        self.policy = policy
        self.action_space = env.action_space
        self.states = None
    
    def reset(self, obs):
        self.obs = obs
        self.policy.eval()
        if self.states is None:
            self.states = self.policy.initial_state
        self.buffer = Buffer(self.states, self.n_steps, self.gamma, self.lam)
        self.dones = [False for _ in range(self.n_envs)]
    
    def clip_actions(self, actions):
        if isinstance(self.action_space, gym.spaces.Box):
            if actions.shape[-1] == self.action_space.low.shape[-1]:
                actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return actions

    def step(self, obs_features=None):
        if obs_features is not None:
            self.obs = np.concatenate((self.obs, obs_features), -1)
        actions, values, self.states, neglogpacs = self.policy.step(self.obs, self.states, self.dones)
        self.buffer.add(self.obs, actions, values, neglogpacs, self.dones)
        clipped_actions = self.clip_actions(actions)
        return clipped_actions

    def add(self, obs, reward, dones, info):
        self.obs = obs
        self.dones = dones
        self.buffer.add_info(reward, dones, info)

    def trajectories(self, obs_features=None):
        if obs_features is not None:
            self.obs = np.concatenate((self.obs, obs_features), -1)
        self.buffer.last_actions, self.buffer.last_values, _, _ = self.policy.step(self.obs, self.states, self.dones)
        return self.buffer.convert_to_numpy()


class Runner:

    def __init__(self, env: VectorizedEnv, vic_policy: PolicyValue, adv_policy: PolicyValue, opp_policy: PolicyValue, disc: Discriminator, n_steps, enhance_reward=False, use_imitation=False, **kwargs):
        self.env = env
        self.n_steps = n_steps
        self.enhance_reward = enhance_reward
        self.use_imitation = use_imitation
        self.vic_agent = Agent(vic_policy, env, n_steps, **kwargs)
        self.opp_agent = Agent(opp_policy, env, n_steps, **kwargs) if opp_policy else None
        self.adv_agent = Agent(adv_policy, env, n_steps, **kwargs)
        self.disc = disc
    
    def run(self, verbose=False):
        vic_obs, adv_obs = self.env.get_current_states()
        opp_obs = np.concatenate((vic_obs, adv_obs), -1) if self.use_imitation else adv_obs
        self.vic_agent.reset(vic_obs)
        self.opp_agent.reset(opp_obs) if self.opp_agent else None
        self.adv_agent.reset(adv_obs) 

        for s in tqdm(range(self.n_steps), "Collecting ...", leave=False, ncols=0, disable=not verbose):
            vic_actions = self.vic_agent.step()
            opp_actions = self.opp_agent.step() if self.opp_agent else None
            opp_features = np.concatenate((vic_obs, opp_actions), -1) if self.use_imitation and self.opp_agent else None
            adv_actions = self.adv_agent.step(opp_features)

            (vic_obs, adv_obs), (vic_reward, adv_reward), dones, (vic_info, adv_info) = self.env.step(vic_actions, adv_actions)
            opp_obs = np.concatenate((vic_obs, adv_obs), -1) if self.use_imitation else adv_obs

            if self.use_imitation and self.disc:
                opp_reward = self.disc.calculate_reward(opp_obs, opp_actions)
                if self.enhance_reward:
                    opp_reward = opp_reward - adv_reward
            else:
                opp_reward = vic_reward

            self.vic_agent.add(vic_obs, vic_reward, dones, vic_info)
            self.opp_agent.add(opp_obs, opp_reward, dones, adv_info) if self.opp_agent else None
            self.adv_agent.add(adv_obs, adv_reward, dones, adv_info)
            
        vic_data = self.vic_agent.trajectories() if self.use_imitation else None
        opp_data = self.opp_agent.trajectories() if self.opp_agent else None
        opp_features = np.concatenate((self.vic_agent.obs, self.opp_agent.buffer.last_actions), -1) if self.use_imitation and self.opp_agent else None
        adv_data = self.adv_agent.trajectories(opp_features)

        return vic_data, adv_data, opp_data


class MixingRunner:

    def __init__(self, env: VectorizedEnv, vic_policy: PolicyValue, adv_policy: PolicyValue, opp_policy: PolicyValue, raw_adv_policy: PolicyValue, n_envs, n_steps, mix_ratio=0.8, use_imitation=False, **kwargs):
        self.env = env
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.use_imitation = use_imitation
        self.vic_agent = Agent(vic_policy, env, n_steps, n_envs, **kwargs)
        self.opp_agent = Agent(opp_policy, env, n_steps, n_envs, **kwargs) if opp_policy else None
        self.adv_agent = Agent(adv_policy, env, n_steps, n_envs, **kwargs)
        self.raw_adv_agent = Agent(raw_adv_policy, env, n_steps, n_envs, **kwargs) if raw_adv_policy else None
        self.mix_ratio = mix_ratio
        self.use_mixing = [False if mix_ratio > 0 else True] * n_envs
    
    def setup_mixing(self, dones):
        for i, done in enumerate(dones):
            if done: self.use_mixing[i] = random.random() < self.mix_ratio
        
    def update_info(self, dones, info):
        for i, done in enumerate(dones):
            if done and self.use_mixing[i]:
                info[i]["use_mixing"] = True
    
    def run(self, verbose=False):
        vic_obs, adv_obs = self.env.get_current_states()
        opp_obs = np.concatenate((vic_obs, adv_obs), -1) if self.use_imitation else adv_obs
        self.vic_agent.reset(vic_obs)
        self.opp_agent.reset(opp_obs) if self.opp_agent else None
        self.adv_agent.reset(adv_obs)
        self.raw_adv_agent.reset(adv_obs) if self.raw_adv_agent else None

        for s in tqdm(range(self.n_steps), "Collecting ...", leave=False, ncols=0, disable=not verbose):
            vic_actions = self.vic_agent.step()
            opp_actions = self.opp_agent.step() if self.opp_agent else None
            opp_features = np.concatenate((vic_obs, opp_actions), -1) if self.use_imitation and self.opp_agent else None
            adv_actions = self.adv_agent.step(opp_features)

            if self.raw_adv_agent:
                raw_adv_actions = self.raw_adv_agent.step()
                for i in range(self.n_envs):
                    if self.use_mixing[i]:
                        adv_actions[i] = raw_adv_actions[i]

            (vic_obs, adv_obs), (vic_reward, adv_reward), dones, (vic_info, adv_info) = self.env.step(vic_actions, adv_actions)
            opp_obs = np.concatenate((vic_obs, adv_obs), -1) if self.use_imitation and self.opp_agent else adv_obs
            
            self.update_info(dones, vic_info)
            self.update_info(dones, adv_info)
            self.setup_mixing(dones)

            self.vic_agent.add(vic_obs, vic_reward, dones, vic_info)
            self.opp_agent.add(opp_obs, adv_reward, dones, adv_info) if self.opp_agent else None
            self.adv_agent.add(adv_obs, adv_reward, dones, adv_info)
            self.raw_adv_agent.add(adv_obs, adv_reward, dones, adv_info) if self.raw_adv_agent else None
            
        vic_data = self.vic_agent.trajectories()
        
        return vic_data

