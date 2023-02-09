import os
import gym
import torch
import random
import numpy as np
from tqdm import tqdm
from env.env import Env
from env.config import args
from datetime import datetime
from agents.agent_policy_pytorch import load_policy

import time

from visualize.video_utils import AnnotatedGymCompete
from visualize.video_wrapper import VideoWrapper
time.clock = time.time

N_GAMES = 1000 if not args.video else 20  # for evaluating


class MaskedEnv(Env):

    def __init__(self, blind_vic=False, blind_adv=False, **kwargs):
        super().__init__(**kwargs)
        self.init_qpos_values = {id: agent.get_other_qpos() for id, agent in self.env.agents.items()}
        if blind_vic:
            self.env.agents[self.vic_id].get_other_qpos = lambda: self.init_qpos_values[self.vic_id]
        if blind_adv:
            self.env.agents[self.adv_id].get_other_qpos = lambda: self.init_qpos_values[self.adv_id]

    def reset(self):
        return self._reset()

    def _reset(self):
        obs = super().reset()
        self.init_qpos_values = {id: agent.get_other_qpos() for id, agent in self.env.agents.items()}
        return obs


def clip_actions(actions, action_space):
    if isinstance(action_space, gym.spaces.Box):
        actions = np.clip(actions, action_space.low, action_space.high)
    return actions
    

def policy_step(env: Env, policy, obs, states):
    if args.tsne:
        policy.actor.set_ff_out(True)
    dones = [False]
    action, _, states, _ = policy.step(obs, states, dones, deterministic=True)
    action = clip_actions(action, env.action_space)
    action = np.squeeze(action)
    activations = policy.actor.ff_out if args.tsne else None
    return action, states, activations


video_params = {
    'single_file': True,              # if False, stores one file per episode
    'annotated': True,                # for gym_compete, color-codes the agents and adds scores
    'annotation_params': {
        'camera_config': 'close',
        'short_labels': False,
        'resolution': (640*2, 420*2),
        'font': 'times',
        'font_size': 36,
    },
}


def get_agent_name(use_retrained=True, path="", enhance=False):
    name = "Baseline"
    if use_retrained:
        if len(path) > 0:
            name = "ADRL" if "ucb" in path else "APL"
        else:
            name = "E-APIL (ours)" if enhance else "APIL (ours)"
    return name


def main():
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = MaskedEnv(**vars(args), shaping_params=args.params)
    if args.video:
        vic_name = get_agent_name(args.use_retrained_vic, args.retrained_vic_path, args.enhance_vic)
        adv_name = get_agent_name(args.use_retrained_adv, args.retrained_adv_path, args.enhance_adv)
        env_name = args.env_name.split("/")[-1]
        os.makedirs("videos", exist_ok=True)
        save_dir = f"videos/{env_name}_{vic_name}_{adv_name}"
        save_dir = save_dir.replace("(ours)", "").replace("\n", "")
        env = AnnotatedGymCompete(env, args.env_name, vic_name, adv_name, None, **video_params['annotation_params'])
        env = VideoWrapper(env, save_dir, video_params['single_file'])

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    n_envs = args.n_envs
    n_steps = args.n_steps
    log_dir = args.log_dir
    env_name = args.env_name.split("/")[-1]
    
    vic_policy = load_policy(ob_dim, ac_dim, n_envs, zoo_path=args.zoo_path, normalize=True).to(args.device).eval()
    if args.use_retrained_vic:
        vic_policy.critic.normalize = False
        vic_log_dir = log_dir.replace(f"env_{n_envs}_step_{n_steps}", "env_8_step_2048")
        vic_log_dir += "_robust_victim"
        if not args.enhance_vic: vic_log_dir = vic_log_dir.replace("_enhance", "")
        vic_path = f"{vic_log_dir}/vic_policy.pt" if len(args.retrained_vic_path)==0 else f"agents/{args.retrained_vic_path}"
        vic_policy.load_state_dict(torch.load(vic_path))

    if args.use_retrained_adv:
        if len(args.retrained_adv_path) == 0:
            adv_log_dir = log_dir.replace(f"env_{n_envs}_step_{n_steps}", "env_32_step_2048" if "Ants" not in env_name else "env_8_step_2048")
            if not args.enhance_adv: adv_log_dir = adv_log_dir.replace("_enhance", "")
            opp_policy = load_policy(ob_dim*2, ac_dim, n_envs, use_lstm=args.use_lstm).to(args.device).eval()
            adv_policy = load_policy(ob_dim*2+ac_dim, ac_dim, n_envs, use_lstm=args.use_lstm).to(args.device).eval()
            opp_policy.load_state_dict(torch.load(f"{adv_log_dir}/opp_policy.pt"))
            adv_policy.load_state_dict(torch.load(f"{adv_log_dir}/adv_policy.pt"))
        else:
            adv_policy = load_policy(ob_dim, ac_dim, n_envs, zoo_path=args.retrained_adv_path, normalize=True).to(args.device).eval()
            opp_policy = None
    else:
        adv_zoo_id = "" if "Sumo" in env_name else (3 - args.vic_agent_id)
        adv_zoo_path = f"agent_zoo_torch_v1/{env_name}/agent{adv_zoo_id}_parameters-v{args.zoo_ver}.pkl"
        adv_policy = load_policy(ob_dim, ac_dim, n_envs, zoo_path=adv_zoo_path, normalize=True).to(args.device).eval()
        opp_policy = None

    logs = {}
    vic_tsne_acts = []
    for step in tqdm(range(N_GAMES), "Evaluating", ncols=0):
        vic_states = vic_policy.initial_state
        opp_states = opp_policy.initial_state if opp_policy else None
        adv_states = adv_policy.initial_state
        vic_obs, adv_obs = env.reset()
        while True:
            opp_obs = np.concatenate((vic_obs, adv_obs), -1) if opp_policy else None
            vic_action, vic_states, vic_tsne = policy_step(env, vic_policy, vic_obs, vic_states)
            opp_action, opp_states, _ = policy_step(env, opp_policy, opp_obs, opp_states) if opp_policy else (None, None, None)
            adv_obs = np.concatenate((adv_obs, vic_obs, opp_action), -1) if opp_policy else adv_obs
            adv_action, adv_states, _ = policy_step(env, adv_policy, adv_obs, adv_states)
            if args.video:
                env.render()
            (vic_obs, adv_obs), _, done, (info, _) = env.step(vic_action, adv_action)
            vic_tsne_acts.append(vic_tsne)
            if done:
                for key in ["winner", "loser", "tie"]:
                    if key in info:
                        logs[key] = logs.get(key, 0) + 1
                break
        
        if (step + 1) % 10 == 0 or step + 1 == N_GAMES:
            rewards = [logs.get(key, 0) / sum(logs.values()) for key in ["winner", "loser", "tie"]]
            rewards = ", ".join(f"{np.mean(x):.4f}" for x in rewards)
            
            name = f"seed_{args.seed}"
            name += '_lstm' if args.use_lstm else '_mlp'
            name += '_opp' if args.use_opp else ''
            name += '_imitation' if args.use_imitation else ''
            name += '_l2t' if args.use_l2t else ''
            name += '_enhance_vic' if args.enhance_vic else ''
            name += '_enhance_adv' if args.enhance_adv else ''
            if args.use_retrained_vic: name = f"{name}_retrained_vic"
            if args.use_retrained_adv: name = f"{name}_retrained_adv"
            if args.blind_vic: name = f"{name}_blind_vic"
            if args.blind_adv: name = f"{name}_blind_adv"

            if args.tsne:
                max_tsne_stack_len = 20000
                if len(vic_tsne_acts) >= max_tsne_stack_len:
                    if len(args.retrained_vic_path) > 0: name = f"{name}_vic_paper_1" if "ucb" in args.retrained_vic_path else f"{name}_vic_paper_2"
                    if len(args.retrained_adv_path) > 0: name = f"{name}_adv_paper_1" if "ucb" in args.retrained_adv_path else f"{name}_adv_paper_2"
                    vic_tsne_acts = torch.stack(vic_tsne_acts[:max_tsne_stack_len])
                    print("vic_tsne_acts:", vic_tsne_acts.shape)
                    torch.save(vic_tsne_acts, f"logs/tsne_{env_name}_{name}.pkl")
                    exit()
            else:
                if len(args.retrained_vic_path) > 0: name = f"{name}|vic_{args.retrained_vic_path}"
                if len(args.retrained_adv_path) > 0: name = f"{name}|adv_{args.retrained_adv_path}"
                with open(f"logs/log_{env_name}.txt", "a") as f:
                    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"[{t}] Step {step+1}: {rewards} - {name}\n")
    env.close()

if __name__ == "__main__":
    main()