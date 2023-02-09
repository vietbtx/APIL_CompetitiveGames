import time
from env.config import args


def main():
    from pprint import pprint
    from env.env import VectorizedEnv
    venv = VectorizedEnv(args)
    ob_dim = venv.ob_dim
    ac_dim = venv.ac_dim
    n_envs = args.n_envs
    n_steps = args.n_steps

    env_name = args.env_name.split("/")[-1]
    adv_zoo_id = "" if "Sumo" in env_name else (3 - args.vic_agent_id)
    adv_zoo_path = f"agent_zoo_torch_v1/{env_name}/agent{adv_zoo_id}_parameters-v{args.zoo_ver}.pkl"
    log_dir = args.log_dir
    args.log_dir += "_robust_victim"
    args.use_l2t = False

    import torch
    import random
    import numpy as np
    from trainer.ppo import PPO2
    from trainer.runner import MixingRunner
    from agents.agent_policy_pytorch import load_policy
    from pprint import pprint

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    vic_policy = load_policy(ob_dim, ac_dim, n_envs, zoo_path=args.zoo_path, normalize=True)
    vic_policy.critic.normalize = False
    vic_agent = PPO2(vic_policy, **vars(args))

    opp_policy = None

    if args.use_imitation:
        opp_policy = load_policy(ob_dim*2, ac_dim, n_envs, use_lstm=args.use_lstm).to(args.device).eval()
        adv_policy = load_policy(ob_dim*2+ac_dim, ac_dim, n_envs, use_lstm=args.use_lstm).to(args.device).eval()
    elif args.use_opp:
        opp_policy = load_policy(ob_dim, ac_dim, n_envs, use_lstm=args.use_lstm).to(args.device).eval()
        adv_policy = load_policy(ob_dim, ac_dim, n_envs, use_lstm=args.use_lstm).to(args.device).eval()
    else:
        adv_policy = load_policy(ob_dim, ac_dim, n_envs, use_lstm=args.use_lstm).to(args.device).eval()
    
    
    log_dir = log_dir.replace(f"env_{n_envs}_step_{n_steps}", "env_32_step_2048" if "Ants" not in env_name else "env_8_step_2048")
    adv_policy.load_state_dict(torch.load(f"{log_dir}/adv_policy.pt"))
    opp_policy.load_state_dict(torch.load(f"{log_dir}/opp_policy.pt")) if opp_policy else None
    raw_adv_policy = load_policy(ob_dim, ac_dim, n_envs, zoo_path=adv_zoo_path, normalize=True).to(args.device).eval()
    
    runner = MixingRunner(venv, vic_policy, adv_policy, opp_policy, raw_adv_policy, **vars(args))
    
    pprint(vars(args))
    
    step = 0
    while True:
        step += 1
        t_start = time.time()
        vic_data = runner.run()
        if not vic_agent.train(step, t_start, vic_data):
            break
        if step % 10 == 0:
            torch.save(vic_policy.state_dict(), f"{args.log_dir}/vic_policy.pt")

if __name__ == "__main__":
    main()

