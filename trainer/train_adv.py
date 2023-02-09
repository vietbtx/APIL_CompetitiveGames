import time
from env.config import args


def main():
    from env.env import VectorizedEnv
    venv = VectorizedEnv(args)
    ob_dim = venv.ob_dim
    ac_dim = venv.ac_dim
    n_envs = args.n_envs

    import torch
    import random
    import numpy as np
    from pprint import pprint
    from trainer.runner import Runner
    from trainer.ppo import PPO2, OppPPO2, ImitationPPO2
    from agents.agent_policy_pytorch import load_policy

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    vic_policy = load_policy(ob_dim, ac_dim, n_envs, zoo_path=args.zoo_path, normalize=True).to(args.device).eval()

    opp_policy = None
    disc = None

    if args.use_imitation:
        opp_policy = load_policy(ob_dim*2, ac_dim, n_envs, use_lstm=args.use_lstm)
        adv_policy = load_policy(ob_dim*2+ac_dim, ac_dim, n_envs, use_lstm=args.use_lstm)
        adv_agent = ImitationPPO2(adv_policy, opp_policy, **vars(args))
        disc = adv_agent.disc
    elif args.use_opp:
        opp_policy = load_policy(ob_dim, ac_dim, n_envs, use_lstm=args.use_lstm)
        adv_policy = load_policy(ob_dim, ac_dim, n_envs, use_lstm=args.use_lstm)
        adv_agent = OppPPO2(adv_policy, opp_policy, **vars(args))
    else:
        adv_policy = load_policy(ob_dim, ac_dim, n_envs, use_lstm=args.use_lstm)
        adv_agent = PPO2(adv_policy, **vars(args))
    
    runner = Runner(venv, vic_policy, adv_policy, opp_policy, disc, **vars(args))
    
    pprint(vars(args))
    
    step = 0
    while True:
        step += 1
        t_start = time.time()
        vic_data, adv_data, opp_data = runner.run()
        if not adv_agent.train(step, t_start, adv_data, opp_data, vic_data):
            break
        if step % 10 == 0:
            torch.save(adv_policy.state_dict(), f"{args.log_dir}/adv_policy.pt")
            torch.save(opp_policy.state_dict(), f"{args.log_dir}/opp_policy.pt") if opp_policy else None
            torch.save(disc.state_dict(), f"{args.log_dir}/disc.pt") if disc else None

if __name__ == "__main__":
    main()

