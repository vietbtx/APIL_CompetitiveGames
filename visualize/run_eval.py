from multiprocessing import Pool
from trainer.utils import ENV_NAMES, run_cmd


VIC_PATH_PAPER1 = {
    "SumoAnts-v0": "agent_zoo_torch_v2/retrained-victim/ucb/SumoAnts/20200818_233706-0/model.pt",
    "SumoHumans-v0": "agent_zoo_torch_v2/retrained-victim/ucb/SumoHumans/20200820_175047-0/model.pt",
    "KickAndDefend-v0": "agent_zoo_torch_v2/retrained-victim/ucb/KickAndDefend/20200819_150752-0/model.pt",
    "YouShallNotPassHumans-v0": "agent_zoo_torch_v2/retrained-victim/ucb/YouShallNotPass/20200820_160129-0/model.pt",
}

VIC_PATH_PAPER2 = {
    "SumoAnts-v0": "agent_zoo_torch_v2/retrained-victim/our_attack/SumoAnts/20200818_124818-0/model.pt",
    "SumoHumans-v0": "agent_zoo_torch_v2/retrained-victim/our_attack/SumoHumans/20200817_093255-0/model.pt",
    "KickAndDefend-v0": "agent_zoo_torch_v2/retrained-victim/our_attack/KickAndDefend/20200818_124108-0/model.pt",
    "YouShallNotPassHumans-v0": "agent_zoo_torch_v2/retrained-victim/our_attack/YouShallNotPass/20200816_125658-0/model.pt",
}

ADV_PATH_PAPER1 = {
    "SumoAnts-v0": "agent_zoo_torch_v2/adv-agent/ucb/ants/model.pt",
    "SumoHumans-v0": "agent_zoo_torch_v2/adv-agent/ucb/humans/model.pt",
    "KickAndDefend-v0": "agent_zoo_torch_v2/adv-agent/ucb/kick/model.pt",
    "YouShallNotPassHumans-v0": "agent_zoo_torch_v2/adv-agent/ucb/you/model.pt",
}

ADV_PATH_PAPER2 = {
    "SumoAnts-v0": "agent_zoo_torch_v2/adv-agent/our_attack/ants/model.pt",
    "SumoHumans-v0": "agent_zoo_torch_v2/adv-agent/our_attack/humans/model.pt",
    "KickAndDefend-v0": "agent_zoo_torch_v2/adv-agent/our_attack/kick/model.pt",
    "YouShallNotPassHumans-v0": "agent_zoo_torch_v2/adv-agent/our_attack/you/model.pt",
}


def generate_cmds(env_name, vic_agents, adv_agents, blind_vic=False, blind_adv=False, use_tsne=False, video=False):
    cmds = []
    base_cmd = f"python -m visualize.evaluate --env-name {env_name} --use-imitation --enhance-reward"
    for vic_agent, vic_args in vic_agents.items():
        for adv_agent, adv_args in adv_agents.items():
            cmd = f"{base_cmd} {vic_args} {adv_args}"
            if vic_agent == "Base" and "Paper" in adv_agent:
                if env_name == "SumoHumans-v0":
                    cmd += " --zoo-ver 3"
                elif env_name == "SumoAnts-v0":
                    cmd += " --zoo-ver 2"
            if blind_vic: cmd += " --blind-vic"
            if blind_adv: cmd += " --blind-adv"
            if use_tsne: cmd += " --tsne"
            if video: cmd += " --video"
            cmds.append(cmd)
    return cmds


def main():
    cmds = []

    for env_name in ENV_NAMES:
        vic_path_1 = VIC_PATH_PAPER1[env_name]
        vic_path_2 = VIC_PATH_PAPER2[env_name]
        adv_path_1 = ADV_PATH_PAPER1[env_name]
        adv_path_2 = ADV_PATH_PAPER2[env_name]

        vic_agents = {
            "Base": "",
            "Paper1": f"--use-retrained-vic --retrained-vic-path {vic_path_1}",
            "Paper2": f"--use-retrained-vic --retrained-vic-path {vic_path_2}",
            "V1": "--use-retrained-vic",
            "V2": "--use-retrained-vic --enhance-vic",
        }

        adv_agents = {
            "Base": "",
            "Paper1": f"--use-retrained-adv --retrained-adv-path {adv_path_1}",
            "Paper2": f"--use-retrained-adv --retrained-adv-path {adv_path_2}",
            "V1": "--use-retrained-adv",
            "V2": "--use-retrained-adv --enhance-adv",
        }
        cmds += generate_cmds(env_name, vic_agents, adv_agents)
        cmds += generate_cmds(env_name, vic_agents, adv_agents, blind_vic=True)
        cmds += generate_cmds(env_name, vic_agents, adv_agents, blind_adv=True)

    cmds = cmds[:1]
    print("cmds:", len(cmds), len(set(cmds)))                
    with Pool(20) as p:
        p.starmap(run_cmd, [((i%20)/100, cmd) for i, cmd in enumerate(cmds)])


if __name__ == "__main__":
    main()