from multiprocessing import Pool
from trainer.utils import ENV_NAMES, run_cmd
from visualize.run_eval import *


def main():
    cmds = []

    for env_name in ENV_NAMES:
        vic_path_1 = VIC_PATH_PAPER1[env_name]
        vic_path_2 = VIC_PATH_PAPER2[env_name]
        adv_path_1 = ADV_PATH_PAPER1[env_name]
        adv_path_2 = ADV_PATH_PAPER2[env_name]

        vic_agents = {
            # "Base": "",
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
        cmds += generate_cmds(env_name, vic_agents, adv_agents, video=True)

    print("cmds:", len(cmds), len(set(cmds)))
    with Pool(5) as p:
        p.starmap(run_cmd, [(0, cmd) for cmd in cmds])


if __name__ == "__main__":
    main()