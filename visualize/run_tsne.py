from visualize.run_eval import *
from multiprocessing import Pool
from trainer.utils import ENV_NAMES, run_cmd


def main():
    cmds = []

    for env_name in ENV_NAMES:
        adv_path_1 = ADV_PATH_PAPER1[env_name]
        adv_path_2 = ADV_PATH_PAPER2[env_name]

        vic_agents = {
            "Base": "",
        }

        adv_agents = {
            "Base": "",
            "Paper1": f"--use-retrained-adv --retrained-adv-path {adv_path_1}",
            "Paper2": f"--use-retrained-adv --retrained-adv-path {adv_path_2}",
            "V1": "--use-retrained-adv",
            "V2": "--use-retrained-adv --enhance-adv",
        }
        cmds += generate_cmds(env_name, vic_agents, adv_agents, use_tsne=True)

    cmds = cmds[:1]
    print("cmds:", len(cmds), len(set(cmds)))
    with Pool(5) as p:
        p.starmap(run_cmd, [((i%20)/100, cmd) for i, cmd in enumerate(cmds)], chunksize=1)

if __name__ == "__main__":
    main()