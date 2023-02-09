from multiprocessing import Pool
from trainer.utils import ENV_NAMES, run_cmd, all_subsets


if __name__ == "__main__":
    cmds = []
    options = ["--enhance-reward"]
    for env_name in ENV_NAMES:
        base_cmd = f"python -m trainer.train_vic --env-name {env_name} --use-imitation"
        for option in all_subsets(options):
            ext_cmd = " ".join(option)
            cmds.append(f"{base_cmd} {ext_cmd}")
    cmds = cmds[:1]
    with Pool(16) as p:
        p.starmap(run_cmd, enumerate(cmds))