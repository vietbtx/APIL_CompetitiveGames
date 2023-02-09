import argparse
from collections import defaultdict
import sys

from trainer.utils import ENV_NAMES

def value_max(d: dict):
    v = max(d.keys())
    return d[v]

def read_data(env_name):
    logs = defaultdict(dict)
    with open(f"logs/log_{env_name}.txt", "r") as f:
        for line in f:
            line = line.split("Step ")[-1].strip()
            parts = line.split(": ", 1)
            step = int(parts[0])
            line = parts[1]
            reward, name = line.split(" - ", 1)
            logs[name][step] = reward
    logs = {k: value_max(v) for k, v in logs.items()}
    return logs

def generate_pairs(names, blind_vic=False, blind_adv=False):
    pairs = {}
    if blind_vic:
        names = [name for name in names if "blind_vic" in name]
    elif blind_adv:
        names = [name for name in names if "blind_adv" in name]
    else:
        names = [name for name in names if "blind" not in name]
    for name in names:
        vic_tag = "Base"
        adv_tag = "Base"
        if "retrained_vic" in name:
            if "vic_agent_zoo" in name:
                vic_tag = "Paper1" if "retrained-victim/ucb" in name else "Paper2"
            else:
                vic_tag = "V1" if "enhance_vic" not in name else "V2"
        if "retrained_adv" in name:
            if "adv_agent_zoo" in name:
                adv_tag = "Paper1" if "adv-agent/ucb" in name else "Paper2"
            else:
                adv_tag = "V1" if "enhance_adv" not in name else "V2"
        pairs[(vic_tag, adv_tag)] = name
    return pairs

def analyze(env_name, blind_vic=False, blind_adv=False):
    print("-"*32)
    print(f"env_name: {env_name} - blind_vic: {blind_vic} - blind_adv: {blind_adv}")
    logs = read_data(env_name)
    names = logs.keys()
    pairs = generate_pairs(names, blind_vic, blind_adv)
    agents = ["Base", "Paper1", "Paper2", "V1", "V2"]
    for vic_agent in agents:
        for adv_agent in agents:
            key = (vic_agent, adv_agent)
            if key in pairs:
                name = pairs[key]
                win, lose, tie = [float(r) for r in logs[name].split(", ")]
                print(f"{win:.4f}", end="\t")
            else:
                print(" - ", end="\t")
        print()

        for adv_agent in agents:
            key = (vic_agent, adv_agent)
            if key in pairs:
                name = pairs[key]
                win, lose, tie = [float(r) for r in logs[name].split(", ")]
                print(f"{win+tie:.4f}", end="\t")
            else:
                print(" - ", end="\t")
        print()

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind-vic", action='store_true')                 # evaluate
    parser.add_argument("--blind-adv", action='store_true')                 # evaluate
    args = parser.parse_args()
    for env_name in ENV_NAMES:
        try:
            analyze(env_name, args.blind_vic, args.blind_adv)
        except:
            pass
        