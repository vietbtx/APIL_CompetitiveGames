import os
import numpy as np
from time import sleep
from itertools import chain, combinations


ENV_NAMES = [
    "YouShallNotPassHumans-v0",
    "KickAndDefend-v0",
    "SumoHumans-v0",
    "SumoAnts-v0",
]


def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))


def run_cmd(pid, cmd):
    cmd = cmd.strip()
    if "YouShallNotPassHumans" in cmd:
        cmd += " --vic-agent-id 2"
    if "train_vic" in cmd:
        cmd += " --n-envs 8"
        cmd += " --nminibatches 2"
        cmd += " --n-timesteps 10000000"
    if "evaluate" in cmd:
        cmd += " --n-envs 1 --n-steps 1"
    print("cmd:", cmd)
    sleep(pid*10)
    os.system(cmd)
    

def swap_and_flatten(arr):
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])


def explained_variance(y_pred, y_true):
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


def get_schedule_fn(value_schedule, schedule):
    if schedule == 'const':
        value_schedule = constfn(value_schedule)
    elif schedule == 'linear':
        value_schedule = linearfn(value_schedule)
    elif schedule == 'step':
        value_schedule = stepfn(value_schedule)
    else:
        assert callable(value_schedule)
    return value_schedule


def linearfn(val):
    def func(epoch, total_epoch):
        frac = 1.0 - (epoch - 1.0) / total_epoch 
        return val*frac
    return func


def stepfn(val):
    def func(epoch, drop=0.8, epoch_drop=400):
        ratio = drop**((epoch+1) // epoch_drop)
        return val*ratio
    return func


def constfn(val):
    def func(_):
        return val
    return func

def count_win_rate(logs={}):
    data = []
    n_games = sum(logs.values())
    if n_games > 0:
        for key in ["win", "lose", "tie"]:
            value = logs.get(key, 0) / n_games
            data.append(value)
    return data, n_games

def read_ep_infos(ep_infos):
    logs, logs_ori = {}, {}
    for info in ep_infos:
        for key in ["winner", "loser", "tie"]:
            if key in info:
                if key == "winner": key = "win"
                if key == "loser": key = "lose"
                if "use_mixing" in info:
                    logs_ori[key] = logs_ori.get(key, 0) + 1
                else:
                    logs[key] = logs.get(key, 0) + 1
    rewards, n_games = count_win_rate(logs)
    rewards_ori, n_games_ori = count_win_rate(logs_ori)
    return rewards, n_games, rewards_ori, n_games_ori