import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gym
import pickle
import torch
import gym_compete
import numpy as np
import tensorflow as tf
from agents import agent_policy_tf_v1
from agents.agent_policy_pytorch import load_policy
from stable_baselines.common.policies import ActorCriticPolicy, get_policy_from_name


ENV_NAMES = [
    "multicomp/RunToGoalAnts-v0",
    "multicomp/RunToGoalHumans-v0",
    "multicomp/YouShallNotPassHumans-v0",
    "multicomp/SumoHumans-v0",
    "multicomp/SumoAnts-v0",
    "multicomp/KickAndDefend-v0",
]

MLP_PARAM_DICT = {
    "input/logstd:0":           "actor.logstd",
    "input/retfilter/sum:0":    "critic.ret_rms.sum",
    "input/retfilter/sumsq:0":  "critic.ret_rms.sumsq",
    "input/retfilter/count:0":  "critic.ret_rms.count",
    "input/obsfilter/sum:0":    "ob_rms.sum",
    "input/obsfilter/sumsq:0":  "ob_rms.sumsq",
    "input/obsfilter/count:0":  "ob_rms.count",
    "input/vff1/w:0":           "critic.net.0.weight",
    "input/vff1/b:0":           "critic.net.0.bias",
    "input/vff2/w:0":           "critic.net.2.weight",
    "input/vff2/b:0":           "critic.net.2.bias",
    "input/vfffinal/w:0":       "critic.net.4.weight",
    "input/vfffinal/b:0":       "critic.net.4.bias",
    "input/pol1/w:0":           "actor.net.0.weight",
    "input/pol1/b:0":           "actor.net.0.bias",
    "input/pol2/w:0":           "actor.net.2.weight",
    "input/pol2/b:0":           "actor.net.2.bias",
    "input/polfinal/w:0":       "actor.net.4.weight",
    "input/polfinal/b:0":       "actor.net.4.bias",
}

LSTM_PARAM_DICT = {
    "input/logstd:0":                   "actor.logstd",
    "input/retfilter/sum:0":            "critic.ret_rms.sum",
    "input/retfilter/sumsq:0":          "critic.ret_rms.sumsq",
    "input/retfilter/count:0":          "critic.ret_rms.count",
    "input/obsfilter/sum:0":            "ob_rms.sum",
    "input/obsfilter/sumsq:0":          "ob_rms.sumsq",
    "input/obsfilter/count:0":          "ob_rms.count",
    "input/fully_connected/weights:0":          "critic.net.fc_in.0.weight",
    "input/fully_connected/biases:0":           "critic.net.fc_in.0.bias",
    "input/lstmv/basic_lstm_cell/kernel:0":     "critic.net.rnn_cell.weight_ih",
    "input/lstmv/basic_lstm_cell/bias:0":       "critic.net.rnn_cell.bias_ih",
    "input/fully_connected_1/weights:0":        "critic.net.fc_out.0.weight",
    "input/fully_connected_1/biases:0":         "critic.net.fc_out.0.bias",
    "input/fully_connected_2/weights:0":        "actor.net.fc_in.0.weight",
    "input/fully_connected_2/biases:0":         "actor.net.fc_in.0.bias",
    "input/lstmp/basic_lstm_cell/kernel:0":     "actor.net.rnn_cell.weight_ih",
    "input/lstmp/basic_lstm_cell/bias:0":       "actor.net.rnn_cell.bias_ih",
    "input/fully_connected_3/weights:0":        "actor.net.fc_out.0.weight",
    "input/fully_connected_3/biases:0":         "actor.net.fc_out.0.bias",
}

def check_equal(x, y):
    try:
        if x is None:
            assert y is None
        else:
            np.testing.assert_almost_equal(x, y, 2)
    except:
        raise

def swap(tensor, n, i, j):
    parts = list(torch.chunk(tensor, n, 0))
    parts[i], parts[j] = parts[j], parts[i]
    tensor = torch.cat(parts, 0)
    return tensor

def convert_tf_to_torch(env, path):
    print("-"*24)
    print("env:", env)
    print("path:", path)
    agent_id = int(path.split("agent")[-1].split("_")[0]) - 1 if "agent_" not in path else 0
    with open(path, "rb") as f:
        params = pickle.load(f)
    print("param:", params.shape)
    ob_space = env.observation_space.spaces[agent_id]
    ac_space = env.action_space.spaces[agent_id]
    ob_dim = ob_space.shape[0]
    ac_dim = ac_space.shape[0]
    tf.reset_default_graph()
    with tf.Session() as sess:
        n_envs = 8 # for testing
        n_steps = 64 # for testing
        if any(_env in path for _env in ["YouShallNotPassHumans", "RunToGoalAnts", "RunToGoalHumans"]):
            policy_class = get_policy_from_name(ActorCriticPolicy, "BansalMlpPolicy")
            use_rnn = False
        else:
            policy_class = get_policy_from_name(ActorCriticPolicy, "BansalLstmPolicy")
            use_rnn = True
        torch_policy = load_policy(ob_dim, ac_dim, n_envs, n_steps, normalize=True, use_lstm=use_rnn)
        tf_policy = policy_class(sess, ob_space, ac_space, n_envs, n_steps, n_envs*n_steps, normalize=True)
        tf_policy.restore(params)
        variables = tf_policy.get_trainable_variables()
        for variable in variables:
            print("tf variable    : ", variable.name)
        for n, p in torch_policy.named_parameters():
            print("torch variable : ", n)
        state_dict = {}
        for variable in variables:
            params = sess.run(variable).T
            params = torch.from_numpy(np.array(params))
            if not use_rnn:
                state_dict[MLP_PARAM_DICT[variable.name]] = params
            else:
                param_name = LSTM_PARAM_DICT[variable.name]
                if "rnn_cell" in param_name:
                    i_name = param_name
                    h_name = param_name.replace("_ih", "_hh")
                    if "bias" not in param_name:
                        w1, w2 = torch.chunk(params, 2, -1)
                        state_dict[i_name] = swap(w1, 4, 1, 2)
                        state_dict[h_name] = swap(w2, 4, 1, 2)
                    else:
                        parts = torch.chunk(params, 4, 0)
                        params = torch.cat([parts[y] + (1 if y == 2 else 0) for y in [0, 2, 1, 3]], 0)
                        state_dict[i_name] = params / 2
                        state_dict[h_name] = params / 2
                else:
                    state_dict[param_name] = params

        torch_policy.load_state_dict(state_dict)

        # test cases
        eps = 1
        obs = np.random.random((n_envs*n_steps, torch_policy.ob_dim)) * eps
        states = np.random.random((n_envs, 4, 128)) * eps
        masks = np.random.random((n_envs*n_steps,)) > 0.8
        if use_rnn:
            a1, v1, s1, n1 = torch_policy.step(obs, states, masks, deterministic=True)
            v1 = v1.reshape(-1)
        else:
            a1, v1, s1, n1 = torch_policy.step(obs, deterministic=True)
        a2, v2, s2, n2 = tf_policy.step(obs, states, masks, deterministic=True)

        check_equal(a1, a2)
        check_equal(v1, v2)
        check_equal(s1, s2)
        # check_equal(n1, n2)  # do not check neglogpacs

    torch_path = "agents/agent_zoo_torch_v1/" + path.split("/", 1)[-1]
    os.makedirs(torch_path.rsplit("/", 1)[0], exist_ok=True)
    torch.save(torch_policy.state_dict(), torch_path)


def convert(env_name):
    print("env_name:", env_name)
    _env_name = env_name.split("/")[-1]
    env = gym.make(env_name)
    zoo_path = f"agents/agent_zoo_tf_v1/{_env_name}"
    for file in os.listdir(zoo_path):
        if not file.endswith(".pkl"):
            continue
        convert_tf_to_torch(env, f"{zoo_path}/{file}")


if __name__ == "__main__":
    for env_name in ENV_NAMES:
        convert(env_name)
