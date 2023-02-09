import os
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from agents.agent_policy_pytorch import load_policy
from agents.convert_zoo_agents_v1 import check_equal, swap
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gym
import torch
import numpy as np
import gym_compete
import tensorflow as tf
from agents.zoo_utils import LSTMPolicy, MlpPolicyValue
from stable_baselines.common.policies import MlpPolicy
from agents.zoo_utils import setFromFlat, load_from_file, load_from_model


ENV_NAMES_1 = {
    "multicomp/YouShallNotPassHumans-v0": "you",
    "multicomp/SumoHumans-v0": "humans",
    "multicomp/SumoAnts-v0": "ants",
    "multicomp/KickAndDefend-v0": "kick",
}

ENV_NAMES_2 = {
    "multicomp/YouShallNotPassHumans-v0": "YouShallNotPass",
    "multicomp/SumoHumans-v0": "SumoHumans",
    "multicomp/SumoAnts-v0": "SumoAnts",
    "multicomp/KickAndDefend-v0": "KickAndDefend",
}

MODEL_BASE = {
    "multicomp/YouShallNotPassHumans-v0": "agent_zoo_torch_v1/YouShallNotPassHumans-v0/agent1_parameters-v1.pkl",
    "multicomp/SumoHumans-v0": "agent_zoo_torch_v1/SumoHumans-v0/agent_parameters-v3.pkl",
    "multicomp/SumoAnts-v0": "agent_zoo_torch_v1/SumoAnts-v0/agent_parameters-v1.pkl",
    "multicomp/KickAndDefend-v0": "agent_zoo_torch_v1/KickAndDefend-v0/agent1_parameters-v1.pkl"
}

NORM_VIC_PAPER1 = {
	"multicomp/KickAndDefend-v0": ['agent_zoo_tf_v2/multiagent-competition/agent-zoo/kick-and-defend/kicker/agent1_parameters-v1.pkl', 'our', 'lstm'],
	"multicomp/YouShallNotPassHumans-v0": ['agent_zoo_tf_v2/multiagent-competition/agent-zoo/you-shall-not-pass/agent2_parameters-v1.pkl', 'our', 'mlp'],
	"multicomp/SumoHumans-v0": ['agent_zoo_tf_v2/multiagent-competition/agent-zoo/sumo/humans/agent_parameters-v3.pkl', 'our', 'lstm'],
	"multicomp/SumoAnts-v0": ['agent_zoo_tf_v2/multiagent-competition/agent-zoo/sumo/ants/agent_parameters-v1.pkl', 'our', 'lstm'],
}

NORM_VIC_PAPER2 = {
	"multicomp/KickAndDefend-v0": ['agent_zoo_tf_v2/multiagent-competition/agent-zoo/kick-and-defend/kicker/agent1_parameters-v1.pkl', 'our', 'lstm'],
	"multicomp/YouShallNotPassHumans-v0": ['agent_zoo_tf_v2/multiagent-competition/agent-zoo/you-shall-not-pass/agent2_parameters-v1.pkl', 'our', 'mlp'],
	"multicomp/SumoHumans-v0": ['agent_zoo_tf_v2/multiagent-competition/agent-zoo/sumo/humans/agent_parameters-v3.pkl', 'our', 'lstm'],
	"multicomp/SumoAnts-v0": ['agent_zoo_tf_v2/multiagent-competition/agent-zoo/sumo/ants/agent_parameters-v1.pkl', 'our', 'lstm'],
}

NORM_ADV_PAPER1 = {
	"multicomp/KickAndDefend-v0": ['agent_zoo_tf_v2/adv-agent/ucb/kick/obs_rms.pkl', 'stable', 'mlp'],
	"multicomp/YouShallNotPassHumans-v0": ['agent_zoo_tf_v2/adv-agent/ucb/you/obs_rms.pkl', 'stable', 'mlp'],
	"multicomp/SumoHumans-v0": ['agent_zoo_tf_v2/adv-agent/ucb/humans/obs_rms.pkl', 'stable', 'mlp'],
	"multicomp/SumoAnts-v0": ['agent_zoo_tf_v2/adv-agent/ucb/ants/obs_rms.pkl', 'stable', 'mlp'],
}

NORM_ADV_PAPER2 = {
	"multicomp/KickAndDefend-v0": ['agent_zoo_tf_v2/adv-agent/our_attack/kick/obs_rms.pkl', 'stable', 'mlp'],
	"multicomp/YouShallNotPassHumans-v0": ['agent_zoo_tf_v2/adv-agent/our_attack/you/obs_rms.pkl', 'stable', 'mlp'],
	"multicomp/SumoHumans-v0": ['agent_zoo_tf_v2/adv-agent/our_attack/humans/obs_rms.pkl', 'stable', 'mlp'],
	"multicomp/SumoAnts-v0": ['agent_zoo_tf_v2/adv-agent/our_attack/ants/obs_rms.pkl', 'stable', 'mlp'],
}

MLP_PARAM_DICT = {
    "model/pi/logstd:0":           "actor.logstd",
    "model/retfilter/sum:0":    "critic.ret_rms.sum",
    "model/retfilter/sumsq:0":  "critic.ret_rms.sumsq",
    "model/retfilter/count:0":  "critic.ret_rms.count",
    "model/obsfilter/sum:0":    "ob_rms.sum",
    "model/obsfilter/sumsq:0":  "ob_rms.sumsq",
    "model/obsfilter/count:0":  "ob_rms.count",
    "model/vf_fc0/w:0":           "critic.net.0.weight",
    "model/vf_fc0/b:0":           "critic.net.0.bias",
    "model/vf_fc1/w:0":           "critic.net.2.weight",
    "model/vf_fc1/b:0":           "critic.net.2.bias",
    "model/vf/w:0":       "critic.net.4.weight",
    "model/vf/b:0":       "critic.net.4.bias",
    "model/pi_fc0/w:0":           "actor.net.0.weight",
    "model/pi_fc0/b:0":           "actor.net.0.bias",
    "model/pi_fc1/w:0":           "actor.net.2.weight",
    "model/pi_fc1/b:0":           "actor.net.2.bias",
    "model/pi/w:0":       "actor.net.4.weight",
    "model/pi/b:0":       "actor.net.4.bias",
}

MLP_PARAM_DICT_2 = {
    "model/logstd:0":           "actor.logstd",
    "model/retfilter/sum:0":    "critic.ret_rms.sum",
    "model/retfilter/sumsq:0":  "critic.ret_rms.sumsq",
    "model/retfilter/count:0":  "critic.ret_rms.count",
    "model/obsfilter/sum:0":    "ob_rms.sum",
    "model/obsfilter/sumsq:0":  "ob_rms.sumsq",
    "model/obsfilter/count:0":  "ob_rms.count",
    "model/vffc1/w:0":           "critic.net.0.weight",
    "model/vffc1/b:0":           "critic.net.0.bias",
    "model/vffc2/w:0":           "critic.net.2.weight",
    "model/vffc2/b:0":           "critic.net.2.bias",
    "model/vffinal/w:0":       "critic.net.4.weight",
    "model/vffinal/b:0":       "critic.net.4.bias",
    "model/polfc1/w:0":           "actor.net.0.weight",
    "model/polfc1/b:0":           "actor.net.0.bias",
    "model/polfc2/w:0":           "actor.net.2.weight",
    "model/polfc2/b:0":           "actor.net.2.bias",
    "model/polfinal/w:0":       "actor.net.4.weight",
    "model/polfinal/b:0":       "actor.net.4.bias",
}

LSTM_PARAM_DICT = {
    "model/logstd:0":                   "actor.logstd",
    "model/retfilter/sum:0":            "critic.ret_rms.sum",
    "model/retfilter/sumsq:0":          "critic.ret_rms.sumsq",
    "model/retfilter/count:0":          "critic.ret_rms.count",
    "model/obsfilter/sum:0":            "ob_rms.sum",
    "model/obsfilter/sumsq:0":          "ob_rms.sumsq",
    "model/obsfilter/count:0":          "ob_rms.count",
    "model/fully_connected/weights:0":          "critic.net.fc_in.0.weight",
    "model/fully_connected/biases:0":           "critic.net.fc_in.0.bias",
    "model/lstmv/basic_lstm_cell/kernel:0":     "critic.net.rnn_cell.weight_ih",
    "model/lstmv/basic_lstm_cell/bias:0":       "critic.net.rnn_cell.bias_ih",
    "model/fully_connected_1/weights:0":        "critic.net.fc_out.0.weight",
    "model/fully_connected_1/biases:0":         "critic.net.fc_out.0.bias",
    "model/fully_connected_2/weights:0":        "actor.net.fc_in.0.weight",
    "model/fully_connected_2/biases:0":         "actor.net.fc_in.0.bias",
    "model/lstmp/basic_lstm_cell/kernel:0":     "actor.net.rnn_cell.weight_ih",
    "model/lstmp/basic_lstm_cell/bias:0":       "actor.net.rnn_cell.bias_ih",
    "model/fully_connected_3/weights:0":        "actor.net.fc_out.0.weight",
    "model/fully_connected_3/biases:0":         "actor.net.fc_out.0.bias",
}

def read_policy(sess, env, n_envs, n_steps, pi0_type, pi0_nn_type, pi0_path, pi0_norm_path):
    print("pi0_type:", pi0_type)
    print("pi0_nn_type:", pi0_nn_type)
    print("pi0_path:", pi0_path)
    print("pi0_norm_path:", pi0_norm_path)

    if pi0_type == 'our':
        if pi0_nn_type == 'mlp':
            policy = MlpPolicyValue(scope="model", reuse=False,
                                         ob_space=env.observation_space.spaces[0],
                                         ac_space=env.action_space.spaces[0],
                                         hiddens=[64, 64], normalize=True)
        else:
            policy = LSTMPolicy(scope="model", reuse=False,
                                     ob_space=env.observation_space.spaces[0],
                                     ac_space=env.action_space.spaces[0],
                                     hiddens=[128, 128], normalize=True, n_envs=n_envs, n_batch_train=n_envs*n_steps)
    else:
        policy = MlpPolicy(sess, env.observation_space.spaces[0], env.action_space.spaces[0], n_envs, n_steps, n_envs*n_steps, reuse=False)

    
    sess.run(tf.variables_initializer(tf.global_variables()))

    # load running mean/variance and model for opp_agent (model)
    if pi0_type == 'our':
        if pi0_nn_type == 'mlp':
            # load running mean/variance and model
            none_trainable_list = policy.get_variables()[:6]
            shapes = list(map(lambda x: x.get_shape().as_list(), none_trainable_list))
            none_trainable_size = np.sum([int(np.prod(shape)) for shape in shapes])
            none_trainable_param = load_from_file(pi0_norm_path)[:none_trainable_size]
            if 'multiagent-competition' in pi0_path:
                trainable_param = load_from_file(pi0_path)[none_trainable_size:]
            else:
                trainable_param = load_from_model(param_pkl_path=pi0_path)
            param = np.concatenate([none_trainable_param, trainable_param], axis=0)
            setFromFlat(policy.get_variables(), param)
        else:
            none_trainable_list = policy.get_variables()[:12]
            shapes = list(map(lambda x: x.get_shape().as_list(), none_trainable_list))
            none_trainable_size = np.sum([int(np.prod(shape)) for shape in shapes])
            none_trainable_param = load_from_file(pi0_norm_path)[:none_trainable_size]
            if 'multiagent-competition' in pi0_path:
                trainable_param = load_from_file(pi0_path)[none_trainable_size:]
            else:
                trainable_param = load_from_model(param_pkl_path=pi0_path)
            param = np.concatenate([none_trainable_param, trainable_param], axis=0)
            setFromFlat(policy.get_variables(), param)
    else:
        param = load_from_model(param_pkl_path=pi0_path)
        setFromFlat(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model'), param)
    return policy

def convert_tf_to_torch(env, folder, model_file, obs_rms_file=None, ret_rms_file=None, p_type=None, nn_type=None, tf_base_path=None, base_params=None):
    print("-"*24)
    print("env:", env, "- p_type:", p_type, "- nn_type:", nn_type)
    print("folder:", f"{folder}/{model_file}")
    print("tf_base_path:", tf_base_path)
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    tf.reset_default_graph()
    
    ob_space = env.observation_space.spaces[0]
    ac_space = env.action_space.spaces[0]
    ob_dim = ob_space.shape[0]
    ac_dim = ac_space.shape[0]
    use_rnn = nn_type == "lstm"
    n_envs = 32                         # for testing
    n_steps = 1 if use_rnn else 64      # for testing

    with tf.Session(config=tf_config) as sess:
        if obs_rms_file is not None:
            normalize = "ob"
            tf_policy = read_policy(sess, env, n_envs, n_steps, p_type, nn_type, f"{folder}/{model_file}", f"{folder}/{obs_rms_file}")
        else:
            normalize = True
            tf_policy = read_policy(sess, env, n_envs, n_steps, p_type, nn_type, f"{folder}/{model_file}", f"agents/{tf_base_path}")
        torch_policy = load_policy(ob_dim, ac_dim, n_envs, n_steps, normalize=normalize, use_lstm=use_rnn)
        variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='model')
        
        state_dict = {}
        for variable in variables:
            print("variable:", variable)
            if "model/q/" in variable.name:
                continue
            params = sess.run(variable).T
            params = torch.from_numpy(np.array(params))
            if obs_rms_file is not None:
                state_dict[MLP_PARAM_DICT[variable.name]] = params
            elif use_rnn:
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
                
            else:
                state_dict[MLP_PARAM_DICT_2[variable.name]] = params

        if obs_rms_file is not None:
            obs_rms = load_from_file(f"{folder}/{obs_rms_file}")
            state_dict["ob_rms.sum"] = torch.tensor(obs_rms.mean)
            state_dict["ob_rms.sumsq"] = torch.tensor(obs_rms.var + np.square(obs_rms.mean))
            state_dict["ob_rms.count"] = torch.tensor(1)

        if ret_rms_file is not None:
            ret_rms = load_from_file(f"{folder}/{ret_rms_file}")
            state_dict["critic.ret_rms.sum"] = torch.tensor(ret_rms.mean)
            state_dict["critic.ret_rms.sumsq"] = torch.tensor(ret_rms.var + np.square(ret_rms.mean))
            state_dict["critic.ret_rms.count"] = torch.tensor(1)
        elif base_params is not None:
            state_dict["critic.ret_rms.sum"] = base_params["critic.ret_rms.sum"]
            state_dict["critic.ret_rms.sumsq"] = base_params["critic.ret_rms.sumsq"]
            state_dict["critic.ret_rms.count"] = base_params["critic.ret_rms.count"]

        torch_policy.load_state_dict(state_dict)

        obs = np.random.random((n_envs*n_steps, torch_policy.ob_dim))
        states = np.random.random((n_envs, 4, 128))
        masks = np.random.random((n_envs*n_steps,)) > 0.8
        if use_rnn:
            a1, v1, s1, n1 = torch_policy.step(obs, states, masks, deterministic=True)
        else:
            a1, v1, s1, n1 = torch_policy.step(obs, deterministic=True)
        
        if obs_rms_file is not None:
            var = np.clip(obs_rms.var[None,:], 1e-2, None)
            obs = np.clip((obs - obs_rms.mean[None,:]) / np.sqrt(var), -5, 5)
        else:
            states = [states[:, i, :] for i in range(4)]
        a2, v2, s2, n2 = tf_policy.step(obs, states, masks, deterministic=True)
        if obs_rms_file is None and s2 is not None:
            s2 = np.swapaxes(s2, 0, 1)
        
        check_equal(a1, a2)
        check_equal(v1, v2)
        check_equal(s1, s2)

    torch_path = f"{folder}/model.pt"
    torch_path = torch_path.replace("_tf_", "_torch_")
    os.makedirs(torch_path.rsplit("/", 1)[0], exist_ok=True)
    torch.save(torch_policy.state_dict(), torch_path)


def _convert(env, zoo_path, norm_data, base_params):
    print("zoo_path:", zoo_path)
    tf_base_path, p_type, nn_type = norm_data
    file_names = os.listdir(zoo_path)
    model_file = "model.pkl" if "model.pkl" in file_names else "model.npy"
    obs_rms_file = "obs_rms.pkl" if "obs_rms.pkl" in file_names else None
    ret_rms_file = "ret_rms.pkl" if "ret_rms.pkl" in file_names else None
    convert_tf_to_torch(env, zoo_path, model_file, obs_rms_file, ret_rms_file, tf_base_path=tf_base_path, p_type=p_type, nn_type=nn_type, base_params=base_params)


def _convert_2(env, zoo_path, norm_data):
    print("zoo_path:", zoo_path)
    tf_base_path, p_type, nn_type = norm_data
    for folder in os.listdir(zoo_path):
        sub_zoo_path = f"{zoo_path}/{folder}"
        file_names = os.listdir(sub_zoo_path)
        model_file = [name for name in file_names if name.endswith(".npy")]
        if len(model_file) == 0:
            model_file = [name for name in file_names if name.endswith(".pkl")]
        model_file = model_file[0]
        convert_tf_to_torch(env, sub_zoo_path, model_file, tf_base_path=tf_base_path, p_type=p_type, nn_type=nn_type)


def convert(env_name):
    print("env_name:", env_name)
    env = gym.make(env_name)
    env_name_1 = ENV_NAMES_1[env_name]
    env_name_2 = ENV_NAMES_2[env_name]
    base_params = torch.load("agents/" + MODEL_BASE[env_name])
    _convert(env, f"agents/agent_zoo_tf_v2/adv-agent/ucb/{env_name_1}", NORM_ADV_PAPER1[env_name], base_params)
    _convert(env, f"agents/agent_zoo_tf_v2/adv-agent/our_attack/{env_name_1}", NORM_ADV_PAPER2[env_name], base_params)
    _convert_2(env, f"agents/agent_zoo_tf_v2/retrained-victim/our_attack/{env_name_2}", NORM_VIC_PAPER1[env_name])
    _convert_2(env, f"agents/agent_zoo_tf_v2/retrained-victim/ucb/{env_name_2}", NORM_VIC_PAPER2[env_name])


if __name__ == "__main__":
    for env_name in ENV_NAMES_1:
        convert(env_name)
