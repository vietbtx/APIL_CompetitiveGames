import os
import torch
import argparse
import traceback
import numpy as np
from joblib import Memory
from trainer.utils import ENV_NAMES
import plotly.graph_objects as go
try:
    from tsnecuda import TSNE
except:
    from sklearn.manifold import TSNE
memory = Memory('__pycache__', verbose=0)


def load_file(path, max_len=10000, seed=0):
    file_data = torch.load(path, "cpu").detach().numpy()
    if len(file_data.shape) == 3:
        file_data = file_data.squeeze(1)
    np.random.seed(seed)
    np.random.shuffle(file_data)
    file_data = file_data[:max_len,:]
    return file_data


@memory.cache
def fit_tsne(dir, env_name):
    activation_paths = get_activation_paths(dir, env_name)
    sub_data = []
    for name, path in activation_paths:
        file_data = load_file(path)
        print("name:", name, file_data.shape)
        sub_data.append(file_data)
    sub_data = np.concatenate(sub_data)
    tsne_obj = TSNE(n_components=2, verbose=1, perplexity=250, n_iter=500)
    tsne_ids = tsne_obj.fit_transform(sub_data)
    return tsne_ids


def plot_graph(dir, env_name, cluster_ids, output_dir):
    colors = {"EC": "#2ba02b", "ADRL": '#1f9cb4', "APL": '#1f77b4', "APIL": "#ff7f0f", "E-APIL": "#ff430f"}
    opponents = {}
    activation_paths = get_activation_paths(dir, env_name)
    for name, path in activation_paths:
        file_data = load_file(path)
        n = file_data.shape[0]
        opponents[name] = {
            "pos": cluster_ids[:n],
            "color": colors[name]
        }
        cluster_ids = cluster_ids[n:]

    fig = go.Figure()
    fig.update_layout(width=420, height=280)
    fig.update_layout(template='plotly_white', margin=dict(l=0, r=0, t=0, b=0, pad=0, autoexpand=True))

    for name in ['EC', 'ADRL', 'APL', 'APIL', 'E-APIL']:
        opp = opponents[name]
        ids = opp["pos"]
        fig.add_trace(go.Scattergl(x=ids[:, 0], y=ids[:, 1], mode='markers', line_color=opp["color"], name=name, marker_size=1, marker_opacity=0.8))
    os.makedirs(f'{output_dir}', exist_ok=True)
    os.makedirs(f'{output_dir}/{env_name}', exist_ok=True)

    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=0.98, xanchor="right", x=0.95, itemsizing="constant"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    fig.write_image(f'{output_dir}/{env_name}/tsne_legend.pdf', scale=1.0)
    fig.update_layout(showlegend=False)
    fig.write_image(f'{output_dir}/{env_name}/tsne.pdf', scale=1.0)


def get_activation_paths(dir, env_name):
    activation_paths = {
        'EC': f'{dir}/tsne_{env_name}_seed_0_mlp_imitation.pkl',
        'ADRL': f'{dir}/tsne_{env_name}_seed_0_mlp_imitation_retrained_adv_adv_paper_1.pkl',
        'APL': f'{dir}/tsne_{env_name}_seed_0_mlp_imitation_retrained_adv_adv_paper_2.pkl',
        'APIL': f'{dir}/tsne_{env_name}_seed_0_mlp_imitation_retrained_adv.pkl',
        'E-APIL': f'{dir}/tsne_{env_name}_seed_0_mlp_imitation_enhance_adv_retrained_adv.pkl',
    }
    if env_name == "SumoAnts-v0":
        activation_paths = {k: v.replace("imitation", "imitation_l2t") for k, v in activation_paths.items()}
    activation_paths = [(k, activation_paths[k]) for k in sorted(activation_paths.keys())]
    return activation_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default='./logs')
    parser.add_argument("--output_dir", type=str, default='./graphs')
    args = parser.parse_args()

    for env_name in ENV_NAMES:
        try:
            print(f"Running ... {env_name}")
            cluster_ids = fit_tsne(args.dir, env_name)
            plot_graph(args.dir, env_name, cluster_ids, args.output_dir)
        except:
            traceback.print_exc()