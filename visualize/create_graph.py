
import os
import numpy as np
import plotly.io as pio
from scipy import signal
from joblib import Memory
from trainer.utils import ENV_NAMES
from multiprocessing import Pool
import plotly.graph_objects as go
from tensorboard.backend.event_processing import event_accumulator
memory = Memory('__pycache__', verbose=0)

try:
    pio.kaleido.scope.mathjax = None
except:
    pass


def find_file(folder, s=None, e=None):
    for file in os.listdir(folder):
        is_correct = True
        if s is not None: is_correct = is_correct and file.startswith(s)
        if e is not None: is_correct = is_correct and file.endswith(e)
        if is_correct:
            return f"{folder}/{file}"


def make_smooth(value, window_length=53, polyorder=3):
    smooth_value = signal.savgol_filter(value, window_length, polyorder)
    return smooth_value


def read_scalars(ea, key):
    step, value = zip(*[[x.step, x.value] for x in ea.Scalars(key)])
    step = np.array(step)
    value = np.array(value)
    return step, value


@memory.cache
def read_tensorboard(path, game_ori=False):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    win_step, win_value = read_scalars(ea, 'game_ori/win' if game_ori else 'game/win')
    lose_step, lose_value = read_scalars(ea, 'game_ori/lose' if game_ori else 'game/lose')
    return win_step, win_value, lose_step, lose_value


def make_upper_lower(value):
    window_length = int(len(value)*0.05)
    upper = [value[i] + np.std(value[i:i+window_length]) for i in range(len(value))]
    lower = [value[i] - np.std(value[i:i+window_length]) for i in range(len(value))]
    upper = [min(x, 1) for x in upper]
    lower = [max(x, 0) for x in lower]
    upper = make_smooth(upper, 27)
    lower = make_smooth(lower, 27)
    return upper, lower


def hex_to_rgb(hex_color: str, opacity=1.0):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba{(r, g, b, opacity)}"


def plot(V1_x, V1_y, V2_x, V2_y, max_x=35e6, max_y=1):
    colors = ['#2385ca', '#ff430f', '#91c2e4', '#ff8e6f', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    V1_y_smooth = make_smooth(V1_y)
    V2_y_smooth = make_smooth(V2_y)

    V1_y_upper, V1_y_lower = make_upper_lower(V1_y)
    V2_y_upper, V2_y_lower = make_upper_lower(V2_y)

    fig = go.Figure()
    fig.update_layout(width=280, height=210)

    fig.add_trace(go.Scatter(x=V1_x, y=V1_y_upper, mode='lines', line_width=0, showlegend=False))
    fig.add_trace(go.Scatter(x=V1_x, y=V1_y_lower, line_width=0, mode='lines', fillcolor=hex_to_rgb(colors[2], 0.5), fill='tonexty', showlegend=False))

    fig.add_trace(go.Scatter(x=V2_x, y=V2_y_upper, mode='lines', line_width=0, showlegend=False))
    fig.add_trace(go.Scatter(x=V2_x, y=V2_y_lower, line_width=0, mode='lines', fillcolor=hex_to_rgb(colors[3], 0.5), fill='tonexty', showlegend=False))

    fig.add_trace(go.Scatter(x=V1_x, y=V1_y_smooth, mode='lines', line_width=4, line_color=colors[0], name="APIL"))
    fig.add_trace(go.Scatter(x=V2_x, y=V2_y_smooth, mode='lines', line_width=4, line_color=colors[1], name="E-APIL"))

    max_V1_y = V1_y_smooth[-1]
    max_V2_y = V2_y_smooth[-1]

    if max_V1_y > 0.1:
        fig.add_trace(go.Scatter(x=[0, max_x], y=[max_V1_y, max_V1_y], mode='lines', line_width=4, line_color=colors[0], line_dash='dash', showlegend=False, opacity=0.5))
    if max_V2_y > 0.1:
        fig.add_trace(go.Scatter(x=[0, max_x], y=[max_V2_y, max_V2_y], mode='lines', line_width=4, line_color=colors[1], line_dash='dash', showlegend=False, opacity=0.9))
    fig.add_trace(go.Scatter(yaxis='y2'))
    fig.add_trace(go.Scatter(yaxis='y3'))

    ticksx = [0, 15e6, 25e6, 35e6] if max_x > 10e6 else [0, 5e6, 10e6]
    
    min_y = min(min(V1_y_smooth), min(V2_y_smooth), 0.75)
    max_y = max(max(V1_y_smooth), max(V2_y_smooth), 0.25)
    min_y = max(np.floor(min_y*8)/8, 0)
    max_y = min(np.ceil(max_y*8)/8, 1)
    delta_y = max_y - min_y
    min_y -= 0.05 * delta_y
    max_y += 0.05 * delta_y
    tickvals_y = [0.25, 0.5, 0.75, 1.0]
    ticktext_y = [f"{x*100:.0f}" if abs(x-max_V1_y)/delta_y > 0.1 and abs(x-max_V2_y)/delta_y > 0.1 and min_y < x < max_y else "" for x in tickvals_y]
    if all(len(x) > 0 for x in ticktext_y[:3]):
        ticktext_y[0] = ""

    fig.update_xaxes(tickvals=ticksx, range=[0, max_x])
    fig.update_layout(template='plotly_white', margin=dict(l=4, r=4, t=4, b=4, pad=4, autoexpand=True), 
        yaxis=dict(
            range=[min_y, max_y],
            tickmode='array',
            tickvals=tickvals_y,
            ticktext=ticktext_y,
        ),
        yaxis2=dict(
            range=[min_y, max_y],
            tickfont=dict(color=colors[0]),
            tickmode='array',
            tickvals=[max_V1_y],
            ticktext=[f"{100*max_V1_y:.0f}" if abs(max_V1_y-max_V2_y)/delta_y > 0.08 and max_V1_y > 0.1 else ""],
            overlaying="y",
            side="left",
        ),
        yaxis3=dict(
            range=[min_y, max_y],
            tickfont=dict(color=colors[1]),
            tickmode='array',
            tickvals=[max_V2_y],
            ticktext=[f"<b>{100*max_V2_y:.0f}</b>" if max_V2_y > 0.1 else ""],
            overlaying="y",
            side="left",
        )
    )
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.5,
        xanchor="left",
        x=0.6,
        traceorder="normal"
    ), font_size=16)
    return fig


def save_fig(fig, path):
    print("Saving ...", path)
    fig.write_image(f"graphs/{path}_legend.pdf", scale=2.0)
    fig.update_layout(showlegend=False)
    fig.write_image(f"graphs/{path}.pdf", scale=2.0)
    

def create_graph(env_name):
    print(env_name)
    os.makedirs("graphs", exist_ok=True)
    os.makedirs(f"graphs/{env_name}", exist_ok=True)
    
    log_folder = f"logs/{env_name}"
    adv_V1 = read_tensorboard(find_file(find_file(log_folder, e="imitation"), s="events"))
    adv_V2 = read_tensorboard(find_file(find_file(log_folder, e="enhance"), s="events"))
    
    vic_V1 = read_tensorboard(find_file(find_file(log_folder, e="imitation_robust_victim"), s="events"))
    vic_V2 = read_tensorboard(find_file(find_file(log_folder, e="enhance_robust_victim"), s="events"))

    vic_ori_V1 = read_tensorboard(find_file(find_file(log_folder, e="imitation_robust_victim"), s="events"), game_ori=True)
    vic_ori_V2 = read_tensorboard(find_file(find_file(log_folder, e="enhance_robust_victim"), s="events"), game_ori=True)

    fig = plot(adv_V1[0], adv_V1[1], adv_V2[0], adv_V2[1], max_x=35e6)
    save_fig(fig, f"{env_name}/adv_win")

    fig = plot(adv_V1[2], 1-adv_V1[3], adv_V2[2], 1-adv_V2[3], max_x=35e6)
    save_fig(fig, f"{env_name}/adv_win+tie")

    fig = plot(vic_V1[0], vic_V1[1], vic_V2[0], vic_V2[1], max_x=10e6)
    save_fig(fig, f"{env_name}/vic_win")
    
    fig = plot(vic_V1[2], 1-vic_V1[3], vic_V2[2], 1-vic_V2[3], max_x=10e6)
    save_fig(fig, f"{env_name}/vic_win+tie")

    fig = plot(vic_ori_V1[0], vic_ori_V1[1], vic_ori_V2[0], vic_ori_V2[1], max_x=10e6)
    save_fig(fig, f"{env_name}/vic_ori_win")
    
    fig = plot(vic_ori_V1[2], 1-vic_ori_V1[3], vic_ori_V2[2], 1-vic_ori_V2[3], max_x=10e6)
    save_fig(fig, f"{env_name}/vic_ori_win+tie")


def main():
    with Pool() as p:
        p.map(create_graph, ENV_NAMES)


if __name__ == "__main__":
    main()