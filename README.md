# Adversarial Policy Imitation Learning in Two-player Competitive Games

### Requirements
- `pip install gym==0.15.4`
- `pip install git+https://github.com/AdamGleave/mujoco-py.git@mj131`
- `pip install --no-deps git+https://github.com/HumanCompatibleAI/multiagent-competition.git`

### Step 0: Convert pre-trained TF agents from previous works to Pytorch
- Download EC pre-trained agents from <a href="https://bit.ly/3WisyX4">here</a> and unzip to `./agents/agent_zoo_tf_v1`
- Download ADRL/APL pre-trained agents from <a href="https://bit.ly/3ffz1RR">here</a> and unzip to `./agents/agent_zoo_tf_v2`
- `python -m agents.convert_zoo_agents_v1`
- `python -m agents.convert_zoo_agents_v2`

### Step 1: Train adversary against baseline victim
- `python -m run_adv`

### Step 2: Retrain victim against new adversary
- `python -m run_vic`

### Step 3: Evaluation and Analysis
- `python -m visualize.run_eval`
- `python -m visualize.analyze`
- `python -m visualize.analyze --blind-vic`
- `python -m visualize.analyze --blind-adv`

### Step 4: Visualization
- `python -m visualize.run_tsne`
- `python -m visualize.create_graph`
- `python -m visualize.plot_tsne`
- `python -m visualize.generate_video`