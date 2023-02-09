# Adversarial Policy Imitation Learning in Two-player Competitive Games

### Introduction
Recent research on vulnerabilities of deep reinforcement learning (RL) has shown that adversarial policies adopted by an adversary agent can influence a target RL agent (victim agent) to perform poorly in a multi-agent environment. In existing studies, adversarial policies are directly trained based on experiences of interacting with the victim agent. There is a key shortcoming of this approach -  knowledge derived from historical interactions may not be properly generalized to unexplored policy regions of the victim agent, making the trained adversarial policy significantly less effective. In this work, we design a new effective adversarial policy learning algorithm that overcomes this shortcoming. The core idea of our new algorithm is to create a new imitator - the imitator will learn to imitate the victim agent's policy while the adversarial policy will be trained not only based on interactions with the victim agent but also based on feedback from the imitator to forecast victim's intention. By doing so, we can leverage the capability of imitation learning in well capturing underlying characteristics of the victim policy only based on sample trajectories of the victim. 
Our victim imitation learning model differs from prior models as the environment's dynamics are driven by adversary's policy and will keep changing during the adversarial policy training. We provide a provable bound to guarantee a desired imitating policy when the adversary's policy becomes stable. 
We further strengthen our adversarial policy learning by making our imitator a stronger version of the victim. That is, we incorporate the opposite of the adversary's value function to the imitation objective, leading the imitator not only to learn the victim policy but also to be adversarial to the adversary. Finally, our extensive experiments using four competitive MuJoCo game environments show that our proposed adversarial policy learning algorithm outperforms state-of-the-art algorithms. 


### Requirements
- `pip install gym==0.15.4`
- `pip install git+https://github.com/AdamGleave/mujoco-py.git@mj131`
- `pip install --no-deps git+https://github.com/HumanCompatibleAI/multiagent-competition.git`

### Code structure
    ├── agents
    │   ├── agent_policy_pytorch.py         # Our policy (PyTorch)
    │   ├── agent_policy_tf_v1.py           # EC policy (TF)
    │   ├── agent_policy_tf_v2.py           # ADRL/APL policy (TF)
    │   ├── convert_zoo_agents_v1.py        # Convert EC to PyTorch
    │   ├── convert_zoo_agents_v2.py        # Convert ADRL/APL to PyTorch
    │   └── zoo_utils.py
    ├── env
    │   ├── config.py                       # Configuration
    │   ├── env.py                          # Compete environment
    │   └── scheduling.py
    ├── run_adv.py                          # Parallel training adversary
    ├── run_vic.py                          # Parallel training victim
    ├── trainer
    │   ├── buffer.py                       # Replay buffer
    │   ├── ppo.py                          # PPO algorithm
    │   ├── runner.py                       # Collect transitions
    │   ├── train_adv.py                    # Adversary trainer
    │   ├── train_vic.py                    # Victim trainer
    │   └── utils.py
    └── visualize
        ├── analyze.py                      # Analyze results
        ├── create_graph.py                 # Draw improvement graphs
        ├── evaluate.py                     # Evaluation
        ├── generate_video.py               # Record simulation
        ├── plot_tsne.py                    # Draw t-SNE figures
        ├── run_eval.py                     # Parallel evaluation
        ├── run_tsne.py                     # Create t-SNE data
        └── video.py                        # Mujoco video wrapper

### Step 0: Convert pre-trained TF agents from previous works to PyTorch
- Download EC pre-trained agents from <a href="https://bit.ly/3WisyX4">here</a> and unzip to `./agents/agent_zoo_tf_v1`
- Download ADRL/APL pre-trained agents from <a href="https://bit.ly/3ffz1RR">here</a> and unzip to `./agents/agent_zoo_tf_v2`
- `python -m agents.convert_zoo_agents_v1`
- `python -m agents.convert_zoo_agents_v2`

### Step 1: Train adversary against baseline victim
- `python -m run_adv`

We train our adversary agent using our proposed algorithms to play against the baseline victim agent. We aim to examine if our generated adversarial policy can trigger the victim agent to perform poorly.

### Step 2: Retrain victim against new adversary
- `python -m run_vic`

We retrain the victim agent against the newly trained adversary agent to examine the resistance of the retrained victim agent against adversarial policies. We further explore the resilience transferability of the retrained victim agents. Specifically, similar to previous work, we retrain the victim agent against a mixed adversary agent of the new adversary (whose policy is trained based on one of the evaluated adversarial training algorithms (i.e., baseline, ADRL, APL and ours) and the baseline adversary.

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