import os
import json
import argparse
import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)
LOG_FILE = os.path.join("checkpoints", "log.json")
RESULTS_FILE = os.path.join("results", "plot_data.pkl")
CHECKPOINT_INTERVAL = 50
NUM_EPISODES = 1000
NUM_SEEDS = 50
GAMMA = 0.99
REPLAY_BUFFER_CAPACITY = int(1e6)
MINI_BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_NAMES = ["Acrobot-v1", "ALE/Assault-ram-v5"]
USE_REPLAY_OPTIONS = [False, True]
EPSILON_LIST = [1.0, 0.1, 0.01]
LR_LIST = [1/4, 1/8, 1/16]
METHODS = ["q_learning", "expected_sarsa"]

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-0.001, b=0.001)
                nn.init.uniform_(m.bias, a=-0.001, b=0.001)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones
    def __len__(self):
        return len(self.buffer)

def preprocess_state(state, env_name):
    if env_name=="ALE/Assault-ram-v5":
        return np.array(state, dtype=np.float32)/255.0
    else:
        return np.array(state, dtype=np.float32)

def select_action(q_net, state, epsilon, action_dim):
    if random.random()<epsilon:
        return random.randint(0, action_dim-1)
    else:
        state_tensor=torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values=q_net(state_tensor)
        return q_values.argmax(dim=1).item()

def expected_q(q_net, next_state, epsilon, action_dim):
    state_tensor=torch.FloatTensor(next_state).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        q_vals=q_net(state_tensor).squeeze(0)
    max_q=q_vals.max().item()
    expected=0.0
    for a in range(action_dim):
        prob=epsilon/action_dim
        if q_vals[a].item()==max_q:
            prob+=1-epsilon
        expected+=prob*q_vals[a].item()
    return expected

def get_experiment_key(env_name, use_replay, epsilon, lr, method, seed):
    return f"{env_name}_{use_replay}_{epsilon}_{lr}_{method}_seed{seed}"

def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return {}

def save_log(log_dict):
    with open(LOG_FILE, "w") as f:
        json.dump(log_dict, f)

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_results(results_dict):
    with open(RESULTS_FILE, "wb") as f:
        pickle.dump(results_dict, f)

def get_checkpoint_filename(exp_key):
    return os.path.join("checkpoints", f"{exp_key}.pt")

def train_one_experiment(env_name, use_replay, epsilon, lr, method, seed):
    exp_key=get_experiment_key(env_name, use_replay, epsilon, lr, method, seed)
    ckpt_file=get_checkpoint_filename(exp_key)
    env=gym.make(env_name)
    state, _=env.reset(seed=seed)
    state=preprocess_state(state, env_name)
    obs_shape=env.observation_space.shape
    state_dim=obs_shape[0] if len(obs_shape)==1 else int(np.prod(obs_shape))
    action_dim=env.action_space.n
    q_net=QNetwork(state_dim, action_dim).to(DEVICE)
    optimizer=optim.Adam(q_net.parameters(), lr=lr)
    if use_replay:
        replay_buffer=ReplayBuffer(REPLAY_BUFFER_CAPACITY)
    start_ep=0
    episode_rewards=[]
    if os.path.exists(ckpt_file):
        ckpt=torch.load(ckpt_file, map_location=DEVICE)
        q_net.load_state_dict(ckpt["q_net_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_ep=ckpt["episode"]+1
        episode_rewards=ckpt["episode_rewards"]
    for ep in range(start_ep, NUM_EPISODES):
        state, _=env.reset(seed=seed+ep)
        state=preprocess_state(state, env_name)
        total_reward=0
        done=False
        while not done:
            action=select_action(q_net, state, epsilon, action_dim)
            next_state, reward, terminated, truncated, _=env.step(action)
            done=terminated or truncated
            next_state_proc=preprocess_state(next_state, env_name)
            if not use_replay:
                state_tensor=torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                q_val=q_net(state_tensor)[0, action]
                if done:
                    target=torch.FloatTensor([reward]).to(DEVICE)
                else:
                    if method=="q_learning":
                        next_state_tensor=torch.FloatTensor(next_state_proc).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            target_val=q_net(next_state_tensor).max().item()
                        target=torch.FloatTensor([reward+GAMMA*target_val]).to(DEVICE)
                    elif method=="expected_sarsa":
                        if done:
                            target=torch.FloatTensor([reward]).to(DEVICE)
                        else:
                            exp_val=expected_q(q_net, next_state_proc, epsilon, action_dim)
                            target=torch.FloatTensor([reward+GAMMA*exp_val]).to(DEVICE)
                loss=(q_val-target).pow(2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                replay_buffer.push(state, action, reward, next_state_proc, done)
                if len(replay_buffer)>=MINI_BATCH_SIZE:
                    states_b, actions_b, rewards_b, next_states_b, dones_b=replay_buffer.sample(MINI_BATCH_SIZE)
                    states_b=torch.FloatTensor(states_b).to(DEVICE)
                    actions_b=torch.LongTensor(actions_b).unsqueeze(1).to(DEVICE)
                    rewards_b=torch.FloatTensor(rewards_b).unsqueeze(1).to(DEVICE)
                    next_states_b=torch.FloatTensor(next_states_b).to(DEVICE)
                    dones_b=torch.FloatTensor(dones_b).unsqueeze(1).to(DEVICE)
                    q_values=q_net(states_b).gather(1, actions_b)
                    with torch.no_grad():
                        if method=="q_learning":
                            next_q=q_net(next_states_b).max(1)[0].unsqueeze(1)
                            targets=rewards_b+GAMMA*next_q*(1-dones_b)
                        elif method=="expected_sarsa":
                            next_q_vals=q_net(next_states_b)
                            max_q_vals, _=next_q_vals.max(dim=1, keepdim=True)
                            probs=torch.ones_like(next_q_vals)*(epsilon/action_dim)
                            best_action_mask=(next_q_vals==max_q_vals)
                            probs[best_action_mask]+=1-epsilon
                            exp_q=(next_q_vals*probs).sum(dim=1, keepdim=True)
                            targets=rewards_b+GAMMA*exp_q*(1-dones_b)
                    loss=nn.MSELoss()(q_values, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            state=next_state_proc
            total_reward+=reward
        episode_rewards.append(total_reward)
        if (ep+1)%CHECKPOINT_INTERVAL==0 or (ep+1)==NUM_EPISODES:
            ckpt={"episode":ep, "episode_rewards":episode_rewards, "q_net_state":q_net.state_dict(), "optimizer_state":optimizer.state_dict()}
            torch.save(ckpt, ckpt_file)
            log_dict=load_log()
            log_dict[exp_key]=ep
            save_log(log_dict)
            results_dict=load_results()
            results_dict[exp_key]=episode_rewards
            save_results(results_dict)
    env.close()
    return episode_rewards

def run_experiments(exp_list):
    log_dict=load_log()
    for params in exp_list:
        env_name, use_replay, epsilon, lr, method, seed=params
        exp_key=get_experiment_key(env_name, use_replay, epsilon, lr, method, seed)
        last_ep=log_dict.get(exp_key, -1)
        if last_ep>=NUM_EPISODES-1:
            continue
        rewards=train_one_experiment(env_name, use_replay, epsilon, lr, method, seed)
        log_dict[exp_key]=NUM_EPISODES-1
        save_log(log_dict)
        results_dict=load_results()
        results_dict[exp_key]=rewards
        save_results(results_dict)

def generate_experiment_list():
    exp_list=[]
    for env_name in ENV_NAMES:
        for use_replay in USE_REPLAY_OPTIONS:
            for eps in EPSILON_LIST:
                for lr in LR_LIST:
                    for method in METHODS:
                        for seed in range(NUM_SEEDS):
                            exp_list.append((env_name, use_replay, eps, lr, method, seed))
    return exp_list

parser=argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "continue", "plot"], required=True)
args=parser.parse_args()
if args.mode=="train":
    exp_list=generate_experiment_list()
    run_experiments(exp_list)
elif args.mode=="continue":
    log_dict=load_log()
    exp_list=generate_experiment_list()
    incomplete=[]
    for params in exp_list:
        exp_key=get_experiment_key(*params)
        last_ep=log_dict.get(exp_key, -1)
        if last_ep<NUM_EPISODES-1:
            incomplete.append(params)
    run_experiments(incomplete)
elif args.mode == "plot":
    results_dict = load_results()

    def pad_run(run, target_length):
        if len(run) < target_length:
            run = run + [run[-1]] * (target_length - len(run))
        return run

    for env_name in ENV_NAMES:
        for use_replay in USE_REPLAY_OPTIONS:
            for eps in EPSILON_LIST:
                for lr in LR_LIST:
                    plt.figure(figsize=(10, 6))
                    for method, color, ls in zip(METHODS, ['green', 'red'], ['solid', 'dashed']):
                        all_runs = []
                        for seed in range(NUM_SEEDS):
                            exp_key = get_experiment_key(env_name, use_replay, eps, lr, method, seed)
                            if exp_key in results_dict:
                                run = results_dict[exp_key]
                                run = pad_run(run, NUM_EPISODES)
                                all_runs.append(np.array(run))
                        if len(all_runs) == 0:
                            continue
                        all_runs = np.stack(all_runs)
                        mean_rewards = all_runs.mean(axis=0)
                        std_rewards = all_runs.std(axis=0)
                        episodes = np.arange(1, NUM_EPISODES + 1)
                        plt.plot(episodes, mean_rewards, label=f"{method}", color=color, linestyle=ls)
                        plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, color=color,
                                         alpha=0.2)
                    replay_str = "With Replay Buffer" if use_replay else "Without Replay Buffer"
                    plt.xlabel("Episode")
                    plt.ylabel("Total Reward")
                    plt.title(f"{env_name} | {replay_str} | ε={eps}, lr={lr}")
                    plt.legend()
                    plt.grid(True)
                    plt.show()







