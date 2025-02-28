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

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)
LOG_FILE_P3 = os.path.join("checkpoints", "log_p3.json")
RESULTS_FILE_P3 = os.path.join("results", "plot_data_p3.pkl")
CHECKPOINT_INTERVAL = 50
NUM_EPISODES = 1000
NUM_SEEDS = 50
GAMMA = 0.99
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_NAMES = ["Acrobot-v1", "ALE/Assault-ram-v5"]
METHODS_P3 = ["reinforce", "actor_critic"]
TEMP_MODES = ["fixed", "decreasing"]
T_FIXED = 1.0
LR_POLICY = 1e-3
LR_VALUE = 1e-3

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
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

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
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

def preprocess_state(state, env_name):
    if env_name == "ALE/Assault-ram-v5":
        return np.array(state, dtype=np.float32)/255.0
    else:
        return np.array(state, dtype=np.float32)

def get_temperature(temp_mode, T_fixed, episode, num_episodes):
    if temp_mode == "fixed":
        return T_fixed
    else:
        return T_fixed * (1 - episode/num_episodes) if episode < num_episodes else T_fixed*0.01

def get_experiment_key_p3(env_name, method, temp_mode, seed):
    return f"{env_name}_{method}_{temp_mode}_seed{seed}"

def load_log_p3():
    if os.path.exists(LOG_FILE_P3):
        with open(LOG_FILE_P3, "r") as f:
            return json.load(f)
    return {}

def save_log_p3(log_dict):
    with open(LOG_FILE_P3, "w") as f:
        json.dump(log_dict, f)

def load_results_p3():
    if os.path.exists(RESULTS_FILE_P3):
        with open(RESULTS_FILE_P3, "rb") as f:
            return pickle.load(f)
    return {}

def save_results_p3(results_dict):
    with open(RESULTS_FILE_P3, "wb") as f:
        pickle.dump(results_dict, f)

def get_checkpoint_filename_p3(exp_key):
    return os.path.join("checkpoints", f"{exp_key}_p3.pt")

def train_one_experiment_p3(env_name, method, temp_mode, seed):
    exp_key = get_experiment_key_p3(env_name, method, temp_mode, seed)
    ckpt_file = get_checkpoint_filename_p3(exp_key)
    env = gym.make(env_name)
    state, _ = env.reset(seed=seed)
    state = preprocess_state(state, env_name)
    obs_shape = env.observation_space.shape
    state_dim = obs_shape[0] if len(obs_shape)==1 else int(np.prod(obs_shape))
    action_dim = env.action_space.n
    policy_net = PolicyNetwork(state_dim, action_dim).to(DEVICE)
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=LR_POLICY)
    if method == "actor_critic":
        value_net = ValueNetwork(state_dim).to(DEVICE)
        optimizer_value = optim.Adam(value_net.parameters(), lr=LR_VALUE)
    start_ep = 0
    episode_rewards = []
    if os.path.exists(ckpt_file):
        ckpt = torch.load(ckpt_file, map_location=DEVICE)
        policy_net.load_state_dict(ckpt["policy_state"])
        optimizer_policy.load_state_dict(ckpt["optimizer_policy_state"])
        if method == "actor_critic":
            value_net.load_state_dict(ckpt["value_state"])
            optimizer_value.load_state_dict(ckpt["optimizer_value_state"])
        start_ep = ckpt["episode"] + 1
        episode_rewards = ckpt["episode_rewards"]
    for ep in range(start_ep, NUM_EPISODES):
        T = get_temperature(temp_mode, T_FIXED, ep, NUM_EPISODES)
        state, _ = env.reset(seed=seed+ep)
        state = preprocess_state(state, env_name)
        log_probs = []
        rewards = []
        values = []
        done = False
        total_reward = 0
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            logits = policy_net(state_tensor)
            dist = torch.distributions.Categorical(logits=logits/T)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action).to(DEVICE))
            if method == "actor_critic":
                value = value_net(state_tensor)
                values.append(value)
            log_probs.append(log_prob)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            rewards.append(reward)
            state = preprocess_state(next_state, env_name)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(DEVICE)
        if method == "reinforce":
            policy_loss = 0
            for log_prob, G in zip(log_probs, returns):
                policy_loss += -log_prob * G
            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()
        elif method == "actor_critic":
            policy_loss = 0
            value_loss = 0
            values = torch.cat(values).squeeze()
            advantages = returns - values
            for log_prob, adv in zip(log_probs, advantages):
                policy_loss += -log_prob * adv.detach()
            value_loss = nn.MSELoss()(values, returns)
            optimizer_policy.zero_grad()
            optimizer_value.zero_grad()
            (policy_loss + value_loss).backward()
            optimizer_policy.step()
            optimizer_value.step()
        episode_rewards.append(total_reward)
        if (ep+1) % CHECKPOINT_INTERVAL == 0 or (ep+1) == NUM_EPISODES:
            ckpt = {"episode": ep, "episode_rewards": episode_rewards, "policy_state": policy_net.state_dict(), "optimizer_policy_state": optimizer_policy.state_dict()}
            if method == "actor_critic":
                ckpt["value_state"] = value_net.state_dict()
                ckpt["optimizer_value_state"] = optimizer_value.state_dict()
            torch.save(ckpt, ckpt_file)
            log_dict = load_log_p3()
            log_dict[exp_key] = ep
            save_log_p3(log_dict)
            results_dict = load_results_p3()
            results_dict[exp_key] = episode_rewards
            save_results_p3(results_dict)
    env.close()
    return episode_rewards

def generate_experiment_list_p3():
    exp_list = []
    for env_name in ENV_NAMES:
        for method in METHODS_P3:
            for temp_mode in TEMP_MODES:
                for seed in range(NUM_SEEDS):
                    exp_list.append((env_name, method, temp_mode, seed))
    return exp_list

def run_experiments_p3(exp_list):
    log_dict = load_log_p3()
    for params in exp_list:
        env_name, method, temp_mode, seed = params
        exp_key = get_experiment_key_p3(env_name, method, temp_mode, seed)
        last_ep = log_dict.get(exp_key, -1)
        if last_ep >= NUM_EPISODES - 1:
            continue
        rewards = train_one_experiment_p3(env_name, method, temp_mode, seed)
        log_dict[exp_key] = NUM_EPISODES - 1
        save_log_p3(log_dict)
        results_dict = load_results_p3()
        results_dict[exp_key] = rewards
        save_results_p3(results_dict)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "continue", "plot"], required=True)
args = parser.parse_args()
if args.mode == "train":
    exp_list = generate_experiment_list_p3()
    run_experiments_p3(exp_list)
elif args.mode == "continue":
    log_dict = load_log_p3()
    exp_list = generate_experiment_list_p3()
    incomplete = []
    for params in exp_list:
        exp_key = get_experiment_key_p3(*params)
        last_ep = log_dict.get(exp_key, -1)
        if last_ep < NUM_EPISODES - 1:
            incomplete.append(params)
    run_experiments_p3(incomplete)
elif args.mode == "plot":
    results_dict = load_results_p3()
    for env_name in ENV_NAMES:
        for temp_mode in TEMP_MODES:
            plt.figure(figsize=(10,6))
            for method, color, ls in zip(METHODS_P3, ['green','red'], ['solid','dashed']):
                all_runs = []
                for seed in range(NUM_SEEDS):
                    exp_key = get_experiment_key_p3(env_name, method, temp_mode, seed)
                    if exp_key in results_dict:
                        all_runs.append(np.array(results_dict[exp_key]))
                if len(all_runs) == 0:
                    continue
                all_runs = np.array(all_runs)
                mean_rewards = all_runs.mean(axis=0)
                std_rewards = all_runs.std(axis=0)
                episodes = np.arange(1, NUM_EPISODES+1)
                plt.plot(episodes, mean_rewards, label=f"{method}", color=color, linestyle=ls)
                plt.fill_between(episodes, mean_rewards-std_rewards, mean_rewards+std_rewards, color=color, alpha=0.2)
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title(f"{env_name} | Temperature Mode: {temp_mode}")
            plt.legend()
            plt.grid(True)
            plt.show()
