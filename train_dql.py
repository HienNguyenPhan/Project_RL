import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import magent2
import os
import gymnasium as gym
from magent2.environments import battle_v4
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from Agent import Agent
from Buffer import Buffer

# Edit environment with lower attack penalty and higher attack reward
env = battle_v4.env(map_size=45, max_cycles=300, render_mode="rgb_array", step_reward=0.001,
dead_penalty=-0.1, attack_penalty=-0.05, attack_opponent_reward=0.5,extra_features=False)
env.reset()
observation_shape = env.observation_space("red_0").shape
action_shape = env.action_space("red_0").n
red_weight_path = "red.pt"
blue_weight_path = None

# Agent (red + blue)
q_network = Agent(observation_shape, action_shape)
if blue_weight_path is not None: 
    q_network.load_state_dict(torch.load(blue_weight_path, weights_only=True))
target_q_network = Agent(observation_shape, action_shape)
target_q_network.load_state_dict(q_network.state_dict())
opponent_agent = Agent(observation_shape, action_shape)
if red_weight_path is not None:
    opponent_agent.load_state_dict(torch.load(red_weight_path, weights_only=True))

# Parameter
gamma = 0.99 
batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = optim.Adam(q_network.parameters(), lr=0.001)
criterion = nn.MSELoss()
epsilon = 1
epsilon_decay = 0.99
epsilon_min = 0.1
tau = 0.95
max_episodes = 300

# Switch to cuda
q_network.to(device)
target_q_network.to(device)
opponent_agent.to(device)

relay_buffer = Buffer(observation_shape, action_shape, capacity= 200000)


frames = []
env.reset()
prev_data = defaultdict(list)
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    agent_team = agent.split('_')[0]
    done = False
    if termination or truncation:
        action = None 
        done = True
    else:
        action = env.action_space(agent).sample()
    
    try:
        if agent in prev_data:
            relay_buffer.add((prev_data[agent][0], prev_data[agent][1], observation, reward, done))
    except Exception as e:
        print(prev_data[agent])
        print(e)
        break
    
    if termination or truncation:
        prev_data[agent] = None
    else:
        prev_data[agent] = [observation, action]
    
    env.step(action) 
    

def get_q_value(agent: nn.Module, observation: torch.Tensor, batch_size: int= 1024, device: torch.device= device):
    if len(observation.shape) == 3:
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32, device=device).permute(2, 0, 1)
        else:
            observation = observation.clone().detach().requires_grad_(False).permute(2, 0, 1)
        q_values = agent(observation)
    else:
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        else:
            observation = observation.clone().detach().requires_grad_(False).permute(0, 3, 1, 2)
        num_batches = (observation.shape[0] + batch_size - 1) // batch_size
        q_values = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, observation.shape[0])
            batch_observation = observation[start_idx:end_idx]
            batch_observation = batch_observation.to(device)
            batch_q_values = agent(batch_observation)
            q_values.append(batch_q_values)
        
        q_values = torch.cat(q_values)
    return q_values

def predict_action(agent: nn.Module, observation: torch.Tensor, batch_size: int= 1024):
    with torch.no_grad():
        agent.eval()
        q_values = get_q_value(agent, observation, batch_size)
        return q_values.argmax(dim=1).item()

def train(agent: nn.Module, target_agent: nn.Module, data: list):
    agent.train()
    random.shuffle(data)
    
    state, action, next_state, reward, done = zip(*data)
    state = np.array(state)
    action = np.array(action)
    next_state = np.array(next_state)
    reward = np.array(reward)
    done = np.array(done)
    
    state = torch.tensor(state, dtype=torch.float32, device=device)
    action = torch.tensor(action, dtype=torch.long, device=device)
    next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
    reward = torch.tensor(reward, dtype=torch.float32, device=device)
    done = torch.tensor(done, dtype=torch.float32, device=device) 
    
    old_q_values = get_q_value(agent, state, batch_size)      
    next_q_values = get_q_value(target_agent, next_state, batch_size)
    
    try:
        predict = old_q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
    except Exception as e:
        print(old_q_values.shape)
        print(old_q_values)
        print(action.shape)
        print(e)
        raise e
    next_max_q_value = next_q_values.max(dim=1).values
    target = reward + gamma * next_max_q_value * (1 - done)
    
    loss = criterion(predict, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def get_blue_agent_action(agent_id: str, agent: nn.Module, observation: torch.Tensor, epsilon: float, env: gym.Env):
    if np.random.rand() < epsilon:
        action = env.action_space(agent_id).sample()
    else:
        action = predict_action(agent=agent, observation=observation)
    return action

def get_red_agent_action(agent_id: str, env: gym.Env):
    return env.action_space(agent_id).sample()

pbar = tqdm(range(max_episodes), position=0, leave=True, desc="Training")
relay_buffer.reset()
for episode in pbar:
    
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    
    env.reset()
    
    rewards = [0, 0]
    cycle_count = 0
    last_agent_team = None
    
    prev_data = defaultdict(list)
    for agent_id in env.agent_iter():
        
        observation, reward, termination, truncation, info = env.last()
        agent_team = agent_id.split('_')[0]
        
        if agent_team == "blue":
            if termination or truncation:
                action = None
                done = True
            else:
                action = get_blue_agent_action(agent_id ,q_network, observation, epsilon, env)
            
            try:
                if agent_id in prev_data:
                    relay_buffer.add((prev_data[agent_id][0], prev_data[agent_id][1], observation, reward, done))
            except Exception as e:
                print(prev_data[agent_id])
                print(e)
                break
            
            if termination or truncation:
                del prev_data[agent_id]
            else:
                prev_data[agent_id] = [observation, action]
        
        else:
            if termination or truncation:
                action = None
            else:
                action = env.action_space(agent_id).sample()
            
        env.step(action)
        rewards[0] += reward if agent_team == "red" else 0
        rewards[1] += reward if agent_team == "blue" else 0
        
        if agent_team != last_agent_team and agent_team == "red":
            cycle_count += 1
        last_agent_team = agent_team
        
    data = relay_buffer.get_all_data()
    train(q_network, target_q_network, data)
    
    for param, target_param in zip(q_network.parameters(), target_q_network.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
      
    pbar.set_description(f"Episode {episode}")
    print(f"Episode {episode} done, step length: {cycle_count}, red reward: {rewards[0]}, blue reward: {rewards[1]}, epsilon: {epsilon:.3f}")


    torch.save(q_network.state_dict(), "blue.pt")