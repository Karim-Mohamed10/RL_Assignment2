import gymnasium as gym
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import wandb
import argparse
import os


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # Simple MLP with 2 hidden layers of 128 neurons
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env, bins=11):
        super().__init__(env)
        self.bins = bins
        self.low = env.action_space.low[0]
        self.high = env.action_space.high[0]
        self.action_space = gym.spaces.Discrete(bins)
        self.actions = np.linspace(self.low, self.high, bins)

    def action(self, action):
        # Map discrete index to continuous value
        return np.array([self.actions[action]], dtype=np.float32)


class Agent:
    def __init__(self, env, config, device):
        self.env = env
        self.config = config
        self.device = device
        
        # Dimensions
        n_observations = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # Networks
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=config.learning_rate, amsgrad=True)
        self.memory = ReplayMemory(config.memory_size)

        self.steps_done = 0

    def select_action(self, state, eval_mode=False):

        sample = random.random()
        
        eps_threshold = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.config.epsilon_decay)
        
        self.steps_done += 1

        # Log epsilon for tracking
        if self.steps_done % 100 == 0 and not eval_mode:
            wandb.log({"epsilon": eps_threshold})

        if eval_mode or sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.config.batch_size:
            return

        transitions = self.memory.sample(self.config.batch_size)
        batch = Transition(*zip(*transitions))


        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool) 
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

 
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        
        next_state_values = torch.zeros(self.config.batch_size, device=self.device)
        
        with torch.no_grad():
            if self.config.algo == "DQN":
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
            elif self.config.algo == "DDQN":
                best_actions = self.policy_net(non_final_next_states).argmax(1).unsqueeze(1)
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, best_actions).squeeze(1)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.config.gamma) + reward_batch

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()


def run_training(args):
    wandb.init(
        project="RL_Assignment_2",
        config=args,
        name=f"{args.env}_{args.algo}_ep{args.episodes}_lr{args.learning_rate}"
    )
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    env = gym.make(config.env, render_mode="rgb_array")
    
    # pendulum 
    if config.env == "Pendulum-v1":
        env = DiscreteActionWrapper(env, bins=11)


    agent = Agent(env, config, device)


    print(f"Starting training on {config.env} with {config.algo}...")
    
    for i_episode in range(config.episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        total_reward = 0
        loss_val = 0
        duration = 0

        for t in range(1000): # Max steps per episode
            action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            
            total_reward += reward
            duration += 1
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store memory
            agent.memory.push(state, action, next_state, torch.tensor([reward], device=device), torch.tensor([done], device=device))

            # Move to next state
            state = next_state

            # Optimize
            loss = agent.optimize_model()
            if loss is not None:
                loss_val += loss

            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*config.tau + target_net_state_dict[key]*(1-config.tau)
            agent.target_net.load_state_dict(target_net_state_dict)

            if done:
                break


        wandb.log({
            "episode_reward": total_reward,
            "episode_duration": duration,
            "loss": loss_val / duration if duration > 0 else 0,
            "episode": i_episode
        })
        
        if (i_episode + 1) % 50 == 0:
            print(f"Episode {i_episode+1}/{config.episodes} | Reward: {total_reward:.2f} | Duration: {duration}")

    print("Training Complete.")
    

    print("Starting Evaluation (100 Episodes)...")
    

    env.close()
    
    eval_env = gym.make(config.env, render_mode="rgb_array")
    if config.env == "Pendulum-v1":
        eval_env = DiscreteActionWrapper(eval_env, bins=11)
        
    eval_env = gym.wrappers.RecordVideo(
        eval_env, 
        video_folder=f"videos/{config.env}_{config.algo}",
        name_prefix=f"test_video",
        episode_trigger=lambda x: x == 0 
    )

    test_durations = []
    test_rewards = []

    for i in range(100):
        state, _ = eval_env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        ep_reward = 0
        ep_duration = 0
        
        while True:
            # Eval mode = True 
            action = agent.select_action(state, eval_mode=True)
            observation, reward, terminated, truncated, _ = eval_env.step(action.item())
            
            ep_reward += reward
            ep_duration += 1
            
            if terminated or truncated:
                break
                
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            
        test_durations.append(ep_duration)
        test_rewards.append(ep_reward)

    avg_duration = sum(test_durations) / len(test_durations)
    avg_reward = sum(test_rewards) / len(test_rewards)

    print(f"Evaluation Results - Avg Duration: {avg_duration}, Avg Reward: {avg_reward}")
    
    # Log final evaluation metrics
    wandb.log({
        "test_avg_duration": avg_duration,
        "test_avg_reward": avg_reward
    })

    eval_env.close()
    wandb.finish()


if __name__ == "__main__":
    # Argument Parser for Hyperparameters and Config
    parser = argparse.ArgumentParser(description='DQN/DDQN Assignment')
    
    # Environment & Algo
    parser.add_argument('--env', type=str, default='CartPole-v1', 
                        help='Gym environment name (CartPole-v1, Acrobot-v1, MountainCar-v0, Pendulum-v1)')
    parser.add_argument('--algo', type=str, default='DQN', choices=['DQN', 'DDQN'],
                        help='Algorithm to use: DQN or DDQN')
    
    # Hyperparameters (These are defaults, change them to "find best setup")
    parser.add_argument('--episodes', type=int, default=600, help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--memory_size', type=int, default=10000, help='Replay memory size')
    parser.add_argument('--epsilon_start', type=float, default=0.9, help='Starting epsilon')
    parser.add_argument('--epsilon_end', type=float, default=0.05, help='Final epsilon')
    parser.add_argument('--epsilon_decay', type=int, default=1000, help='Epsilon decay rate')
    parser.add_argument('--tau', type=float, default=0.005, help='Target network soft update rate')

    args = parser.parse_args()
    
    run_training(args)