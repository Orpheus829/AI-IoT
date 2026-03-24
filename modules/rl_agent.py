"""
Reinforcement Learning Agent for Task Allocation
Implements Q-learning and MDP from Chapter 11.5
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

@dataclass
class MDPState:
    """State representation for MDP"""
    fatigue_level: float  # 0-1
    queue_length: int     
    ambiguity: float      # 0-1
    
    def to_index(self, fatigue_bins=5, queue_bins=10, ambig_bins=5) -> int:
        """Convert continuous state to discrete index"""
        f_idx = min(int(self.fatigue_level * fatigue_bins), fatigue_bins - 1)
        q_idx = min(self.queue_length, queue_bins - 1)
        a_idx = min(int(self.ambiguity * ambig_bins), ambig_bins - 1)
        
        return f_idx * (queue_bins * ambig_bins) + q_idx * ambig_bins + a_idx

class QLearningAgent:
    """
    Q-Learning agent for human-robot task allocation
    
    Q(s,a) ← Q(s,a) + η[r + γ max Q(s',a') - Q(s,a)]
    """
    
    def __init__(self, n_states: int, n_actions: int,
                 learning_rate: float = 0.001,
                 discount_factor: float = 0.95,
                 exploration_rate: float = 0.1):
        """
        Initialize Q-learning agent
        
        Args:
            n_states: Number of discrete states
            n_actions: Number of actions
            learning_rate: η
            discount_factor: γ
            exploration_rate: ε for ε-greedy
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.eta = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        
        # Q-table
        self.Q = np.zeros((n_states, n_actions))
        
        # Statistics
        self.total_reward = 0
        self.episode_rewards = []
    
    def select_action(self, state: int, greedy: bool = False) -> int:
        """
        Select action using ε-greedy policy
        
        Args:
            state: Current state index
            greedy: If True, always pick best action
            
        Returns:
            Action index
        """
        if not greedy and np.random.rand() < self.epsilon:
            # Explore
            return np.random.randint(self.n_actions)
        else:
            # Exploit
            return np.argmax(self.Q[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int):
        """
        Q-learning update
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
        """
        # Temporal difference error
        td_target = reward + self.gamma * np.max(self.Q[next_state])
        td_error = td_target - self.Q[state, action]
        
        # Update Q-value
        self.Q[state, action] += self.eta * td_error
        
        self.total_reward += reward
    
    def train_episode(self, env, max_steps: int = 100) -> float:
        """
        Train for one episode
        
        Args:
            env: Environment with reset() and step(action) methods
            max_steps: Maximum steps per episode
            
        Returns:
            Total reward for episode
        """
        state = env.reset()
        episode_reward = 0
        
        for _ in range(max_steps):
            action = self.select_action(state)
            next_state, reward, done = env.step(action)
            
            self.update(state, action, reward, next_state)
            episode_reward += reward
            
            state = next_state
            if done:
                break
        
        self.episode_rewards.append(episode_reward)
        return episode_reward
    
    def get_policy(self) -> np.ndarray:
        """Extract greedy policy from Q-table"""
        return np.argmax(self.Q, axis=1)


class TaskAllocationEnvironment:
    """
    MDP environment for task allocation
    
    States: (Fatigue, Queue Length, Ambiguity)
    Actions: {Assign to Human, Assign to Robot, Pause}
    Reward: Minimize time + fatigue accumulation
    """
    
    def __init__(self, max_queue: int = 20):
        """
        Initialize environment
        
        Args:
            max_queue: Maximum queue size
        """
        self.max_queue = max_queue
        
        # Action space
        self.ASSIGN_HUMAN = 0
        self.ASSIGN_ROBOT = 1
        self.PAUSE_LINE = 2
        self.n_actions = 3
        
        # State discretization
        self.fatigue_bins = 5
        self.queue_bins = 10
        self.ambig_bins = 5
        self.n_states = self.fatigue_bins * self.queue_bins * self.ambig_bins
        
        # Current state
        self.fatigue = 0.0
        self.queue = 0
        self.ambiguity = 0.3
        
        # Weights for reward function
        self.w1 = 1.0  # Cycle time weight
        self.w2 = 2.0  # Fatigue weight
    
    def reset(self) -> int:
        """Reset environment to initial state"""
        self.fatigue = np.random.uniform(0.2, 0.4)
        self.queue = np.random.randint(0, 5)
        self.ambiguity = np.random.uniform(0.1, 0.4)
        
        return self._get_state_index()
    
    def _get_state_index(self) -> int:
        """Convert continuous state to discrete index"""
        f_idx = min(int(self.fatigue * self.fatigue_bins), self.fatigue_bins - 1)
        q_idx = min(self.queue, self.queue_bins - 1)
        a_idx = min(int(self.ambiguity * self.ambig_bins), self.ambig_bins - 1)
        
        return f_idx * (self.queue_bins * self.ambig_bins) + q_idx * self.ambig_bins + a_idx
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Take action and observe outcome
        
        Args:
            action: Action index
            
        Returns:
            (next_state, reward, done)
        """
        # Task arrival (Poisson-like)
        if np.random.rand() < 0.3:
            self.queue = min(self.queue + 1, self.max_queue)
        
        cycle_time = 0.0
        fatigue_change = 0.0
        
        if action == self.ASSIGN_HUMAN:
            # Human processes task
            if self.queue > 0:
                self.queue -= 1
                cycle_time = 1.0 + self.ambiguity * 0.5  # Ambiguity slows down
                fatigue_change = 0.05 + self.ambiguity * 0.02
        
        elif action == self.ASSIGN_ROBOT:
            # Robot processes task (slower but no fatigue)
            if self.queue > 0:
                self.queue -= 1
                cycle_time = 1.5
                fatigue_change = -0.02  # Human rests
        
        elif action == self.PAUSE_LINE:
            # Pause for recovery
            cycle_time = 2.0
            fatigue_change = -0.1
        
        # Update fatigue
        self.fatigue = np.clip(self.fatigue + fatigue_change, 0, 1)
        
        # Update ambiguity (random walk)
        self.ambiguity = np.clip(self.ambiguity + np.random.normal(0, 0.05), 0, 1)
        
        # Calculate reward (negative cost)
        reward = -(self.w1 * cycle_time + self.w2 * fatigue_change)
        
        # Penalize high queue
        if self.queue > 10:
            reward -= (self.queue - 10) * 0.5
        
        # Penalize high fatigue
        if self.fatigue > 0.8:
            reward -= 2.0
        
        # Episode ends if queue empty and fatigue low
        done = (self.queue == 0 and self.fatigue < 0.3)
        
        return self._get_state_index(), reward, done


def train_rl_agent(episodes: int = 1000, verbose: bool = True) -> QLearningAgent:
    """
    Train RL agent for task allocation
    
    Args:
        episodes: Number of training episodes
        verbose: Print progress
        
    Returns:
        Trained agent
    """
    env = TaskAllocationEnvironment()
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.01,
        discount_factor=0.95,
        exploration_rate=0.1
    )
    
    for ep in range(episodes):
        reward = agent.train_episode(env, max_steps=50)
        
        if verbose and (ep + 1) % 100 == 0:
            avg_reward = np.mean(agent.episode_rewards[-100:])
            print(f"Episode {ep+1}/{episodes}, Avg Reward: {avg_reward:.2f}")
    
    return agent


if __name__ == "__main__":
    print("Reinforcement Learning Agent Test")
    print("=" * 50)
    
    # Train agent
    print("\nTraining Q-learning agent...")
    agent = train_rl_agent(episodes=500, verbose=True)
    
    # Test learned policy
    print("\nTesting learned policy...")
    env = TaskAllocationEnvironment()
    state = env.reset()
    
    total_reward = 0
    for step in range(20):
        action = agent.select_action(state, greedy=True)
        action_name = ['Human', 'Robot', 'Pause'][action]
        
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        print(f"Step {step}: Action={action_name}, Queue={env.queue}, "
              f"Fatigue={env.fatigue:.2f}, Reward={reward:.2f}")
        
        state = next_state
        if done:
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")




class DQNNetwork(nn.Module):
    """Deep Q-Network for task allocation"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)


class DQNAgent:
    """Deep Q-Learning Agent with Experience Replay"""
    
    def __init__(self, state_dim=3, action_dim=3, 
                 learning_rate=0.001, gamma=0.95, epsilon=0.1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Q-networks
        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64
        
        # Metrics
        self.episode_rewards = []
        self.losses = []
    
    def select_action(self, state, greedy=False):
        """Epsilon-greedy action selection"""
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, 2)  # Random action
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """Train on a minibatch from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample random minibatch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
        
        # Compute loss and update
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        return loss.item()
    
    def update_target_network(self):
        """Soft update of target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train_episode(self, env, max_steps=100):
        """Train for one episode"""
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select and perform action
            action = self.select_action(state)
            next_state, reward, done = env.step(action)
            
            # Store transition
            self.store_transition(state, action, reward, next_state, done)
            
            # Train on minibatch
            loss = self.train_step()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Update target network every episode
        if len(self.episode_rewards) % 10 == 0:
            self.update_target_network()
        
        self.episode_rewards.append(episode_reward)
        return episode_reward


# Training function
def train_dqn_agent(episodes=1000, verbose=True):
    """Train DQN agent"""
    env = TaskAllocationEnvironment()
    
    # Convert state to continuous representation
    state_dim = 3  # [fatigue, queue, ambiguity]
    action_dim = 3  # [human, robot, pause]
    
    agent = DQNAgent(state_dim, action_dim)
    
    for ep in range(episodes):
        reward = agent.train_episode(env, max_steps=50)
        
        if verbose and (ep + 1) % 100 == 0:
            avg_reward = np.mean(agent.episode_rewards[-100:])
            avg_loss = np.mean(agent.losses[-100:]) if agent.losses else 0
            print(f"Episode {ep+1}/{episodes}, Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}")
    
    return agent