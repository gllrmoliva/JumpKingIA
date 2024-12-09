import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Train import Agent, State
from Constants import LEVEL_MATRIX_HORIZONTAL_SIZE, LEVEL_MATRIX_VERTICAL_SIZE, LEVEL_HORIZONTAL_SIZE, LEVEL_VERTICAL_SIZE
from datetime import datetime

example_state = State()
example_state.level = 0
example_state.x = 0
example_state.y = 0
example_state.jumpCount = 0
example_state.done = False
example_state.level_matrix = np.zeros((
                      LEVEL_MATRIX_VERTICAL_SIZE,
                      LEVEL_MATRIX_HORIZONTAL_SIZE),
                      dtype=np.uint8)

# Define the neural network for the Q-function
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent(Agent):
    def __init__(self, state_size=-1, action_size=4, gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        state_size = len(State.encode(example_state))
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.memory = []  # Replay buffer
        self.batch_size = 64
        self.max_memory_size = 10000

        self.episode_reward = 0

    def start_episode(self):
        self.episode_reward = 0

    def select_action(self, coded_state: bytes):
        state = State.decode(coded_state)
        state_tensor = torch.tensor(np.frombuffer(coded_state, dtype=np.uint8), dtype=torch.float32).unsqueeze(0)

        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                return torch.argmax(q_values).item()

    def train(self, coded_state: bytes, action: int, coded_next_state: bytes):
        state = State.decode(coded_state)
        next_state = State.decode(coded_next_state)

        state_vector = coded_state
        next_state_vector = coded_next_state

        height = state.y + state.level * LEVEL_VERTICAL_SIZE
        next_height = next_state.y + next_state.level * LEVEL_VERTICAL_SIZE

        reward = next_height - height  # Example reward based on height
        done = next_state.done

        self.memory.append((state_vector, action, reward, next_state_vector, done))
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)

        if len(self.memory) >= self.batch_size:
            self.replay()

        self.episode_reward += reward

    def replay(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Compute Q(s, a)
        q_values = self.policy_net(states).gather(1, actions)

        # Compute max_a' Q(s', a') for target
        with torch.no_grad():
            target_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        targets = rewards + (1 - dones) * self.gamma * target_q_values

        # Update policy network
        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def end_episode(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def load(self, path):
        if path != None:
            checkpoint = torch.load(path)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']

    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

        print(datetime.now().time())
