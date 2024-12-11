import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Train import Agent, State

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

class PPOAgent(Agent):
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

        self.memory = Memory()

    def start_episode(self):
        self.memory.clear_memory()

    def select_action(self, state: State):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.policy_old(state)
        action = np.random.choice(len(action_probs[0]), p=action_probs[0].numpy())
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.logprobs.append(torch.log(action_probs[0][action]))
        return action

    def train(self, state: State, action: int, next_state: State):
        reward = self.compute_reward(state, next_state)
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(next_state.done)

        if next_state.done:
            self.update()

    def end_episode(self):
        pass

    def compute_reward(self, state, next_state):
        # Ejemplo de lógica para calcular la recompensa
        reward = 0

        # Recompensa por avanzar a un nuevo nivel
        if next_state.level > state.level:
            reward += 100

        # Penalización por caer
        if next_state.y > state.y:
            reward -= 10

        # Recompensa por saltar
        if next_state.jumpCount > state.jumpCount:
            reward += 1

        # Penalización si el juego ha terminado
        if next_state.done:
            reward -= 50

        return reward

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        old_states = torch.cat(self.memory.states)
        old_actions = torch.tensor(self.memory.actions, dtype=torch.long)
        old_logprobs = torch.cat(self.memory.logprobs)

        for _ in range(self.K_epochs):
            logprobs, state_values = self.policy(old_states)
            dist_entropy = -torch.sum(logprobs * torch.log(logprobs), dim=1)
            logprobs = logprobs.gather(1, old_actions.unsqueeze(1)).squeeze(1)
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach().squeeze()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values.squeeze(), rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]