import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from plot_training_results import MetricsPlotter
from Train import Agent, State

# Configuración del logger para seguimiento
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value


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


class PPOAgent(Agent):
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, Lambda=0.95, eps_clip=0.2, K_epochs=4, epsilon=0.1):
        super().__init__()
        self.gamma = gamma
        self.Lambda = Lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.epsilon = epsilon

        self.policy = ActorCritic(state_dim, action_dim)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()

        self.memory = Memory()

        # Métricas de entrenamiento
        self.plotter = MetricsPlotter()
        self.total_rewards = []  # Lista de recompensas acumuladas por episodio
        self.current_episode_rewards = 0  # Recompensa acumulada actual
        self.episode_steps = 0  # Número de pasos en el episodio actual

    def start_episode(self):
        """
        Limpia la memoria al inicio de cada episodio y resetea métricas.
        """
        self.memory.clear_memory()
        self.current_episode_rewards = 0
        self.episode_steps = 0

        # Reducir epsilon gradualmente para favorecer explotación
        self.epsilon = max(self.epsilon * 0.995, 0.05)
        logging.info(f"Inicio del episodio {len(self.total_rewards) + 1} - Epsilon: {self.epsilon:.4f}")


    def select_action(self, state: State):
        """
        Selecciona una acción basada en el estado actual.
        """
        state_tensor = torch.FloatTensor([state.level, state.x, state.y, state.jumpCount]).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.policy_old(state_tensor)
        
        if np.random.rand() < self.epsilon:
            # Exploración: selecciona una acción aleatoria
            action = np.random.choice(len(action_probs[0]))
        else:
            # Explotación: selecciona la mejor acción conocida
            action = np.random.choice(len(action_probs[0]), p=action_probs[0].numpy())

        # Almacenar en memoria
        self.memory.states.append(state_tensor)
        self.memory.actions.append(action)
        self.memory.logprobs.append(torch.log(action_probs[0][action]))

        return action

    def train(self, state: State, action: int, next_state: State):
        """
        Entrena al agente utilizando las transiciones almacenadas.
        """
        # Calcular recompensa y almacenar en memoria
        reward = self.compute_reward(state, next_state)
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(next_state.done)

        # Actualizar métricas
        self.current_episode_rewards += reward
        self.episode_steps += 1

        # Mostrar progreso cada 10 pasos
        if self.episode_steps % 25 == 0:
            logging.info(f"Progreso: Episodio {len(self.total_rewards) + 1}, Pasos: {self.episode_steps}, Recompensa Acumulada: {self.current_episode_rewards:.2f}")

        # Si el episodio terminó, realiza la actualización del modelo
        if next_state.done:
            self.update()
            self.total_rewards.append(self.current_episode_rewards)
            logging.info(f"Episodio {len(self.total_rewards)} terminado. Recompensa Total: {self.current_episode_rewards:.2f}, Pasos Totales: {self.episode_steps}")

    def end_episode(self):
        """
        Finaliza el episodio. Las métricas ya se gestionan en `train`.
        """
        pass

    def compute_reward(self, state: State, next_state: State):
        """
        Calcula la recompensa basada en el cambio de estado.
        """
        reward = (next_state.max_height - state.max_height) * 10

        # Penalizar comportamiento repetitivo
        if state.level == next_state.level and state.x == next_state.x:
            reward -= 3  # Penalización por falta de progreso

        # Recompensa adicional por terminar con éxito
        if next_state.done:
            reward += 100 if next_state.max_height > state.max_height else -50

        # Incentivar exploración
        reward += np.random.uniform(-1, 1) * 0.1

        return reward

    def update(self):
        """
        Realiza la actualización de los modelos utilizando el algoritmo PPO.
        """
        rewards, advantages = self.compute_rewards_and_advantages()

        # Preparar memoria
        old_states = torch.cat(self.memory.states)
        old_actions = torch.tensor(self.memory.actions, dtype=torch.long)

        if len(self.memory.logprobs) > 0 and all(logprob.dim() > 0 for logprob in self.memory.logprobs):
            old_logprobs = torch.cat(self.memory.logprobs)
        else:
            old_logprobs = torch.zeros(len(self.memory.actions), dtype=torch.float32)

        total_loss = 0
        total_entropy = 0
        mean_gradients = 0

        # Actualizar K_epochs veces
        for _ in range(self.K_epochs):
            logprobs, state_values = self.policy(old_states)
            dist_entropy = -(logprobs * torch.log(logprobs + 1e-8)).sum(dim=1).mean()
            action_logprobs = logprobs.gather(1, old_actions.unsqueeze(1)).squeeze(1)

            # Clipping para PPO
            ratios = torch.exp(action_logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss_policy = -torch.min(surr1, surr2).mean()

            # Pérdida de valor
            loss_value = self.MseLoss(state_values.squeeze(), rewards)

            # Total loss
            loss = loss_policy + 0.5 * loss_value - 0.01 * dist_entropy
            total_loss += loss.item()
            total_entropy += dist_entropy.item()

            # Optimización
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            grad_sum = sum(param.grad.abs().mean().item() for param in self.policy.parameters() if param.grad is not None)
            mean_gradients += grad_sum

        self.plotter.store_metrics(len(self.total_rewards) + 1,
                                self.current_episode_rewards,
                                mean_gradients / self.K_epochs,
                                total_loss / self.K_epochs,
                                total_entropy / self.K_epochs)

        # Actualizar política antigua
        self.policy_old.load_state_dict(self.policy.state_dict())

    def compute_rewards_and_advantages(self):
        """
        Calcula las ventajas (GAE) y los valores esperados.
        """
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)

        # Escalado de recompensas
        if rewards.std() > 0:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Calcular ventajas
        old_states = torch.cat(self.memory.states)
        _, state_values = self.policy(old_states)
        state_values = state_values.squeeze()

        advantages = rewards - state_values.detach()

        # Normalizar ventajas
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return rewards, advantages


    def save(self, path):
        """
        Guarda el modelo en un archivo.
        """
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        """
        Carga el modelo desde un archivo.
        """
        self.policy.load_state_dict(torch.load(path))

    def plot(self):
            """
            Muestra las gráficas de las métricas de entrenamiento.
            """
            self.plotter.plot_all()