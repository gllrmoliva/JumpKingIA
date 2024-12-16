import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from plot_training_results import MetricsPlotter
from Train import Agent, State
from torch.distributions import Categorical
import os
import Constants as C
import csv

# Configuración del logger para seguimiento
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PPOMemory:
    def __init__(self, batch_size):
        """
        Inicializa la memoria de PPO, que almacena estados, acciones, probabilidades,
        valores, recompensas y estados de finalización.
        
        Parámetros:
            batch_size (int): Tamaño del lote para el entrenamiento por batch.
        """
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        """
        Genera lotes de índices random para entrenar las redes.
        
        Retorno:
            tuple: Arrays de estados, acciones, probabilidades, valores, recompensas,
                   estados 'done', y los índices divididos en lotes.
        """
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return  np.array(self.states), \
                np.array(self.actions), \
                np.array(self.probs), \
                np.array(self.vals), \
                np.array(self.rewards), \
                np.array(self.dones), \
                batches
    
    def store_memory(self, state, action, probs, vals, reward, done):
        """
        Almacena una transición de experiencia en la memoria.
        """
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear_memory(self):
        """
        Limpia la memoria después de cada actualización.
        """
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, alpha, hidden_dim=256, chkpt_dir='tmp/ppo'):
        """
        Inicializa la red del actor que predice distribuciones de políticas.
        
        Parámetros:
            state_dim (int): Dimensión del espacio de estados.
            action_dim (int): Dimensión del espacio de acciones.
            alpha (float): Tasa de aprendizaje.
            hidden_dim (int): Tamaño de las capas ocultas.
            chkpt_dir (str): Directorio donde se guardan los checkpoints.
        """
        super(ActorNetwork, self).__init__()
        
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Pasa un estado a través de la red para obtener la distribución de acciones.
        """
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist
    
    def save_checkpoint(self):
        """
        Guarda los pesos del modelo.
        """
        logging.info('Guardando modelo...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Carga los pesos del modelo desde un archivo.
        """
        logging.info('Cargando modelo...')
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, alpha, hidden_dim=256, chkpt_dir='tmp/ppo'):
        """
        Inicializa la red del crítico que predice el valor de un estado dado.
        
        Parámetros:
            state_dim (int): Dimensión del espacio de estados.
            alpha (float): Tasa de aprendizaje.
            hidden_dim (int): Tamaño de las capas ocultas.
            chkpt_dir (str): Directorio donde se guardan los checkpoints.
        """
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Pasa un estado a través de la red para obtener su valor.
        """
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        """
        Guarda los pesos del modelo.
        """
        logging.info('Guardando modelo...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Carga los pesos del modelo desde un archivo.
        """
        logging.info('Cargando modelo...')
        self.load_state_dict(T.load(self.checkpoint_file))


class PPOAgent(Agent):

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95, eps_clip=0.2, batch_size=64, K_epochs=4):
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.actor = ActorNetwork(state_dim, action_dim, lr)
        self.critic = CriticNetwork(state_dim, lr)
        self.memory = PPOMemory(batch_size)

        # Métricas de entrenamiento
        self.plotter = MetricsPlotter()
        self.total_rewards = []  # Lista de recompensas acumuladas por episodio
        self.current_episode_rewards = 0  # Recompensa acumulada actual
        self.episode_steps = 0  # Número de pasos en el episodio actual

    def remember(self, state, action, prob, value, reward, done):
        state_array = np.array(self.normalize_state(state), dtype=np.float32)
        self.memory.states.append(state_array)
        self.memory.actions.append(action)
        self.memory.probs.append(prob)
        self.memory.vals.append(value)
        self.memory.rewards.append(reward)
        self.memory.dones.append(done)
        
    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
    
    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        
    def start_episode(self):
        """
        Limpia la memoria al inicio de cada episodio y resetea métricas.
        """
        self.memory.clear_memory()
        self.current_episode_rewards = 0
        self.episode_steps = 0
        
        logging.info(f"Inicio del episodio {len(self.total_rewards) + 1}")
    
    def select_action(self, state: State):
        normalized_state = self.normalize_state(state)
        state = T.tensor([normalized_state], dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = int(T.squeeze(action).item())
        value = T.squeeze(value).item()

        self.memory.probs.append(probs)
        self.memory.vals.append(value)

        return action

          
    def train(self, state: State, action: int, next_state: State):
        """
        Realiza el entrenamiento del agente. Calcula las recompensas y actualiza los modelos Actor y Critic.
        """
        # Calcular recompensa con `compute_reward`
        reward = self.compute_reward(state, action, next_state)
        self.remember(state, action, self.memory.probs[-1], self.memory.vals[-1], reward, next_state.done)

        # Actualizar métricas
        self.current_episode_rewards += reward
        self.episode_steps += 1

        # Mostrar progreso
        if self.episode_steps % 25 == 0:
            logging.info(
                f"Progreso: Episodio {len(self.total_rewards) + 1}, "
                f"Pasos: {self.episode_steps}, "
                f"Recompensa acumulada: {self.current_episode_rewards:.2f}"
            )
            #self.save_episode_data()

        # Si el episodio termina, realizar la actualización de la red
        if next_state.done:
            self._update_models()
            self.total_rewards.append(self.current_episode_rewards)
            logging.info(f"Episodio {len(self.total_rewards)} terminado. Recompensa Total: {self.current_episode_rewards:.2f}, Pasos Totales: {self.episode_steps}")

    def _update_models(self):
        """
        Realiza la actualización de las redes Actor y Critic, calcula las ventajas y entrena por lotes.
        """
        for _ in range(self.K_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # Calcular ventajas (GAE)
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (
                        reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k]
                    )
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)

            # Entrenamiento por lotes
            for batch in batches:
                # Asegúrate de que el índice batch esté correcto
                states = np.array([state_arr[i] for i in batch], dtype=np.float32)
                old_probs = np.array([old_prob_arr[i] for i in batch], dtype=np.float32)
                actions = np.array([action_arr[i] for i in batch], dtype=np.int64)  # Asegúrate de que las acciones sean de tipo entero

                # Convierte los lotes de numpy a tensores de PyTorch
                states = T.tensor(states).to(self.actor.device)
                old_probs = T.tensor(old_probs).to(self.actor.device)
                actions = T.tensor(actions).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states).squeeze()

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = ((returns - critic_value) ** 2).mean()

                # Actualizar redes
                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
    
        # Actualizar recompensas y almacenar métricas
        self.current_episode_rewards = sum(reward_arr)  # Asegúrate de que esto esté bien calculado
        self.plotter.store_metrics(len(self.total_rewards) + 1, self.current_episode_rewards, total_loss.item())


        self.memory.clear_memory()


    def end_episode(self):
        pass

    def compute_reward(self, state: State, action: int, next_state: State):
        """
        Calcula la recompensa basada en el cambio de estado.
        """
        reward = 0
        if action == 0 or action == 1:
            reward += -2
        else:
            reward += -5
        
        delta_height = next_state.height - state.height
        if delta_height < 0:
            fall = delta_height * -1
            reward += -2 * min(fall, 3)
        else:
            reward += 1 * delta_height
        
        # Recompensa por alcanzar una altura mayor
        new_max_height = next_state.max_height - state.max_height
        if new_max_height > 0:
            reward += 5 * new_max_height
        # Recompensa por ganar
        if next_state.win:
            reward += 10000

        return reward

    def save(self, path):
        """
        Guarda el modelo global (actor, crítico, optimizadores) y la memoria.
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor.optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic.optimizer.state_dict(),
            'gamma': self.gamma,
            'eps_clip': self.eps_clip,
            'K_epochs': self.K_epochs,
            'memory': self.memory,  # Guarda la memoria para el agente
            'total_rewards': self.total_rewards,
        }
        
        T.save(checkpoint, path)
        logging.info(f"Modelo guardado en {path}")
    
    def load(self, path):
        """
        Carga el modelo global (actor, crítico, optimizadores) y la memoria.
        """
        checkpoint = T.load(path)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor.optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic.optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        # Recuperar otros parámetros
        self.gamma = checkpoint['gamma']
        self.eps_clip = checkpoint['eps_clip']
        self.K_epochs = checkpoint['K_epochs']
        self.memory = checkpoint['memory']
        self.total_rewards = checkpoint['total_rewards']

        logging.info(f"Modelo cargado desde {path}")


    def plot(self):
        """
        Muestra las gráficas de las métricas de entrenamiento.
        """
        self.plotter.plot_all()
        #self.plotter.plot_action_probabilities(self.memory.probs)


    def normalize_state(self, state: State) -> list:
        """
        Normaliza los atributos de una instancia de State y los devuelve como una lista.
        
        Parámetros:
            state (State): Instancia de la clase State.

        Retorno:
            list: Lista con los valores normalizados.
        """
        # Normalización (ajusta según las necesidades de tu modelo)
        normalized = [
            state.level / C.EPISODE_MAX_LEVEL if C.EPISODE_MAX_LEVEL else 0,
            state.x / C.LEVEL_HORIZONTAL_SIZE if C.LEVEL_HORIZONTAL_SIZE else 0,
            state.height / C.GAME_MAX_HEIGHT if C.GAME_MAX_HEIGHT else 0,
            state.max_height / C.GAME_MAX_HEIGHT if C.GAME_MAX_HEIGHT else 0,
            1 if state.done else 0,
        ]
        return normalized
    
    def save_episode_data(self):
        """
        Guarda los datos del episodio (número de episodio, pasos y recompensa acumulada)
        en un archivo CSV cada 25 pasos.
        """
        file_exists = os.path.isfile('episode_data_ppo.csv')
 
        episode_data = [len(self.total_rewards) + 1, self.episode_steps, self.current_episode_rewards]
        with open('episode_data_ppo.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Episodio', 'Pasos', 'Recompensa Acumulada'])
            
            writer.writerow(episode_data)
