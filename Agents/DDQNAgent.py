import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
import random

from Train import Agent, State
from Constants import *

class DQN(nn.Module):

    def __init__(self, state_dim: int, action_dim: int , hidden_dim = 256):
        super(DQN,self).__init__()
        self.model = nn.Sequential(
                        nn.Linear(state_dim, hidden_dim),
                        nn.ReLU(), 
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, action_dim)
                    )
    
    def forward(self,x):
        return self.model(x)

class ReplayMemory():
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)
        if seed is not None:
            random.seed(seed)
    
    # Queremos que transition = (state, action, new_state, reward, terminated)
    def append(self, transition):
        self.memory.append(transition)
    
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class DDQNAgent(Agent):
    def __init__(self, state_dim: int, action_dim: int, is_training: bool):

        # Entrenamos en GPU si se puede
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Variables de entrenamiento
        self.action_dim = action_dim # FIXME: Esto se va a tener que arreglar dependiendo de cual va a ser el espacio de acciones
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.1
        self.replay_memory_size = 50000
        self.batch_size = 64 
        self.episode_reward = 0
        self.step_count = 0
        self.network_sync_rate = 100
        self.learning_rate_a = 0.01
        self.discount_factor_gamma = 0.99

        if(not is_training):
            self.epsilon = 0
            self.epsilon_decay = 1
            self.epsilon_min = 0

        self.loss_fn = nn.MSELoss()         # NN Loss Function. 
        self.optimizer = None               # NN Optimizer. 

        # Red Policy
        self.policy_dqn = DQN(state_dim, action_dim).to(self.device)
        self.is_training = is_training

        # Si vamos a entrenar usamos esto.
        if self.is_training:
            self.memory = ReplayMemory(self.replay_memory_size)
            self.target_dqn = DQN(state_dim, action_dim).to(self.device)
            self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

            self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr = self.learning_rate_a)


    # Se llama cuando inicia un episodio.
    # Deberia, por ejemplo, inicializar variables locales a un solo episodio
    def start_episode(self):
        self.episode_reward = 0

    # Devuelve que acción tomar segun un estado de juego
    def select_action(self, state: 'State'):

        state_tensor = self.state_to_tensor(state)

        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim-1)

        else: 
            with torch.no_grad():
                q_values = self.policy_dqn(state_tensor.unsqueeze(0))
                action = torch.argmax(q_values).item()
        
        return action

    # Funcion de recompensa, esto es importante!!!!!
    def calculate_reward(self, state: 'State', action: int, next_state: 'State') -> int:
        """
        k1 = 50
        k2 = 10
        k3 = 5
        k4 = 1
        delta_max_height = next_state.max_height - state.max_height
        delta_height = next_state.height - state.height
        if delta_height >= 0:
            delta_heght_positive = delta_height
            delta_heght_negative = 0
        else:
            delta_heght_positive = 0
            delta_heght_negative = -1 * delta_height

        r = k1 * delta_max_height + k2 * delta_heght_positive - k3 * delta_heght_negative - k4
        """
        # Objetivo: ir hacia la izquierda
        left_k = 10
        right_k = -1
        delta_x = next_state.x_normalized - state.x_normalized

        if delta_x < 0:
            left_mov = -1 * delta_x
            right_mov = 0
        else:
            left_mov = 0
            right_mov = delta_x

        reward = left_k * left_mov + right_k * right_mov

        return reward
  

    # Entrena al modelo sabiendo que acción tomó en un estado, y a que otro estado llevó
    # Acá podria por ejemplo: Calcular recompensa, actualizar recompensa acumulada, etc.
    def train(self, state: 'State', action: int, next_state: 'State'):
        reward = self.calculate_reward(state, action, next_state)
        #print("Recompensa: {}".format(reward))
        self.episode_reward += reward
        done = next_state.win

        # Transformamos los estados a tensores
        state_tensor = self.state_to_tensor(state)
        next_state_tensor = self.state_to_tensor(next_state)

        # Solo si estamos entrenando utilizamos la memoria
        if self.is_training:
            self.memory.append((state_tensor, action, next_state_tensor, reward, done))

            self.step_count += 1
            # Si suficiente experiencia a sido recolectada
            if len(self.memory) > self.batch_size:
                # Sample de memory
                batch = self.memory.sample(self.batch_size)
                # Optimize
                self.optimize(batch, self.policy_dqn, self.target_dqn)

                if self.step_count > self.network_sync_rate:
                    self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                    self.step_count = 0
        
    # Se llama cuando acaba el episodio cambiamos el epsilon.

    def optimize(self, batch, policy_dqn, target_dqn):

        states, actions, new_states, rewards, terminations = zip(*batch)

        states = torch.stack(states)

        #actions = torch.stack(actions)
        # Aqui me tiraba un error, con esto se arreglo, pero no se porque
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)

        new_states = torch.stack(new_states)

        # rewards = torch.stack(rewards)
        # Aqui lo mismo
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)

        #terminations = torch.tensor(terminations).float().to(self.device)
        # Aqui lo mismo
        terminations = torch.tensor(terminations, dtype=torch.float, device=self.device)

        # Validate actions
        assert torch.all((actions >= 0) & (actions < self.action_dim)), "Actions out of bounds"

        with torch.no_grad():
            # Calculate target Q values (expected returns)

            best_actions_from_policy = self.policy_dqn(new_states).argmax(dim=1)

            assert torch.all((best_actions_from_policy >= 0) & (best_actions_from_policy < self.action_dim)), \
            "Best actions out of bounds"

            target_q =  rewards + (1- terminations) * self.discount_factor_gamma * \
                        target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()

        
        # Calculate Q values from current policy 
        current_q = policy_dqn(states).gather(dim=1, index= actions.unsqueeze(dim=1)).squeeze()

        assert current_q.shape == target_q.shape, \
        f"Shape mismatch: current_q {current_q.shape}, target_q {target_q.shape}"

        # Compute loss for the whole batch
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients (backpropagation)
        self.optimizer.step()       # Update Network parameters i.e. weigths and biases


    def end_episode(self):
        if (self.is_training):
            # Al terminar episodio 
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


        print("End Episode, Current epsilon: {epsilon} ,Acumulative reward: {reward}".format(reward = self.episode_reward, epsilon = self.epsilon))

    # Para cargar datos de entrenamientos previos
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        # Cargar los parámetros del modelo y el optimizador
        self.policy_dqn.load_state_dict(checkpoint['policy_dqn_state_dict'])

        if (self.is_training):
            self.target_dqn.load_state_dict(checkpoint['target_dqn_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']

        print(f"Modelo cargado desde {path}")

    # Para guardar datos del entrenamiento actual
    def save(self, path):
        if (self.is_training):
            checkpoint = {
                'policy_dqn_state_dict': self.policy_dqn.state_dict(),
                'target_dqn_state_dict': self.target_dqn.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                }
            torch.save(checkpoint, path)
            print(f"Modelo guardado en {path}")

    def state_to_tensor(self, state: 'State') -> torch.Tensor:
        """Transforma los estados a tensores que pueden ser procesados por la Red.
        """
        # FIXME: Ahora mismo se van a hacer pruebas con estados más pequeños y tontos
        scalar_values = torch.tensor(
            [state.x_normalized, state.y_normalized, state.level_normalized], 
            dtype=torch.float
        ).to(self.device)
        
        """
        # Aplanar la matriz 2D del nivel y convertirla en un tensor
        level_matrix_tensor = torch.tensor(state.level_matrix.flatten(), dtype=torch.float).to(self.device)
        
        # Concatenar los valores escalares y la matriz aplanada
        full_state_tensor = torch.cat((scalar_values, level_matrix_tensor)).to(self.device)
        """

        full_state_tensor = scalar_values.to(self.device)
        
        return full_state_tensor
