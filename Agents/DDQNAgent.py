import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
import random

from Train import Agent, State

class DQN(nn.Module):

    def __init__(self, state_dim: int, action_dim: int , hidden_dim = 256):
        super(DQN,self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

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
        self.action_space = action_dim # FIXME: Esto se va a tener que arreglar dependiendo de cual va a ser el espacio de acciones
        self.epsilon = 1
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.05
        self.replay_memory_size = 10000
        self.batch_size = 64 
        self.episode_reward = 0
        self.step_count = 0
        self.network_sync_rate = 100
        self.learning_rate_a = 0.001
        self.discount_factor_gamma = 0.99

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
            action = random.choice(range(self.action_space))

        with torch.no_grad():
            q_values = self.policy_dqn(state_tensor.unsqueeze(0))
            action = torch.argmax(q_values).item()
        
        return action

    # Funcion de recompensa, esto es importante!!!!!
    def calculate_reward(self, state: 'State', action: int, next_state: 'State') -> int:
        return 0

    # Entrena al modelo sabiendo que acción tomó en un estado, y a que otro estado llevó
    # Acá podria por ejemplo: Calcular recompensa, actualizar recompensa acumulada, etc.
    # FIXME: Quizas tiene más sentido llamar a esto run()
    def train(self, state: 'State', action: int, next_state: 'State'):
        reward = self.calculate_reward(state, action, next_state)
        self.episode_reward += reward
        done = next_state.done

        # Transformamos los estados a tensores
        state_tensor = self.state_to_tensor(state)
        next_state_tensor = self.state_to_tensor(next_state)

        # Solo si estamos entrenando utilizamos la memoria
        if self.is_training:
            self.memory.append((state_tensor, action, next_state_tensor, reward, done))
            self.step_count += 1
        
    # Se llama cuando acaba el episodio cambiamos el epsilon.

    def optimize(self, batch, policy_dqn, target_dqn):

        states, actions, new_states, rewards, terminations = zip(*batch)

        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)

        terminations = torch.tensor(terminations).float().to(self.device)

        with torch.no_grad():
            # Calculate target Q values (expected returns)

            best_actions_from_policy = self.policy_dqn(new_states).argmax(dim=1)

            target_q =  rewards + (1- terminations) * self.discount_factor_gamma * \
                        target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()

        
        # Calculate Q values from current policy 
        current_q = policy_dqn(states).gather(dim=1, index= actions.unsqueeze(dim=1)).squeeze()

        # Compute loss for the whole batch
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients (backpropagation)
        self.optimizer.step()       # Update Network parameters i.e. weigths and biases


    def end_episode(self):
        # Al terminar episodio 
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Si suficiente experiencia a sido recolectada
        if len(self.memory) > self.batch_size:
            # Sample de memory
            batch = self.memory.sample(self.batch_size)
            # Optimize
            self.optimize(batch, self.policy_dqn, self.target_dqn)

            if self.step_count > self.network_sync_rate:
                self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                self.step_count = 0

        print("End Episode, Acumulative reward: {}".format(self.episode_reward))

    # Para cargar datos de entrenamientos previos
    def load(self, path):
        pass
    # Para guardar datos del entrenamiento actual
    def save(self, path):
        pass

    def state_to_tensor(self, state: 'State') -> torch.Tensor:
        """Transforma los estados a tensores que pueden ser procesados por la Red.
        """
        scalar_values = torch.tensor(
            [state.x, state.y, state.level, int(state.done)], 
            dtype=torch.float
        ).to(self.device)
        
        # Aplanar la matriz 2D del nivel y convertirla en un tensor
        level_matrix_tensor = torch.tensor(state.level_matrix.flatten(), dtype=torch.float).to(self.device)
        
        # Concatenar los valores escalares y la matriz aplanada
        full_state_tensor = torch.cat((scalar_values, level_matrix_tensor)).to(self.device)
        
        return full_state_tensor
