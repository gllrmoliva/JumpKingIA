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
        
class ImageConvNet(nn.Module):
    def __init__(self, input_shape=(1, LEVEL_MATRIX_VERTICAL_SIZE, LEVEL_HORIZONTAL_SIZE), output_size=128):
        super(ImageConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4, padding=0)  # Reduce tamaño
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)

        # Calcular la salida de las convoluciones para inicializar el linear layer
        with torch.no_grad():
            sample_input = torch.zeros(1, *input_shape)  # Entrada ficticia
            conv_output_size = self._get_conv_output(sample_input)
        
        self.fc = nn.Linear(conv_output_size, output_size)  # Fully connected para reducir dimensiones

    def _get_conv_output(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(torch.flatten(x, start_dim=1).shape[1])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)  # Aplanar para la capa totalmente conectada
        x = self.fc(x)
        return x


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
        self.epsilon_decay = 0.9995 
        self.epsilon_min = 0.05
        self.replay_memory_size = 50000
        self.batch_size = 128 
        self.episode_reward = 0
        self.step_count = 0
        self.network_sync_rate = 100
        self.learning_rate_a = 0.01
        self.discount_factor_gamma = 0.99
        self.CNN = ImageConvNet().to(self.device)   # Red CNN
        self.levels_CNN = [-1] * MAX_LEVEL          # Que guardara el nivel CNN.

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
        #/* k1 >> k2 */
        k1 = 10 
        k2 = 1 

        p1 = k1*(next_state.height - state.height)                # global 
        p2 = k2*(next_state.max_height_last_step - state.height)  # local 
        r = p1 + p2

        return r

    # Entrena al modelo sabiendo que acción tomó en un estado, y a que otro estado llevó
    # Acá podria por ejemplo: Calcular recompensa, actualizar recompensa acumulada, etc.
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

            # Si suficiente experiencia a sido recolectada
            if len(self.memory) > self.batch_size:
                # Sample de memory
                batch = self.memory.sample(self.batch_size)
                # Optimize
                self.optimize(batch, self.policy_dqn, self.target_dqn)

                if self.step_count > self.network_sync_rate:
                    self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                    self.step_count = 0

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
            print(f"Modelo guardado en {path}")

    def state_to_tensor(self, state: 'State') -> torch.Tensor:
        """Transforma los estados a tensores que pueden ser procesados por la Red.
        """
        # En teoria esto es 3
        scalar_values = torch.tensor(
            [state.x, state.y, int(state.done)], 
            dtype=torch.float
        ).to(self.device)
        
        # En teoria esto es 128
        if(self.levels_CNN[state.level] == -1):
            input_image_tensor = torch.tensor(state.level_matrix).unsqueeze(0).unsqueeze(0).float().to(self.device)
            level_tensor = self.CNN(input_image_tensor)
            self.levels_CNN[state.level] = level_tensor
        


        full_state_tensor = torch.cat((scalar_values, self.levels_CNN[state.level])).to(self.device)

        """
        # Aplanar la matriz 2D del nivel y convertirla en un tensor
        level_matrix_tensor = torch.tensor(state.level_matrix.flatten(), dtype=torch.float).to(self.device)
        
        # Concatenar los valores escalares y la matriz aplanada
        full_state_tensor = torch.cat((scalar_values, level_matrix_tensor)).to(self.device)

        full_state_tensor = scalar_values.to(self.device)
        """
        
        return full_state_tensor
