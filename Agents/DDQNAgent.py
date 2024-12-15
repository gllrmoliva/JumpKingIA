import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from Train import Agent, State, Train
from ActionSpace import generate_action_space

REWARD_FOR_WALKING =                        -2
REWARD_FOR_JUMPING =                        -5
REWARD_FOR_GOING_DOWN =                     -5   # Asumiendo una caida promedio de 720, para un total promedio de 50
REWARD_FOR_GOING_UP =                        5       # Asumiendo una subida promedio de 50 (¡Pero la mayoria de saltos suelen ser caidas!), para un total promedio de 250
REWARD_FOR_NEW_MAX_HEIGHT =                 10      # Recompensamos el doble, para un total promedio de 750
STATIC_REWARD_FOR_NEW_MAX_HEIGHT =          100     # Para un caso borde donde la altura ganada es poca (¡Igual nos interesa premiar!)
REWARD_FOR_NEW_MAX_LEVEL =                  2000    # Ocurre muy poco
REWARD_FOR_WIN =                            100000   # Ocurre una vez



class DQN(nn.Module):
    """ Red neuronal que se utilizara para el DDQN.
    obs: esta puede ser modificada para optimizar su uso con el JumpKing
    """

    def __init__(self, state_size, action_size, hidden_dim = 128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))

        return self.fc5(x)

class DDQNAgent(Agent):
    def __init__(self,
                 state_size: int,                   # Cantidad de nodos de entrada (dimensión de estados)
                 action_size: int,                  # Cantidad de acciones que se pueden hacer 
                 gamma: float = 0.99,               # 
                 lr: float = 1e-3,                  # Learning rate, que tan rapido aprende
                 batch_size: int = 64,              # Tamaño del batch
                 memory_size: int = 10000,          # Cantidad de memoria.Son tuplas (state, action, new_state, reward, terminated)
                 target_update_freq: int = 100,     # Cada cuantas acciones se actualiza la Target Network
                 epsilon_start: float = 1.0,        # Epsilon inicial
                 epsilon_end: float = 0.1,          # Valor minimo que puede alcanzar epsilon
                 epsilon_decay: int = 1000):        # Cuantos episodios de demoraria en bajar a 0 el epsilon
                                                    # Baja de forma lineal ( epsilon = epsilon - (1/decay) )
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)         # Memory replay
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_count = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"    # Intenta utilizar cuda si este esta disponible
        self.episode_reward = 0

        self.policy_net = DQN(state_size, action_size).to(self.device)          # Red policy
        self.target_net = DQN(state_size, action_size).to(self.device)          # Red Target
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)   # Optimizador


        self.target_net.load_state_dict(self.policy_net.state_dict())           # Copiamos los datos de Policy en target 
        self.target_net.eval()                                                  # Modo evaluacion, Al parecer es bueno, lo recomiendan en 
                                                                                # internet
        
        print("Dispositivo utilizado: ", self.device)

    def start_episode(self):
        # LLevamos tracking de la recompensa total dada en cada episodio
        self.episode_reward = 0

    def select_action(self, state: State):

        # Sumamos uno a las acciones tomadas
        self.step_count += 1

        # Metodo epsilon Greddy
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state_tensor = self._state_to_tensor(state).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item()

    def train(self, state: State, action: int, next_state: State):

        # calculamos recompensa
        reward = self._calculate_reward(state, next_state)
        done = next_state.win


        # Guardamos una transision en meMoria
        self.memory.append((
            self._state_to_tensor(state).to(self.device),
            action,
            torch.tensor([reward], dtype=torch.float32).to(self.device),
            self._state_to_tensor(next_state).to(self.device),
            torch.tensor([done], dtype=torch.float32).to(self.device)
        ))

        # La red se entrena solo si tenemos suficiente experiencia
        if len(self.memory) >= self.batch_size:
            self._replay_experience()

        # actualizamos la network target
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def end_episode(self):
        # Al terminar episodio actualizamos el epsilon

        self.epsilon = max(self.epsilon_end, self.epsilon - (1 / self.epsilon_decay))

        print("Epsilon: {}, Recompensa acumulada en episodio: {}".format(self.epsilon, self.episode_reward))
        pass

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def _calculate_reward(self, state: State, next_state: State):

        reward = 0

        delta_height = next_state.height - state.height
        # Recompensa o penalizacion por subir o bajar
        if delta_height < 0:
            how_much_down = -1 * delta_height
            reward += REWARD_FOR_GOING_DOWN * how_much_down
        else:
            how_much_up = delta_height
            reward += REWARD_FOR_GOING_UP * how_much_up

        # Recompensa por superar la actual altura máxima
        delta_max_height = next_state.max_height - state.max_height
        if delta_max_height > 0:
            reward += REWARD_FOR_NEW_MAX_HEIGHT * delta_max_height  + STATIC_REWARD_FOR_NEW_MAX_HEIGHT

        # Recompensa por ganar
        if next_state.win:
            reward += REWARD_FOR_WIN

        self.episode_reward += reward

        return reward

    def _replay_experience(self):
        # Obtenemos un sample de experiencias y actualizamos la red.

        batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.cat(rewards).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        dones = torch.cat(dones).to(self.device)

        # Obtenemos Q values
        q_values = self.policy_net(states).gather(1, actions).squeeze()

        # Obtenemos Q values maximos de acciones siguiente
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Computamos perdida
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimizamos el modelo (Magia negra)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _state_to_tensor(self, state: State):
        # Dado un estado creamos un tensor

        normalized_state = torch.tensor([
            state.level_normalized,
            state.x_normalized,
            state.y_normalized
        ], dtype=torch.float32).to(self.device)
        return normalized_state.unsqueeze(0)