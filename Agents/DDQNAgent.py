import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
import random

from Train import Agent, State
from Constants import *

REWARD_FOR_WALKING =                        -2
REWARD_FOR_JUMPING =                        -5
REWARD_FOR_GOING_DOWN =                     -3   # Asumiendo una caida promedio de 720, para un total promedio de 50
REWARD_FOR_GOING_UP =                       5       # Asumiendo una subida promedio de 50 (¡Pero la mayoria de saltos suelen ser caidas!), para un total promedio de 250
REWARD_FOR_NEW_MAX_HEIGHT =                 10      # Recompensamos el doble, para un total promedio de 750
STATIC_REWARD_FOR_NEW_MAX_HEIGHT =          100     # Para un caso borde donde la altura ganada es poca (¡Igual nos interesa premiar!)
REWARD_FOR_NEW_MAX_LEVEL =                  2000    # Ocurre muy poco
REWARD_FOR_WIN =                            10000   # Ocurre una vez

class DQN(nn.Module):
    """ Red neuronal que se utilizara para el DDQN.
    obs: esta puede ser modificada para optimizar su uso con el JumpKing
    """

    def __init__(self, state_dim: int, action_dim: int , hidden_dim = 256):
        super(DQN,self).__init__()

        # Ahora mismo tenemos 4 hidden layers, no nos importa mucho el overfitting
        self.model = nn.Sequential(
                        nn.Linear(state_dim, hidden_dim),
                        nn.ReLU(), 
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, action_dim)
                    )
    
    def forward(self,x):
        return self.model(x)

class ReplayMemory():
    """
    Clase que modela Replay Memory, basicamente es un deque pero solo con las funciones que se 
    utilizaran en el replay memory. Recordar que ReplayMemory guarda tuplas (trancisiones):
    (state, action, new_state, reward, terminated)
    """

    def __init__(self, maxlen, seed=None):
        """ - maxlen: cantidad máxima de elementos que puede guardar Replay Memory
        """
        self.memory = deque([], maxlen=maxlen)
        if seed is not None:
            random.seed(seed)
    
    # Queremos que transition = (state, action, new_state, reward, terminated)
    def append(self, transition):
        self.memory.append(transition)
    
    def sample(self, sample_size):
        """ Devuelve una muestra de la memoria.
        """
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class DDQNAgent(Agent):
    

    def __init__(self, state_dim: int, action_dim: int, is_training: bool):
        """Contructor de DDQNAgent
        - state_dim: Cantidad de nodos de entrada (dimensión de estados)
        - action_dim: Cantidad de nodos de salida (acciones posibles del agente)
        - is_trainig: True, Entrena al modelo (modifica la red); False, El modelo solo toma decisiones (epsilon=0, no modifica la red)
        """

        # Entrenamos en GPU si se puede
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Dispositivo utilizado: {self.device}")

        self.is_training = is_training

        # Variables de entrenamiento
        self.action_dim = action_dim        # Cantidad de acciones que se pueden hacer 
        self.learning_rate_a = 0.01         # Que tan rapido "aprende" el agente
        self.discount_factor_gamma = 0.99   # 
        self.epsilon = 1                    # Epsilon (probabilidad inicial de hacer acciones aleatorias)
        self.epsilon_decay = 0.9995         # Cuando disminuye por episodio la probabilidad de hacer acciones aleatorias
        self.epsilon_min = 0.05             # Valor minimo que puede alcanzar el epsilon en el periodo de entramiento
        self.replay_memory_size = 50000     # Tamaño de la Replay Memory
        self.batch_size = 128               # Cantidad de muestras que que extraen del Replay Memory
        self.episode_reward = 0             # Recompensa total del episodio
        self.step_count = 0                 # Cantidad de pasos dados

        self.network_sync_rate = 128        # Cantidad de pasos en los que las redes (policy y target) se sincronizan, esto se podria
                                            # Cambiar a 0, para que se sincronizen a cada paso, pero hace que el entrenamiento se 
                                            # Relentice mucho

        self.loss_fn = nn.MSELoss()         # NN Loss Function. 
        self.optimizer = None               # NN Optimizer. 

        self.policy_dqn = DQN(state_dim, action_dim).to(self.device)    # Red Policy (red a la que se le hacen las consultas)

        # Si Entrenamos: 
        if self.is_training:
            self.memory = ReplayMemory(self.replay_memory_size)                 # Generamos la deque ReplayMemory

            self.target_dqn = DQN(state_dim, action_dim).to(self.device)        # Generamos la red objetivo (hacia donde quiere ir la policy)
            self.target_dqn.load_state_dict(self.policy_dqn.state_dict())       # Copiamos Policy en target

            self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(),     # Cargamos optimizador de Pytorch (Magia o Matemática¿?)
                                               lr = self.learning_rate_a)
        # Si no entrenamos
        else: 
            self.epsilon = 0
            self.epsilon_decay = 1
            self.epsilon_min = 0


    def start_episode(self):
        """ Acción que se hace al iniciar un episodio.
        """

        # Reseteamos La recompensa del episodio
        self.episode_reward = 0

    # Devuelve que acción tomar segun un estado de juego
    def select_action(self, state: 'State'):
        """ Seleccionamos una acción del espacio de acciones, esta decisión puede ser tomada:
        1. De forma aleatoria (greedy)
        2. Utilizando la Policy_DQN
        """

        # Transformamos el estado a un tensor (utilizado por RedDQN)
        state_tensor = self.state_to_tensor(state)

        # Probabilidad de hacer movimiento aleatorio (epsilon)
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim-1)

        # Probabilidad de hacer un movimiento a traves de Policy (1-epsilon)
        else: 
            with torch.no_grad():
                q_values = self.policy_dqn(state_tensor.unsqueeze(0))
                action = torch.argmax(q_values).item()
        
        return action

    def calculate_reward(self, state: 'State', action: int, next_state: 'State') -> int:
        """Función de recompensa. Se utiliza al entrenar el modelo.
        OBS: esta debe ser modificada hasta optimizar el modelo.
        """
        reward = 0

        if action == 0 or action == 1: 
            reward += REWARD_FOR_WALKING
        else:
            reward += REWARD_FOR_JUMPING

        delta_height = next_state.height - state.height
        if delta_height < 0:
            how_much_down = -1 * delta_height
            reward += REWARD_FOR_GOING_DOWN * how_much_down
        else:
            how_much_up = delta_height
            reward += REWARD_FOR_GOING_UP * how_much_up

        delta_max_height = next_state.max_height - state.max_height
        if delta_max_height > 0:
            reward += REWARD_FOR_NEW_MAX_HEIGHT * delta_max_height  + STATIC_REWARD_FOR_NEW_MAX_HEIGHT

        if next_state.win:
            reward += REWARD_FOR_WIN

        return reward

    # Entrena al modelo sabiendo que acción tomó en un estado, y a que otro estado llevó
    # Acá podria por ejemplo: Calcular recompensa, actualizar recompensa acumulada, etc.
    def train(self, state: 'State', action: int, next_state: 'State'):
        """ Entrenamos a la Red Policy de la siguiente manera:
        1. Calculamos Recompensa
        2. Si se está entrenando: 
        2.a) Añadimos transición a Replay Memory
        2.b) Entrenamos la Policy DQN.
        3. Añadimos recompenza a la recompensa por episodio
        """
        
        # FIXME: .done tiene que ser cambiado por otra cosa jejeje. Posiblemente poner un estado TOPE

        # 1. Calcular Recompensa
        reward = self.calculate_reward(state, action, next_state)

        done = next_state.win

        # Transformamos los estados a tensores
        state_tensor = self.state_to_tensor(state)
        next_state_tensor = self.state_to_tensor(next_state)

        # 2.  
        # Se mejora la red con back propagation, esto igualmente se puede cambiar a END_EPISODE si se quiere hacer
        # de forma menos recurrente, OJO: Ahora los episodios deberias ser más o menos lentos
        if self.is_training:

            # 2.a
            self.memory.append((state_tensor, action, next_state_tensor, reward, done))
            self.step_count += 1

           # Si suficiente experiencia a sido recolectada
           # 2.b
            if len(self.memory) > self.batch_size:

                # Sample de memory
                batch = self.memory.sample(self.batch_size)

                # Optimize
                self.optimize(batch, self.policy_dqn, self.target_dqn)

                # Sincronizar Redes
                if self.step_count > self.network_sync_rate:
                    self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                    self.step_count = 0

        # Sumamos recompensa actual a episodio
        self.episode_reward += reward
        

    def optimize(self, batch, policy_dqn, target_dqn):
        """
        Funcion para aplicar la optimización. (Vodoo)
        """

        # Del sample, recuperamos las transiciones
        states, actions, new_states, rewards, terminations = zip(*batch)

        states = torch.stack(states)

        actions = torch.tensor(actions, dtype=torch.long, device=self.device)

        new_states = torch.stack(new_states)

        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)

        terminations = torch.tensor(terminations, dtype=torch.float, device=self.device)

        assert torch.all((actions >= 0) & (actions < self.action_dim)), "Actions out of bounds"

        with torch.no_grad():

            # Calcular target Q values (Returns esperados)
            best_actions_from_policy = self.policy_dqn(new_states).argmax(dim=1)

            assert torch.all((best_actions_from_policy >= 0) & (best_actions_from_policy < self.action_dim)), \
            "Best actions out of bounds"

            # Funcion DDQN
            target_q =  rewards + (1- terminations) * self.discount_factor_gamma * \
                        target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()

        
        # Calcular Q values desde Policy actual
        current_q = policy_dqn(states).gather(dim=1, index= actions.unsqueeze(dim=1)).squeeze()

        assert current_q.shape == target_q.shape, \
        f"Shape mismatch: current_q {current_q.shape}, target_q {target_q.shape}"

        # Computar Loss 
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients (backpropagation)
        self.optimizer.step()       # Update Network parameters i.e. weigths and biases


    def end_episode(self):
        """Terminar el episodio, entregar datos de episodio y disminuir epsilon (progresión geometrica)
        """
        # Modificamos el Epsilon
        if (self.is_training):
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        print("Current epsilon: {epsilon} ,Acumulative reward: {reward}".format(reward = self.episode_reward, epsilon = self.epsilon))

    def load(self, path):
        """ Cargamos datos de entrenamientos previos. Esto seria:
        1. Si NO estamos entrenando solo red policy
        2. Si estamos entrenando: red policy, última red target, optimizer, epsilon
        """
        
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
        """Solo si estamos entrenando guardamos, toda la red.
        """
        
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
