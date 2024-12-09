from JumpKing import JKGame
import struct
import csv
from pathlib import Path
from Constants import *
from Matrix import *

'''
Este archivo tiene la función de generalizar lo que es entrenar un agente,
con el objetivo de idealmente programar un agente con cierta independencia del resto del programa

El repositorio original esta limitado en ciertas funcionalidades (principalmente estados de juego)
por lo tanto acá se definen clases extras como State, Environment que añaden estas funcionalidades
'''




"""
La representación de un estado en el repositorio original es bastante limitada
Está clase modela un estado y 
"""
class State():

    # Attributes
    level : int = None
    x : int = None
    y : int = None
    jumpCount : int = None
    done : bool = None
    level_matrix : npt.NDArray[np.uint64] = None

    '''
    Busca los valores que describen el estado y los almacena en los atributos
    Recibe como parametro:
        env: instancia de Environment. Esto es porque es la unica forma de acceder a variables del repositorio original
        built_in_state : La representación de un estado en el repositorio original
                         Conformado por: Nivel actual, x, y, contador del sato
        done: Verdadero si se llegó al final del episodio
    '''                  
    @staticmethod
    def get_state_from_built_in_state(env, built_in_state, done) -> 'State':
        state = State()

        state.level = built_in_state[0]
        state.x = built_in_state[1]
        state.y = built_in_state[2]
        state.jumpCount = built_in_state[3]
        state.done = done
        state.level_matrix = get_level_matrix(env, state.level)

        return state
    
    '''
    Codifica un estado como una serie de bytes (Tengo entendido que es obligatorio para meterlo a una red neuronal)
    El tamaño de la codificación es fijo.
    ¡Agregar/eliminar/modificar un atributo de la clase implica tener que cambiar este metodo!
    '''
    @staticmethod
    def encode(state : 'State') -> bytes:
        # Pack integers and boolean
        packed_data = struct.pack(
            '4i?',  # 4 integers and 1 boolean
            state.level, state.x, state.y, state.jumpCount, state.done
        )
        
        # Flatten the 2D numpy matrix (uint8) and pack it as bytes
        flattened_matrix = state.level_matrix.flatten()
        packed_matrix = struct.pack(f'{flattened_matrix.size}B', *flattened_matrix)  # 'B' for uint8 (1 byte per element)
        
        # Combine everything into one packed binary string
        final_packed_data = packed_data + packed_matrix
        return final_packed_data
    
    '''
    Decodifica la codificación de encode(), devolviendo una instancia de State cuyas variables corresponden a los valores codificados
    ¡Agregar/eliminar/modificar un atributo de la clase implica tener que cambiar este metodo!
    '''
    @staticmethod
    def decode(coded_state : bytes) -> 'State':
        # Unpack the first part (4 integers and 1 boolean)
        unpacked_data = struct.unpack('4i?', coded_state[:17])  # 4 integers (4*4 bytes) + 1 boolean (1 byte)
        level, x, y, jumpCount, done = unpacked_data
        
        # Unpack the 2D matrix (after the header part)
        matrix_size = LEVEL_MATRIX_VERTICAL_SIZE * LEVEL_MATRIX_HORIZONTAL_SIZE
        matrix_data = coded_state[17:]
        unpacked_matrix = struct.unpack(f'{matrix_size}B', matrix_data)  # 'B' for uint8 (1 byte per element)
        
        # Convert the unpacked matrix into a 2D numpy array
        level_matrix = np.array(unpacked_matrix, dtype=np.uint8).reshape((LEVEL_MATRIX_VERTICAL_SIZE, LEVEL_MATRIX_HORIZONTAL_SIZE))

        state = State()
        
        state.level = level
        state.x = x
        state.y = y
        state.jumpCount = jumpCount
        state.done = done
        state.level_matrix = level_matrix
        
        return state

'''
Interfaz que modela un agente.
Cualquier agente especifico que se haga deberia de heredar de esta clase.
Es un contrato de que funcionalidades serán llamadas cuando se haga el entrenamiento.
'''
class Agent():
    # Se llama cuando inicia un episodio.
    # Deberia, por ejemplo, inicializar variables locales a un solo episodio
    def start_episode(self):
        pass
    # Devuelve que acción tomar segun un estado de juego
    def select_action(self, coded_state: bytes):
        pass
    # Entrena al modelo sabiendo que acción tomó en un estado, y a que otro estado llevó
    # Acá podria por ejemplo: Calcular recompensa, actualizar recompensa acumulada, etc.
    def train(self, coded_state: bytes, action: int, coded_next_state: bytes):
        pass
    # Se llama cuando acaba el episodio.
    def end_episode(self):
        pass
    # Para cargar datos de entrenamientos previos
    def load(self, path):
        pass
    # Para guardar datos del entrenamiento actual
    def save(self, path):
        pass

'''
Es una clase que 'envuelve' a JKGame para que sus metodos devuelvan los estados como instancias de State()
'''
class Environment():
    def __init__(self, steps_per_episode, steps_per_second):
        self.game = JKGame(steps_per_episode=steps_per_episode, steps_per_seconds=steps_per_second)
    
    def reset(self):
        done, state = self.game.reset()
        return State.get_state_from_built_in_state(self, state, done)

    def step(self, action):
        next_state , reward, done = self.game.step(action)
        return State.get_state_from_built_in_state(self, next_state, done)

'''
Para iniciar el juego con función de entrenar un agente
'''
class Train():
    '''
    Parametros:
        agent: Qué agente (es decir, que método de aprendizaje) se va a usar. Recibe una instancia.
        steps_per_episode: Cuantos 'pasos' realizar por episodio
        number_of_episodes: Cuantos episodios a realizar
		steps_per_seconds: Cantidad de 'pasos' de la simulación que se realizan en un segundo
			-1: Desbloqueado, ejecuta al mayor ritmo que puede.
        csv_agentname: Nombre del agente en el .csv generado
        csv_savepath: Ruta donde guardar el .csv con las estadisticas del entrenamiento
    '''
    def __init__(self,
                 agent : Agent,
                 steps_per_episode=STEPS_PER_EPISODE,
                 number_of_episodes=NUMBER_OF_EPISODES,
                 steps_per_second=STEPS_PER_SECOND,
                 csv_agentname="UNNAMED",
                 csv_savepath=None):
        
        self.agent : Agent = agent
        self.steps_per_episode = steps_per_episode
        self.numbers_of_episode = number_of_episodes
        self.env : Environment = Environment(self.steps_per_episode, steps_per_second)

        self.csv : CSV = CSV(csv_agentname, csv_savepath, self)

    def run(self):

        self.episode = 1

        while self.episode <= self.numbers_of_episode:

            self.agent.start_episode()

            self.state = self.env.reset()

            self.step = 0

            while not self.state.done:

                self.csv.update()

                action = self.agent.select_action(State.encode(self.state))

                if action not in ACTION_SPACE.keys() : 
                    raise ValueError("Given action not in Action Space!")

                next_state = self.env.step(action)

                self.agent.train(State.encode(self.state), action, State.encode(next_state))

                self.state = next_state

                self.step += 1
            
            self.agent.end_episode()

            self.episode += 1
        
        self.csv.end()

'''
TODO: Para iniciar el juego con función de evaluar un agente ya entrenado
'''
class Evaluate():
    pass

'''
Clase para separar la lógica de la generación de CSV
'''
class CSV():
    def __init__(self, agentname, savepath, train : Train):
        self.agentname = agentname
        self.path = str(Path(savepath))
        self.train = train
        
        self.file = open(self.path, mode='w', newline='')
        self.writer = csv.writer(self.file)
        self.max_height = -1
        self.current_episode = -1

        self.writer.writerow(['AGENT_NAME', 'EPISODE', 'STEP', 'MAX_HEIGHT'])
    
    def update(self):
        if self.train.step % CSV_COOLDOWN == 0:

            if self.current_episode != self.train.episode:
                self.current_episode = self.train.episode
                self.max_height = -1

            current_height = self.train.state.y + self.train.state.level * LEVEL_VERTICAL_SIZE
            self.max_height = max(self.max_height, current_height)

            self.writer.writerow([self.agentname, self.train.episode, self.train.step, self.max_height])

    def end(self):
        self.file.close()

    

    