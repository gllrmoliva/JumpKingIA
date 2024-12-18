from JumpKing import JKGame
import struct
import csv
from pathlib import Path
from Constants import *
from Matrix import *
from datetime import datetime
from typing import Tuple
import ActionSpace


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

    # Atributos
    level : int = None                                  # Número del nivel actual. Desde 0 a 42
    max_level: int = None                               # Nivel máximo alcanzado en el episodio. Desde 0 a 42
    x : int = None                                      # Coordenada x del King. Aumenta de izquierda a derecha
    y : int = None                                      # Coordenada y del King. ¡Aumenta de arriba hacia abajo!
    height : int = None                                 # Altura total del King, incluyendo niveles anteriores. ¡Aumenta de abajo hacia arriba!
    max_height : int = None                             # Altura total máxima alcanzada en el episodio
    max_height_last_step : int = None                   # Altura total máxima alcanzada durante el último paso de juego
    jumpCount : int = None                              # Cuantos pasos se ha esta 'cargando' el salto
    done : bool = None                                  # Es el ultimo estado del episodio
    win : bool = None                                   # Se llegó al último nivel. Notar que: win = True => done = True

    if not NO_LEVEL_MATRIX:
        level_matrix : npt.NDArray[np.uint64] = None        # Matriz de colisiones del nivel
        next_level_matrix : npt.NDArray[np.uint64] = None   # Matriz de colisiones del nivel siguiente

    # Atributos privados
    _normalized = False

    '''
    Busca los valores que describen el estado y los almacena en los atributos
    Recibe como parametro:
        game: instancia de JKGame. Esto es porque es la forma de acceder a variables del repositorio original
    '''                  
    @staticmethod
    def get_state_from_env(env: 'Environment') -> 'State':
        state = State()

        state.level = env.game.king.levels.current_level
        state.max_level = env.max_level

        if DEBUG_OLD_COORDINATE_SYSTEM:
            state.x = env.game.king.x + 5 # numero magico
            state.y = env.game.king.y + 9 # numero magico
        else:
            state.x = round(env.game.king.rect_x)
            state.y = round(env.game.king.rect_y)
        
        state.height = env.game.height
        state.max_height = env.game.max_height
        state.max_height_last_step = env.game.max_height_last_step
        state.jumpCount = env.game.king.jumpCount
        state.done = env.done
        state.win = env.win

        if not NO_LEVEL_MATRIX:
            state.level_matrix = get_level_matrix(env.game, state.level, debug=True,
                                                position_rounding=round, thickness_rounding=ceil)
            if state.level + 1 <= GAME_MAX_LEVEL:
                state.next_level_matrix = get_level_matrix(env.game, state.level + 1,
                                                        matrix_width=NEXT_LEVEL_MATRIX_HORIZONTAL_SIZE,
                                                        matrix_height=2*NEXT_LEVEL_MATRIX_VERTICAL_SIZE,
                                                        position_rounding=round, thickness_rounding=ceil,
                                                        )[NEXT_LEVEL_MATRIX_VERTICAL_SIZE : ] # Solamente la mitad de abajo

            else:
                state.next_level_matrix = np.ones((
                                            NEXT_LEVEL_MATRIX_HORIZONTAL_SIZE,
                                            NEXT_LEVEL_MATRIX_VERTICAL_SIZE),
                                            dtype=np.uint8)
                
        return state

    '''
    ATRIBUTOS NORMALIZADOS: Cuyo valor esta entre 0 y 1
    ¡Notar que en este proceso se transforman los valores a float!
    '''
    @property
    def level_normalized(self): return self.level / GAME_MAX_LEVEL
    @property
    def max_level_normalized(self): return self.max_level / GAME_MAX_LEVEL
    @property
    def x_normalized(self): return self.x / LEVEL_HORIZONTAL_SIZE
    @property
    def y_normalized(self): return self.y / LEVEL_VERTICAL_SIZE
    @property
    def height_normalized(self): return self.height / GAME_MAX_HEIGHT
    @property
    def max_height_normalized(self): return self.max_height / GAME_MAX_HEIGHT
    @property
    def max_height_last_step_normalized(self): return self.max_height_last_step / GAME_MAX_HEIGHT
    @property
    def jumpCount_normalized(self): return self.jumpCount / JUMPCOUNT_MAX
    

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
    def select_action(self, state: State):
        pass
    # Entrena al modelo sabiendo que acción tomó en un estado, y a que otro estado llevó
    # Acá podria por ejemplo: Calcular recompensa, actualizar recompensa acumulada, etc.
    def train(self, state: State, action: int, next_state: State):
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
        self.game = JKGame(steps_per_seconds=steps_per_second)
        self.steps_per_episode = steps_per_episode
        self.step_counter = 0
        self.max_level = 0
        self.done = False
        self.win = False
    
    def reset(self):
        self.game.reset()
        self.step_counter = 0
        self.max_level = 0
        self.done = False
        self.win = False

        return State.get_state_from_env(self)

    def step(self, action):

        (elemental_action, repeat, action_name) = action

        for i in range(repeat + 1):
            if i < repeat:
                if      elemental_action == ActionSpace.LEFT:           self._elemental_step(ActionSpace.LEFT)
                elif    elemental_action == ActionSpace.RIGHT:          self._elemental_step(ActionSpace.RIGHT)
                elif    elemental_action == ActionSpace.SPACE_LEFT:     self._elemental_step(ActionSpace.SPACE)
                elif    elemental_action == ActionSpace.SPACE_RIGHT:    self._elemental_step(ActionSpace.SPACE)
                elif    elemental_action == ActionSpace.SPACE:          self._elemental_step(ActionSpace.SPACE)
            else: # Last repeat
                if      elemental_action == ActionSpace.LEFT:           break
                elif    elemental_action == ActionSpace.RIGHT:          break
                elif    elemental_action == ActionSpace.SPACE_LEFT:     self._elemental_step(ActionSpace.LEFT)
                elif    elemental_action == ActionSpace.SPACE_RIGHT:    self._elemental_step(ActionSpace.RIGHT)
                elif    elemental_action == ActionSpace.SPACE:          self._elemental_step(ActionSpace.IDLE)

        self.step_counter += 1

        current_level = self.game.king.levels.current_level
        if current_level == EPISODE_MAX_LEVEL or current_level == GAME_MAX_LEVEL:
            self.done = True
            self.win = True
        elif self.step_counter >= self.steps_per_episode:
            self.done = True
        else:
            self.done = False

        if self.max_level < current_level: self.max_level = current_level

        return State.get_state_from_env(self)
    
    # Método privado, ignorar
    def _elemental_step(self, elemental_action):
        self.game.step(elemental_action)


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
        agent_loadpath: Ruta donde cargar el entrenamiento del agente
        agent_savepath: Ruta donde guardar el entrenamiento del agente
        csv_agentname: Nombre del agente en el .csv generado
        csv_savepath: Ruta donde guardar el .csv con las estadisticas del entrenamiento
    '''
    def __init__(self,
                 agent : Agent,
                 steps_per_episode=STEPS_PER_EPISODE,
                 number_of_episodes=NUMBER_OF_EPISODES,
                 steps_per_second=STEPS_PER_SECOND,
                 action_space=None,
                 agent_loadpath=None,
                 agent_savepath=None,
                 csv_agentname="UNNAMED",
                 csv_savepath=None):
        
        self.agent : Agent = agent

        self.env : Environment = Environment(steps_per_episode, steps_per_second)
        self.steps_per_episode = steps_per_episode
        self.numbers_of_episode = number_of_episodes

        self.agent_loadpath = agent_loadpath
        self.agent_savepath = agent_savepath
        self.csv : CSV = CSV(csv_agentname, csv_savepath, self)

        if action_space == None: raise ValueError("Action space needed!")
        self.action_space : dict[int, Tuple[int, int, str]] = action_space

        
        

    def run(self):

        self.episode = 1
        
        if self.agent_loadpath != None: self.agent.load(str(Path(self.agent_loadpath + ".pth")))

        while self.episode <= self.numbers_of_episode:

            self.agent.start_episode()
            self.state = self.env.reset()
            self.step = 0

            while not self.state.done:

                self.csv.update()

                action = self.agent.select_action(self.state)
                if action not in self.action_space.keys() : 
                    raise ValueError("Given action not in Action Space!")
                next_state = self.env.step(self.action_space[action])             
                self.agent.train(self.state, action, next_state)
                self.state = next_state

                self.step += 1
            
            self.agent.end_episode()

            if self.episode % SAVE_COOLDOWN == 0:
                if self.agent_savepath != None: self.agent.save(str(Path(self.agent_savepath + f"_{self.episode}.pth")))

            print("Episodio {} Terminado\n".format(self.episode))
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
        self.path = str(Path(savepath+".csv"))
        self.train = train
        
        self.file = open(self.path, mode='w', newline='')
        self.writer = csv.writer(self.file)

        self.writer.writerow(['AGENT_NAME',
                              'EPISODE',
                              'STEP',
                              'DATE',
                              'TIME',
                              'LEVEL',
                              'MAX_LEVEL',
                              'HEIGHT',
                              'MAX_HEIGHT',
                              'MAX_HEIGHT_LAST_STEP',
                              'X',
                              'Y'])
    
    def update(self):
        if self.train.step % CSV_COOLDOWN == 0:

            now = datetime.now()

            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            row = [self.agentname,
                   self.train.episode,
                   self.train.step,
                   date,
                   time,
                   self.train.state.level,
                   self.train.state.max_level,
                   self.train.state.height,
                   self.train.state.max_height,
                   self.train.state.max_height_last_step,
                   self.train.state.x,
                   self.train.state.y]

            self.writer.writerow(row)

    def end(self):
        self.file.close()

    

    