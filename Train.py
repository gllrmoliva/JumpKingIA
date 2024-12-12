from JumpKing import JKGame
import struct
import csv
from pathlib import Path
from Constants import *
from Matrix import *
from datetime import datetime

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
    level : int = None                                  # Número del nivel actual
    x : int = None                                      # Coordenada x del King. Aumenta de izquierda a derecha
    y : int = None                                      # Coordenada y del King. ¡Aumenta de arriba hacia abajo!
    height : int = None                                 # Altura total del King, incluyendo niveles anteriores. ¡Aumenta de abajo hacia arriba!
    max_height : int = None                             # Altura total máxima alcanzada en el episodio
    max_height_last_step : int = None                   # Altura total máxima alcanzada durante el último paso de juego
    jumpCount : int = None                              # Cuantos pasos se ha esta 'cargando' el salto
    done : bool = None                                  # Acabo el episodio
    level_matrix : npt.NDArray[np.uint64] = None        # Matriz de colisiones del nivel
    next_level_matrix : npt.NDArray[np.uint64] = None   # Matriz de colisiones del nivel siguiente

    '''
    Busca los valores que describen el estado y los almacena en los atributos
    Recibe como parametro:
        game: instancia de JKGame. Esto es porque es la forma de acceder a variables del repositorio original
    '''                  
    @staticmethod
    def get_state_from_game(game : JKGame) -> 'State':
        state = State()

        state.level = game.king.levels.current_level
        if DEBUG_OLD_COORDINATE_SYSTEM:
            state.x = game.king.x + 5 # numero magico
            state.y = game.king.y + 9 # numero magico
        else:
            state.x = round(game.king.rect_x)
            state.y = round(game.king.rect_y)
        
        state.height = game.height
        state.max_height = game.max_height
        state.max_height_last_step = game.max_height_last_step
        state.jumpCount = game.king.jumpCount
        state.done = game.done
        state.level_matrix = get_level_matrix(game, state.level, debug=True, position_rounding=round, thickness_rounding=ceil)
        if state.level + 1 <= game.king.levels.max_level:
            state.next_level_matrix = get_level_matrix(game, state.level + 1,
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
        self.game = JKGame(steps_per_episode=steps_per_episode, steps_per_seconds=steps_per_second)
    
    def reset(self):
        self.game.reset()
        return State.get_state_from_game(self.game)

    def step(self, action):
        self.game.step(action)
        return State.get_state_from_game(self.game)

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
                 agent_loadpath=None,
                 agent_savepath=None,
                 csv_agentname="UNNAMED",
                 csv_savepath=None):
        
        self.agent : Agent = agent
        self.steps_per_episode = steps_per_episode
        self.numbers_of_episode = number_of_episodes
        self.agent_loadpath = agent_loadpath
        self.agent_savepath = agent_savepath
        self.env : Environment = Environment(self.steps_per_episode, steps_per_second)

        self.csv : CSV = CSV(csv_agentname, csv_savepath, self)

    def run(self):

        self.episode = 1
        
        if self.agent_loadpath != None: self.agent.load(str(Path(self.agent_loadpath)))

        while self.episode <= self.numbers_of_episode:

            self.agent.start_episode()
            
            '''Entrega de parámetros individuales del estado actual'''
            self.state_level = self.env.reset().level
            self.state_x = self.env.reset().x
            self.state_y = self.env.reset().y
            self.state_height = self.env.reset().height
            self.state_max_height = self.env.reset().max_height
            self.state_max_height_last_step = self.env.reset().max_height_last_step
            self.state_done = self.env.reset().done
            self.state_jumpCount = self.env.reset().jumpCount

            self.step = 0

            while not self.state_done:

                self.csv.update()

                action = self.agent.select_action(self.state_level, self.state_x, self.state_y, self.state_jumpCount)

                if action not in ACTION_SPACE.keys() : 
                    raise ValueError("Given action not in Action Space!")

                '''Entrega de parámetros individuales del estado siguiente'''
                self.next_state_level = self.env.step(action).level
                self.next_state_x = self.env.step(action).x
                self.next_state_y = self.env.step(action).y
                self.next_state_height = self.env.step(action).height
                self.next_state_max_height = self.env.step(action).max_height
                self.next_state_max_height_last_step = self.env.step(action).max_height_last_step
                self.next_state_done = self.env.step(action).done
                self.next_state_jumpCount = self.env.step(action).jumpCount

                self.agent.train(action, self.state_level, self.state_x, self.state_y, self.state_max_height, 
                                 self.state_max_height_last_step, self.state_done, self.state_jumpCount,
                                 self.next_state_level, self.next_state_x, self.next_state_y, self.next_state_max_height,
                                 self.next_state_done)

                self.state_level = self.next_state_level
                self.state_x = self.next_state_x
                self.state_y = self.next_state_y
                self.state_height = self.next_state_height
                self.state_max_height = self.next_state_max_height
                self.state_max_height_last_step = self.next_state_max_height_last_step
                self.state_done = self.next_state_done
                self.state_jumpCount = self.next_state_jumpCount

                self.step += 1
            
            self.agent.end_episode()

            if self.agent_savepath != None: self.agent.save(str(Path(self.agent_savepath)))

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
        self.path = str(Path(savepath))
        self.train = train
        
        self.file = open(self.path, mode='w', newline='')
        self.writer = csv.writer(self.file)

        self.writer.writerow(['AGENT_NAME',
                              'EPISODE',
                              'STEP',
                              'DATE',
                              'TIME',
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
                   self.train.state_height,
                   self.train.state_max_height,
                   self.train.state_max_height_last_step,
                   self.train.state_x,
                   self.train.state_y]

            self.writer.writerow(row)

    def end(self):
        self.file.close()

    

    