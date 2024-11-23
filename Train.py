from JumpKing import JKGame
import numpy as np
import numpy.typing as npt
import cv2
import struct

'''
Este archivo tiene la función de generalizar lo que es entrenar un agente,
con el objetivo de idealmente programar un agente con cierta independencia del resto del programa

El repositorio original esta limitado en ciertas funcionalidades (principalmente estados de juego)
por lo tanto acá se definen clases extras como State, Environment que añaden estas funcionalidades
'''

'''
---- Parametros, constantes y variables importantes ----
'''
# Parametros de depuración
DEBUG_LEVEL_MATRIX = True # Imprime la matriz del nivel en una ventana aparte ¡Ralentiza mucho el programa!

# Parametros (o constante?) del espacio de acciones
# ¡No se esta usando en ningun lado! tampoco creo que sea necesario. Sirve más de documentación que otra cosa
ACTION_SPACE : dict[int, str] = {
0: 'right',
1: 'left',
2: 'right+space',
3: 'left+space',
# 4: 'space',   # No es util considerar esta acción, por ahora al menos.
# 5: 'idle',    # No es util considerar esta acción
}

# Parametros de la matriz del nivel
LEVEL_MATRIX_HORIZONTAL_SIZE = 48
LEVEL_MATRIX_VERTICAL_SIZE = 36

# Constantes del nivel
LEVEL_HORIZONTAL_SIZE = 480 # Cuanto mide el nivel horizontalmente, es una constante del repositorio original, no modificable
LEVEL_VERTICAL_SIZE = 360 # Cuanto mide el nivel verticalmente, es una constante del repositorio original, no modificable





if DEBUG_LEVEL_MATRIX:
    cv2.namedWindow("DEBUG_LEVEL_MATRIX", cv2.WINDOW_NORMAL)

# Función que obtiene una matriz que representa las colisiones del nivel
# Una celda vale 1 si hay una hitbox en la sección correspondiente del nivel. 0 en otro caso
# Recibe una instancia de Environment y del numero del nivel en cuestion.
def get_level_matrix(env, level):
    matrix = np.zeros((
                      LEVEL_MATRIX_VERTICAL_SIZE,
                      LEVEL_MATRIX_HORIZONTAL_SIZE),
                      dtype=np.uint8)
    
    platforms = env.game.levels.platforms.rectangles.levels[level]

    for p in platforms:
        x = round(p[0] * (LEVEL_MATRIX_HORIZONTAL_SIZE/LEVEL_HORIZONTAL_SIZE))
        if x < 0: x = 0
        elif x >= LEVEL_MATRIX_HORIZONTAL_SIZE: x = LEVEL_MATRIX_HORIZONTAL_SIZE - 1

        y = round(p[1] * (LEVEL_MATRIX_VERTICAL_SIZE/LEVEL_VERTICAL_SIZE))
        if y < 0: y = 0
        elif y >= LEVEL_MATRIX_VERTICAL_SIZE: y = LEVEL_MATRIX_VERTICAL_SIZE - 1

        w = round(p[2] * (LEVEL_MATRIX_HORIZONTAL_SIZE/LEVEL_HORIZONTAL_SIZE))
        if x + w >= LEVEL_MATRIX_HORIZONTAL_SIZE: w = LEVEL_MATRIX_HORIZONTAL_SIZE - 1 - x

        h = round(p[3] * (LEVEL_MATRIX_VERTICAL_SIZE/LEVEL_VERTICAL_SIZE))
        if y + h >= LEVEL_MATRIX_VERTICAL_SIZE: h = LEVEL_MATRIX_VERTICAL_SIZE - 1 - y

        slope = p[4]

        if slope == 0:
            matrix[y:y+h, x:x+w] = 1
        elif slope == (1, 1):
            triangle_mask = np.tril(np.ones((h, w)))
            reflected_triangle_mask = triangle_mask[:, ::-1]
            matrix[y:y+h, x:x+w] = reflected_triangle_mask 
        elif slope == (-1, 1):
            triangle_mask = np.tril(np.ones((h, w)))
            matrix[y:y+h, x:x+w] = triangle_mask 
        elif slope == (1, -1):
            triangle_mask = np.triu(np.ones((h, w)))
            reflected_triangle_mask = triangle_mask[:, ::-1]
            matrix[y:y+h, x:x+w] = reflected_triangle_mask 
        elif slope == (-1, -1):
            triangle_mask = np.triu(np.ones((h, w)))
            matrix[y:y+h, x:x+w] = triangle_mask 

    if DEBUG_LEVEL_MATRIX:
        frame = cv2.resize(matrix * 255, (400, 300))
        cv2.imshow("DEBUG_LEVEL_MATRIX", frame)
        cv2.waitKey(1) #ms
    
    return matrix

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
    def __init__(self, steps_per_episode):
        self.game = JKGame(max_step=steps_per_episode)
    
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
    def __init__(self,
                 agent : Agent,
                 steps_per_episode=1000,
                 numbers_of_episode=100000):
        
        self.agent = agent
        self.steps_per_episode = steps_per_episode
        self.numbers_of_episode = numbers_of_episode
        self.env = Environment(self.steps_per_episode)

    def run(self):

        for i in range(self.numbers_of_episode):

            self.agent.start_episode()

            state = self.env.reset()

            while not state.done:

                action = self.agent.select_action(State.encode(state))

                if action not in ACTION_SPACE.keys() : 
                    raise ValueError("Given action not in Action Space!")

                next_state = self.env.step(action)

                self.agent.train(State.encode(state), action, State.encode(next_state))

                state = next_state
            
            self.agent.end_episode()

'''
TODO: Para iniciar el juego con función de evaluar un agente ya entrenado
'''
class Evaluate():
    pass

    

    