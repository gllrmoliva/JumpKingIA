'''
Agente de prueba. ¡Crashea fácil al cambiar parametros Train.py, lo hice sólo para probar algunas cosas rápidamente!
Alterna entre ir hacia la derecha o izquierda conforme detecta algo al frente
Lo importante es que muestra un ejemplo de recibir el estado codificado, decodificarlo, y hacer algo con esa información.
'''

from Train import Agent, State

# Variables
from Train import LEVEL_HORIZONTAL_SIZE, LEVEL_VERTICAL_SIZE, LEVEL_MATRIX_HORIZONTAL_SIZE, LEVEL_MATRIX_VERTICAL_SIZE

XRANGE = round(50 * (LEVEL_MATRIX_HORIZONTAL_SIZE / LEVEL_HORIZONTAL_SIZE))
YRANGE = round(5 * (LEVEL_MATRIX_VERTICAL_SIZE / LEVEL_VERTICAL_SIZE))

class DumbAgent(Agent):

    def start_episode(self):
        self.direction = 'right'

    def select_action(self, coded_state):
        state = State.decode(coded_state)

        # transformar x,y en una fila,columna de la matriz del nivel
        x = int(state.x * (LEVEL_MATRIX_HORIZONTAL_SIZE / LEVEL_HORIZONTAL_SIZE))
        y = int(state.y * (LEVEL_MATRIX_VERTICAL_SIZE / LEVEL_VERTICAL_SIZE))

        if self.direction == 'right':
            right_area = state.level_matrix[y+YRANGE:y+YRANGE+2, x:x+XRANGE].sum()  
            if right_area > 0: # ¿hay algo al frente?
                self.direction = 'left'
        elif self.direction == 'left':
            left_area = state.level_matrix[y+YRANGE:y+YRANGE+2, x-XRANGE+1:x+1].sum()
            if left_area > 0: # ¿hay algo al frente?
                self.direction = 'right'
    
        if self.direction == 'left': return 1
        else: return 0