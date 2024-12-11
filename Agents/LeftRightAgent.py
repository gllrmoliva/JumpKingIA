'''
Agente de prueba. ¡Crashea fácil al cambiar parametros, lo hice sólo para probar algunas cosas rápidamente!
Alterna entre ir hacia la derecha o izquierda conforme detecta algo al frente
Lo importante es que muestra un ejemplo de recibir el estado codificado, decodificarlo, y hacer algo con esa información.
'''

from Train import Agent, State
from Matrix import area_to_matrix_area

# Constantes (¡Modificarlas puede romper el funcionamiento!)
DETECTION_WIDTH = 50
DETECTION_HEIGHT = 20
VERTICAL_DETECTION_PHASE = 5

class LeftRightAgent(Agent):

    def start_episode(self):
        self.direction = 'right'

    def select_action(self, state):

        # transformar coordenada en celda de la matriz de colisiones
        x, y, w, h = area_to_matrix_area(state.x, state.y + VERTICAL_DETECTION_PHASE, DETECTION_WIDTH, DETECTION_HEIGHT)

        if self.direction == 'right':
            right_area = state.level_matrix[y : y+h, x : x+w].sum()  
            if right_area > 0: # ¿hay algo al frente?
                self.direction = 'left'
        elif self.direction == 'left':
            left_area = state.level_matrix[y : y+h, x-w+1 : x+1].sum()
            if left_area > 0: # ¿hay algo al frente?
                self.direction = 'right'
    
        if self.direction == 'left': return 1
        else: return 0