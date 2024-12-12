'''
Agente de prueba. Tiene una lista de acciones a realizar y las realiza conforme la simulación avanza un 'paso'
'''

##---------------------------------------------------------####
##        OBSOLETO: LO VOY A CORREGIR DESPUES               ###
##---------------------------------------------------------####

from Train import Agent, State
from typing import List, Tuple

'''
Keywords. Para mayor legibilidad
'''
RIGHT : int = 0
LEFT : int = 1
SPACE_RIGHT : int = 2
SPACE_LEFT : int = 3
SPACE : int = 4
IDLE : int = 5

'''
Lista de acciones que hará el agente.
Cada elemento de la lista es de la forma (a, t), con:
    a: Una acción del espacio de acciones
    t: Por cuantos 'pasos' repetira dicha acción
'''
ACTIONS_LIST : List[Tuple[int, int]] = [
    (SPACE_RIGHT, 30), (RIGHT, 1),  
    (RIGHT, 20),                    
    (SPACE_LEFT, 30), (LEFT, 1),    
    (LEFT, 40),                     
    (SPACE_RIGHT, 30), (RIGHT, 1),
]
# Despues de terminar un salto es importante especificar en que dirección
# de hecho, SPACE_RIGHT (por ejemplo) no implica que se saltará hacia la derecha
# tan sólo que para ese 'paso' se estará pulsando la tecla de salto y la flecha derecha
# Pero la dirección la determina el 'paso' siguiente a liberar el salto

class ListAgent(Agent):

    def start_episode(self):
        self.actions : List[Tuple[int, int]] = []
        actions_list_copy = ACTIONS_LIST.copy()

        while len(actions_list_copy) != 0:
            action, repeat = actions_list_copy.pop()
            for i in range(repeat):
                self.actions.append(action)
    
    def select_action(self, state):

        if len(self.actions) != 0:
            return self.actions.pop()
        else:
            return IDLE
