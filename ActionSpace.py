from typing import Tuple
from Constants import JUMPCOUNT_MAX, ACTION_SPACE_SIZE, INCLUDE_VERTICAL_SPACE, WALKING_LENGTH

'''
Valores repositorio orignal ¡No usar! ¡¡Es solo para tener de referencia!!
'''
RIGHT : int = 0
LEFT : int = 1
SPACE_RIGHT : int = 2
SPACE_LEFT : int = 3
SPACE : int = 4
IDLE : int = 5

'''
Hacer esto así por flojera más que nada
'''
DEFAULT_SEQUENCE = [30, 5, 17, 24, 11, 20, 14, 8, 26, 15, 2, 28, 7, 23, 18, 3, 21, 12, 9, 25, 13, 1, 22, 10, 4, 28, 16, 6, 19, 29]


def generate_action_space(num_of_actions=ACTION_SPACE_SIZE, include_vertical=INCLUDE_VERTICAL_SPACE, sequence=DEFAULT_SEQUENCE):
    # tupla la forma (NÚMERO ACCIÓN ELEMENTAL, NÚMERO DE PASOS A REPETIR, DESCRIPCIÓN VERBAL)
    action_space : dict[int, Tuple[int, int, str]] = {}
    cont = 0

    if cont < num_of_actions:
        action_space[cont] = (LEFT, WALKING_LENGTH, "LEFT")
        cont += 1
    if cont < num_of_actions:
        action_space[cont] = (RIGHT, WALKING_LENGTH, "RIGHT")
        cont += 1
    if include_vertical and cont < num_of_actions:
        action_space[cont] = (SPACE, JUMPCOUNT_MAX, "VERTICAL_SPACE_FOR_" + str(JUMPCOUNT_MAX) + "_STEPS")
        cont += 1

    turn = "LEFT"
    steps = 0
    while cont < num_of_actions:

        if steps >= 30:
            raise ValueError("Too many actions in action space!")

        if turn == "LEFT":
            action_space[cont] = (SPACE_LEFT, sequence[steps], "LEFT_SPACE_FOR_" + str(sequence[steps]) + "_STEPS")
            turn = "RIGHT"

        else:
            action_space[cont] = (SPACE_RIGHT, sequence[steps], "RIGHT_SPACE_FOR_" + str(sequence[steps]) + "_STEPS")
            turn = "LEFT"
            steps += 1

        cont += 1
    
    return action_space