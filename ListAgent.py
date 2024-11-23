from Train import Agent, State
from typing import List, Tuple

RIGHT : int = 0
LEFT : int = 1
SPACE_RIGHT : int = 2
SPACE_LEFT : int = 3
SPACE : int = 4
IDLE : int = 5

ACTIONS_LIST : List[Tuple[int, int]] = [
    (SPACE_RIGHT, 30), (RIGHT, 1),
    (RIGHT, 20),
    (SPACE_LEFT, 30), (LEFT, 1),
    (LEFT, 40),
    (SPACE_RIGHT, 30), (RIGHT, 1),
]

class ListAgent(Agent):

    def start_episode(self):
        self.actions : List[Tuple[int, int]] = []
        actions_list_copy = ACTIONS_LIST.copy()

        while len(actions_list_copy) != 0:
            action, repeat = actions_list_copy.pop()
            for i in range(repeat):
                self.actions.append(action)
    
    def select_action(self, coded_state):

        if len(self.actions) != 0:
            return self.actions.pop()
        else:
            return IDLE
