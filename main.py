import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
from pathlib import Path

from DDQN import DDQN

import Train
from Agents.RandomAgent import RandomAgent
from Agents.LeftRightAgent import LeftRightAgent
from Agents.ListAgent import ListAgent
from Agents.DDQNAgent import DDQNAgent
from Agents.LoadSaveAgent import LoadSaveAgent
from ActionSpace import generate_action_space
from JumpKing import JKGame
from Constants import *

'''
¡Obsoleto!
Esta función corresponde el entrenamiento del repositorio original.
Ya no esta siendo utilizada. Aun así creo que es bueno dejarla por ahora.
En cambio en Train.py hay una clase Train que pretende generalizar a un Agente con un método cualquiera y añadir funcionalidades.

def train():
	# Funcion para entrenar la IA
	action_dict = {
		0: 'right',
		1: 'left',
		2: 'right+space',
		3: 'left+space',
		# 4: 'space',
		# 5: 'idle',
	}
	agent = DDQN()
	env = JKGame(steps_per_episode=1000)
	num_episode = 100000

	for i in range(num_episode):
		done, state = env.reset()

		running_reward = 0
		while not done:
			action = agent.select_action(state)
			#print(action_dict[action])
			next_state, reward, done = env.step(action)

			running_reward += reward
			sign = 1 if done else 0
			agent.train(state, action, reward, next_state, sign)
			state = next_state
		print (f'episode: {i}, reward: {running_reward}')
'''

if __name__ == "__main__":
	#Game = JKGame()
	#Game.running()

	#train()
	path = "model_ddqn_episode"

	action_space = generate_action_space(num_of_actions=12)

	# t = Train.Train(DDQNAgent(	state_dim=5,
	#					    	action_dim=len(action_space),
	#							is_training=True),
	#						action_space=action_space,
	#						csv_savepath= path + ".csv",
	#						agent_savepath= path + ".pth",
	#						#agent_loadpath= path + ".pth"
	#						)


	t = Train.Train(RandomAgent(len(action_space)), action_space=action_space, csv_savepath="test.csv")

	t.run()