import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
from pathlib import Path

from Agents.PPOAgent import PPOAgent
from DDQN import DDQN

import Train
from Agents.RandomAgent import RandomAgent
from Agents.LeftRightAgent import LeftRightAgent
from Agents.ListAgent import ListAgent
from JumpKing import JKGame

'''
¡Obsoleto!
Esta función corresponde el entrenamiento del repositorio original.
Ya no esta siendo utilizada. Aun así creo que es bueno dejarla por ahora.
En cambio en Train.py hay una clase Train que pretende generalizar a un Agente con un método cualquiera y añadir funcionalidades.
'''
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

if __name__ == "__main__":
	#Game = JKGame()
	#Game.running()

	#train()
	torch.device("cuda" if torch.cuda.is_available() else "cpu")

	""" t = Train.Train(ListAgent(), csv_savepath="test.csv")
	t.run() """
	state_dim = 4
	action_dim = 4
	agent = PPOAgent(state_dim, action_dim)
	trainer = Train.Train(agent, csv_savepath="ppo_training.csv")
	trainer.run()