import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
from pathlib import Path

from Agents.PPOAgent import PPOAgent

import Train
from Agents.RandomAgent import RandomAgent
from Agents.LeftRightAgent import LeftRightAgent
from Agents.ListAgent import ListAgent
from Agents.DDQNAgent import DDQNAgent
from Agents.LoadSaveAgent import LoadSaveAgent
from ActionSpace import generate_action_space
from JumpKing import JKGame
from Constants import *

if __name__ == "__main__":
	#Game = JKGame()
	#Game.running()

	train = True

	# Si se esta probando PPO = True
	# SI se esta probando DDQN = False
	PPO = True 

	if (PPO):
		"""PPO"""
		torch.device("cuda" if torch.cuda.is_available() else "cpu")

		""" t = Train.Train(ListAgent(), csv_savepath="test.csv")
		t.run() """
		state_dim = 4
		action_space = generate_action_space(num_of_actions=10)
		print(action_space)

		agent = PPOAgent(state_dim, len(action_space))

		trainer = Train.Train(agent,
							action_space=action_space,
							agent_savepath="model_ppo_episode.pth",
							csv_savepath="ppo_training.csv")
		trainer.run()
		agent.plot()

	else:

		path = "model_ddqn_episode"
		ddqn_state_dimension = 3
		action_space = generate_action_space(num_of_actions=10)

		if (train):
			t = Train.Train(DDQNAgent(	state_dim=ddqn_state_dimension,
						    			action_dim=len(action_space),
										is_training=train),
									action_space=action_space,
									csv_savepath= path + ".csv",
									agent_savepath= path + ".pth",
									)
		else:
			t = Train.Train(DDQNAgent(	state_dim=ddqn_state_dimension,
						    			action_dim=len(action_space),
										is_training=train),
									action_space=action_space,
									csv_savepath= path + ".csv",
									agent_loadpath= path + ".pth"
									)

		t.run()


	
