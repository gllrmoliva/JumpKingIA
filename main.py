import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

	train = False 

	# Si se esta probando PPO = True
	# SI se esta probando DDQN = False
	PPO = False 

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
		action_space = generate_action_space(12)
		state_dimension = 3
		path = "Model/model_ddqn_episode"

		
		if (train):
			t = Train.Train(DDQNAgent(state_size=state_dimension,
						    	action_size=len(action_space)),
						action_space=action_space,
						csv_savepath= path,
						agent_savepath= path)
		else:
			load_path = "Model/model_ddqn_episode"
			t = Train.Train(DDQNAgent(state_size=state_dimension,
						    	action_size=len(action_space),
                            	lr=0,
                            	epsilon_start=0,
                            	epsilon_decay=1,
                            	epsilon_end=0),
						action_space=action_space,
						agent_loadpath= load_path,
						csv_savepath= path,
						)

	t.run()


	
