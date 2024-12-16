import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import argparse

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

def print_instructions():
    print("Uso del script:")
    print("  --DDQN --train   : Entrena el modelo DDQN")
    print("  --PPO --train    : Entrena el modelo PPO")
    print("  --DDQN           : Evalúa el modelo DDQN (sin entrenar)")
    print("  --PPO            : Evalúa el modelo PPO (sin entrenar)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Selecciona el modo de ejecución del script.")
    parser.add_argument('--DDQN', action='store_true', help="Utiliza el modelo DDQN.")
    parser.add_argument('--PPO', action='store_true', help="Utiliza el modelo PPO.")
    parser.add_argument('--train', action='store_true', help="Entrena el modelo seleccionado.")

    args = parser.parse_args()

    if not (args.DDQN or args.PPO):
        print("[Error] Debes especificar un modelo (--DDQN o --PPO).")
        print_instructions()
        exit(1)

    if args.DDQN and args.PPO:
        print("[Error] No puedes seleccionar ambos modelos (--DDQN y --PPO) simultáneamente.")
        print_instructions()
        exit(1)

    train_mode = args.train

    if args.PPO:
        print(f"Modo seleccionado: {'Entrenamiento' if train_mode else 'Evaluación'} con PPO")
        path = "ModelsPPO/model_ppo_episode"
        load_path = "ModelsPPO/PPO_trained"

        # Configuración del dispositivo
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Definición del agente PPO
        state_dim = 5
        action_space = generate_action_space(num_of_actions=10)
        agent = PPOAgent(state_dim, len(action_space))

        trainer = Train.Train(
        agent,
        action_space=action_space,
        agent_savepath=path if train_mode else None,
        agent_loadpath=None if train_mode else load_path,
        csv_savepath=path
        )

        trainer.run()

    elif args.DDQN:
        print(f"Modo seleccionado: {'Entrenamiento' if train_mode else 'Evaluación'} con DDQN")
        action_space = generate_action_space(12)
        state_dimension = 3
        path = "ModelsDDQN/model_ddqn_episode"

        load_path = "ModelsDDQN/DDQN_trained"

        if train_mode:
            agent = DDQNAgent(
                state_size=state_dimension,
                action_size=len(action_space)
            )

            trainer = Train.Train(
                agent,
                action_space=action_space,
                agent_savepath=path,
                csv_savepath=path
            )
        else:
            agent = DDQNAgent(
                state_size=state_dimension,
                action_size=len(action_space),
                lr=0,
                epsilon_start=0,
                epsilon_decay=1,
                epsilon_end=0
            )

            trainer = Train.Train(
                agent,
                action_space=action_space,
                agent_loadpath=load_path,
                csv_savepath=path
            )

        trainer.run()
