import pygame
import torch
import sys
from vpg import VPG
from ppo import PPO
from dqn import DQN
from a2c import A2C
from run_methods import *
from constants import *

string_to_algo = {
    "vpg": VPG,
    "ppo": PPO,
    "dqn": DQN,
    "a2c": A2C
}

if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    if sys.argv[1] == 'train':
        train(string_to_algo[sys.argv[2]], screen, clock)
    elif sys.argv[1] == 'train_all':
        train_all(screen, clock)
    elif sys.argv[1] == 'play_vs':
        computer_vs_human(string_to_algo[sys.argv[2]], screen, clock)
    elif sys.argv[1] == 'watch':
        computer_vs_computer(string_to_algo[sys.argv[2]], string_to_algo[sys.argv[3]], screen, clock)