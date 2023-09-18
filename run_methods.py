import torch.nn as nn
import pygame
from game import Game, Paddle, Ball
from constants import *
from copy import deepcopy

from vpg import VPG
from a2c import A2C
from ppo import PPO
from dqn import DQN

algolist = [VPG, A2C, PPO, DQN]

neurons = 128
policy = nn.Sequential(
    nn.Linear(5, neurons),
    nn.ReLU(),
    nn.Linear(neurons, neurons),
    nn.ReLU(),
    nn.Linear(neurons, neurons),
    nn.ReLU(),
    nn.Linear(neurons, 3),
    nn.Softmax()
)   

def game_loop(game, clock):
    running = True
    fps = 3000 if game.train else 15
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE or event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

            if event.type == pygame.KEYDOWN:
                if game.train:
                    if event.key == pygame.K_UP:
                        if fps == 15:
                            fps = 30
                        else:
                            fps = 3000
                else:
                    game.right_paddle.act(0)
            elif event.key == pygame.K_DOWN:
                if game.train:
                    if fps == 3000:
                        fps = 30
                    else:
                        fps = 15
                else:
                    game.right_paddle.act(1)

        if game.step():
            running = False
        
        pygame.display.flip()

        clock.tick(fps)

def computer_vs_human(algo, screen, clock):
    ball = Ball((255, 255, 255))
    left_paddle = Paddle("left", (75, 75, 255), ai=algo("left", BATCHES, BATCH_SIZE, EPISODE_LENGTH, load=True))
    right_paddle = Paddle("right", (255, 75, 75), ai=None)
    ball.reset()
    game = Game(screen, left_paddle, right_paddle, ball, load=False, train=False)
    game_loop(game, clock)


def computer_vs_computer(left_algo, right_algo, screen, clock):
    ball = Ball((255, 255, 255))
    left_paddle = Paddle("left", (75, 75, 255), ai=left_algo("left", BATCHES, BATCH_SIZE, EPISODE_LENGTH, load=True))
    right_paddle = Paddle("right", (255, 75, 75), ai=right_algo("right", BATCHES, BATCH_SIZE, EPISODE_LENGTH, load=True))
    ball.reset()
    game = Game(screen, left_paddle, right_paddle, ball, load=False, train=False)
    game_loop(game, clock)


def train(algo, screen, clock):
    ball = Ball((255, 255, 255))
    left_paddle = Paddle("left", (75, 75, 255), ai=algo("left", BATCHES, BATCH_SIZE, EPISODE_LENGTH, load=False, p=deepcopy(policy)))
    right_paddle = Paddle("right", (255, 75, 75), ai=algo("right", BATCHES, BATCH_SIZE, EPISODE_LENGTH, load=False, p=deepcopy(policy)))
    ball.reset()
    game = Game(screen, left_paddle, right_paddle, ball, load=False, train=True)
    game_loop(game, clock)
    left_paddle.ai.save()
    right_paddle.ai.save()
    game.save(left_paddle.ai.net)


def train_all(screen, clock):
    for i in range(4):
        ball = Ball((255, 255, 255))
        left_paddle = Paddle("left", (75, 75, 255), ai=algolist[i]("left", BATCHES, BATCH_SIZE, EPISODE_LENGTH, load=False, p=deepcopy(policy)))
        right_paddle = Paddle("right", (255, 75, 75), ai=algolist[i]("right", BATCHES, BATCH_SIZE, EPISODE_LENGTH, load=False, p=deepcopy(policy)))
        ball.reset()
        game = Game(screen, left_paddle, right_paddle, ball, load=False, train=True)
        game_loop(game, clock)
        left_paddle.ai.save()
        right_paddle.ai.save()
        game.save(left_paddle.ai.net)
    


