import pygame
from game import Game, Paddle, Ball
from a2c import A2C
from vpg import VPG
from ppo import PPO
from dqn import DQN

from constants import *



if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    
    left_paddles = [
        Paddle("left", (75, 75, 255), ai=VPG("left", BATCHES, EPISODE_LENGTH, load=False)),
        Paddle("left", (75, 75, 255), ai=A2C("left", BATCHES, EPISODE_LENGTH, load=False)),
        Paddle("left", (75, 75, 255), ai=PPO("left", BATCHES, EPISODE_LENGTH, load=False)),
        Paddle("left", (75, 75, 255), ai=DQN("left", BATCHES, EPISODE_LENGTH, load=False))
    ]

    right_paddles = [
        Paddle("right", (255, 75, 75), ai=VPG("right", BATCHES, EPISODE_LENGTH, load=False)),
        Paddle("right", (255, 75, 75), ai=A2C("right", BATCHES, EPISODE_LENGTH, load=False)),
        Paddle("right", (255, 75, 75), ai=PPO("right", BATCHES, EPISODE_LENGTH, load=False)),
        Paddle("right", (255, 75, 75), ai=DQN("right", BATCHES, EPISODE_LENGTH, load=False))
    ]

    # left_paddle = Paddle("left", (75, 75, 255), ai=DQN("left", BATCHES, EPISODE_LENGTH, load=False))
    # right_paddle = Paddle("right", (255, 75, 75), ai=DQN("right", BATCHES, EPISODE_LENGTH, load=False))
    
    
    for i in range(4):
        ball = Ball((255, 255, 255))
        ball.reset()
        game = Game(screen, left_paddles[i], right_paddles[i], ball, load=False)    
        
        running = True
        fps = 3000

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE or event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False

            if game.step():
                running = False
            

            pygame.display.flip()

            clock.tick(fps)


        left_paddles[i].ai.save()
        right_paddles[i].ai.save()

        game.save(left_paddles[i].ai.net)
    