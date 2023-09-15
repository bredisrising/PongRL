import pygame
from game import Game, Paddle, Ball
from a2c import A2C
from vpg import VPG

from constants import *



if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    


    left_paddle = Paddle("left", (75, 75, 255), ai=VPG("left", BATCHES, EPISODE_LENGTH, load=True))
    right_paddle = Paddle("right", (255, 75, 75), ai=VPG("right", BATCHES, EPISODE_LENGTH, load=True))
    ball = Ball((255, 255, 255))
    ball.reset()
    
    game = Game(screen, left_paddle, right_paddle, ball, load=True)    
    
    running = True
    fps = 30


    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE or event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and ALLOW_INPUT:
                if event.key == pygame.K_w:
                    left_paddle.act(0)
                    right_paddle.act(0)
                elif event.key == pygame.K_s:
                    left_paddle.act(1)
                    right_paddle.act(1) 
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    if fps == 15:
                        fps = 30
                    else:
                        fps = 2000
                elif event.key == pygame.K_DOWN:
                    if fps == 2000:
                        fps = 30
                    else:
                        fps = 15

        game.step()

        pygame.display.flip()

        clock.tick(fps)


    left_paddle.ai.save()
    right_paddle.ai.save()

    game.save("vpg")
    