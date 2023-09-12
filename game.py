import torch
import pygame
import random
import numpy as np
from vpg_value import VPGValue

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

HEADLESS = False

PADDLE_SIZE = 75
PADDLE_WIDTH = 15
PADDLE_X = 25

BALL_RADIUS = 10

PADDLE_RANGE = HEIGHT - 75
BALLX_RANGE = WIDTH - PADDLE_X*2
BALLY_RANGE = HEIGHT 

ALLOW_INPUT = False


EPISODE_LENGTH = 30 * 2.5 #time steps / frames
BATCHES = 1

FONT = pygame.font.SysFont("Arial", 30)

class Ball:
    def __init__(self, color):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.speed = 25
        self.color = color

        self.vx = 0
        self.vy = 0

    def init_vel(self, consistent=False):
        if consistent:
            self.vx = pygame.mouse.get_pos()[0] - WIDTH // 2
            self.vy = pygame.mouse.get_pos()[1] - HEIGHT // 2
        else:
            self.vx = random.uniform(0.5, 1.0) * (random.choice([1, -1]))
            self.vy = random.uniform(0.6, 0.9) * (random.choice([1, -1]))
        
        mag = np.sqrt(self.vx**2 + self.vy**2)
        self.vx /= mag
        self.vy /= mag
        
        self.vx *= self.speed
        self.vy *= self.speed  

    def reset(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.init_vel(True)

    def render(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), BALL_RADIUS)

    def normed_state(self, norm_type):
        if norm_type == 0:
            return torch.tensor([self.x/BALLX_RANGE, self.y/BALLY_RANGE, self.vx/self.speed, self.vy/self.speed], dtype=torch.float32)
        elif norm_type == 1:
            return torch.tensor([self.x/BALLX_RANGE*2-1, self.y/BALLY_RANGE*2-1, self.vx/self.speed, self.vy/self.speed], dtype=torch.float32)

    def update(self):
        self.x += self.vx
        self.y += self.vy

        if self.y <= BALL_RADIUS or self.y >= HEIGHT - BALL_RADIUS:
            self.y = BALL_RADIUS if self.y <= BALL_RADIUS else HEIGHT - BALL_RADIUS
            self.vy *= -1
        
        

class Paddle:
    def __init__(self, side, color, ai=None):
        self.x = PADDLE_X if side == "left" else WIDTH - PADDLE_X - PADDLE_WIDTH

        self.y = HEIGHT // 2 - PADDLE_SIZE // 2
        self.speed = 25//2
        self.color = color

        self.ai = ai

    def render(self, screen):
        pygame.draw.rect(screen, self.color, pygame.Rect(self.x, self.y, PADDLE_WIDTH, PADDLE_SIZE))



    def update(self, ball_state, p=False):
        state = torch.cat((ball_state, torch.tensor([self.y/PADDLE_RANGE*2-1], dtype=torch.float32)))
        if self.ai == None:
            print("NO AI")
            return
        
        action, output = self.ai.sample_action(state)
        if p == True:
            print(self.ai.value(state), "   ", output, end="\r")


        self.act(action)


    def act(self, action):
        if action == 0:
            self.y -= self.speed
        elif action == 1:
            self.y += self.speed
        elif action == 2:
            # do nothing action
            pass

        if self.y <= 0:
            self.y = 0
        elif self.y >= PADDLE_RANGE:
            self.y = PADDLE_RANGE


left_paddle = Paddle("left", (75, 75, 255), ai=VPGValue(BATCHES, EPISODE_LENGTH))
right_paddle = Paddle("right", (255, 75, 75), ai=VPGValue(BATCHES, EPISODE_LENGTH))
ball = Ball((255, 255, 255))
ball.reset()

fps = 30

running = True
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

    ball_state = ball.normed_state(1)
    #do actions
    left_paddle.update(ball_state, True)
    right_paddle.update(ball_state)


    # check collisions
    ball.update()

    if ball.x + BALL_RADIUS < PADDLE_X:
        # right paddle wins

        distance = -abs(ball.y - (left_paddle.y + PADDLE_SIZE // 2)) / PADDLE_RANGE * 2
        left_paddle.ai.reward(distance)
        ball.reset()
    elif ball.x - BALL_RADIUS > WIDTH - PADDLE_X - PADDLE_WIDTH:
        # left paddle wins

        distance = -abs(ball.y - (right_paddle.y + PADDLE_SIZE // 2)) / PADDLE_RANGE * 2
        right_paddle.ai.reward(distance)
        ball.reset()

    if ball.x - BALL_RADIUS > 0:
        if left_paddle.y <= ball.y <= left_paddle.y + PADDLE_SIZE and ball.x <= left_paddle.x + PADDLE_WIDTH + BALL_RADIUS:
            ball.x = left_paddle.x + PADDLE_WIDTH + BALL_RADIUS
            ball.vx *= -1
            
            left_paddle.ai.reward(1)
    
    if ball.x < WIDTH - BALL_RADIUS:
        if right_paddle.y <= ball.y <= right_paddle.y + PADDLE_SIZE and ball.x >= right_paddle.x - BALL_RADIUS:
            ball.x = right_paddle.x - BALL_RADIUS
            ball.vx *= -1

            right_paddle.ai.reward(1)



    # rendering
    screen.fill((0, 0, 0))

    left_paddle.render(screen)
    right_paddle.render(screen)
    ball.render(screen)

    pygame.display.flip()

    clock.tick(fps)



