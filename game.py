import torch
import pygame
import random
import numpy as np
from constants import *
import pickle

class Ball:
    def __init__(self, color):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.speed = 25
        self.color = color

        self.vx = 0
        self.vy = 0

        self.direction = -1

    def init_vel(self, consistent=False):
        if consistent:
            self.vx = pygame.mouse.get_pos()[0] - WIDTH // 2
            self.vy = pygame.mouse.get_pos()[1] - HEIGHT // 2
        else:
            self.vx = random.uniform(0.6, 1.0) * self.direction
            self.vy = random.uniform(0.1, 0.9) * (random.choice([1, -1]))
        
        mag = np.sqrt(self.vx**2 + self.vy**2)
        self.vx /= mag
        self.vy /= mag
        
        self.vx *= self.speed
        self.vy *= self.speed  

    def reset(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.direction *= -1
        self.init_vel(False)

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
        self.speed = 40
        self.color = color

        self.score = 0

        self.ai = ai

    def render(self, screen):
        pygame.draw.rect(screen, self.color, pygame.Rect(self.x, self.y, PADDLE_WIDTH, PADDLE_SIZE))


    def update(self, ball_state, p=False):
        if self.ai == None:
            #print("NO AI")
            return
        state = torch.cat((ball_state, torch.tensor([self.y/PADDLE_RANGE], dtype=torch.float32)))
        action, output = self.ai.sample_action(state)
        if p == True:
            pass
            #print(output, end="\r")
            #print(self.ai.value(state), "   ", output, end="\r")

        self.act(action)


    def act(self, action):
        if action == 0:
            self.y -= self.speed
        elif action == 1:
            self.y += self.speed
        elif action == 2:
            # do nothing action - technically elif statement not need
            pass

        if self.y <= 0:
            self.y = 0
        elif self.y >= PADDLE_RANGE:
            self.y = PADDLE_RANGE



class Game:
    def __init__(self, screen, left_paddle, right_paddle, ball, load=False, train=False):
        self.screen = screen
        self.left_paddle = left_paddle
        self.right_paddle = right_paddle
        self.ball = ball
        
        self.consecutive_hit_counter = 0
        self.max_consecutive_hits = 15
        
        self.train = False

        if self.train:
            self.max_consecutive_hits = 5
        
        if load:
            self.reward_per_match = pickle.load(open("./train_logs/vpg_rewards.pkl", "rb"))
        else:
            self.reward_per_match = []
        self.rolling_average = 0

        self.font = pygame.font.Font('freesansbold.ttf', 24)
    
    def save(self, name):
        # pickle the reward
        with open("./train_logs/"+name+"_rewards.pkl", "wb") as f:
            pickle.dump(self.reward_per_match, f)

    def handle_collisions(self):
        # check collisions
        self.ball.update()

        # if self.ball.x - BALL_RADIUS > 0:
        if self.left_paddle.y - BALL_RADIUS<= self.ball.y <= self.left_paddle.y + PADDLE_SIZE + BALL_RADIUS and self.ball.x <= self.left_paddle.x + PADDLE_WIDTH + BALL_RADIUS + 3:
            self.ball.x = self.left_paddle.x + PADDLE_WIDTH + BALL_RADIUS + 1
            self.ball.vx *= -1
            
            self.left_paddle.ai.reward(1)
            self.consecutive_hit_counter += 1
        
        # if self.ball.x < WIDTH - BALL_RADIUS:
        if self.right_paddle.y - BALL_RADIUS <= self.ball.y <= self.right_paddle.y + PADDLE_SIZE + BALL_RADIUS and self.ball.x >= self.right_paddle.x - BALL_RADIUS - 3:
            self.ball.x = self.right_paddle.x - BALL_RADIUS - 1
            self.ball.vx *= -1

            self.right_paddle.ai.reward(1)
            self.consecutive_hit_counter += 1

        if self.ball.x - BALL_RADIUS <= 0:
            # right paddle wins
            self.right_paddle.score += 1
            distance = -abs(self.ball.y - (self.left_paddle.y + PADDLE_SIZE // 2)) / PADDLE_RANGE * 2
            self.left_paddle.ai.reward(distance)
            self.consecutive_hit_counter = 0
            self.ball.reset()
        
        elif self.ball.x + BALL_RADIUS >= WIDTH:
            # left paddle wins
            self.left_paddle.score += 1
            distance = -abs(self.ball.y - (self.right_paddle.y + PADDLE_SIZE // 2)) / PADDLE_RANGE * 2
            self.right_paddle.ai.reward(distance)
            self.consecutive_hit_counter = 0
            self.ball.reset()

        

    def step(self):   
        if self.consecutive_hit_counter >= self.max_consecutive_hits:
            self.left_paddle.score += 1
            self.right_paddle.score += 1  
            self.ball.reset() 
            self.consecutive_hit_counter = 0


        self.handle_collisions()

        ball_state = self.ball.normed_state(1)
        self.left_paddle.update(ball_state, True)
        self.right_paddle.update(ball_state)

        #print(self.left_paddle.ai.update_counter, end="\r")

        # rendering
        self.screen.fill((0, 0, 0))

        self.left_paddle.render(self.screen)
        self.right_paddle.render(self.screen)
        self.ball.render(self.screen)

        if not self.train:
            return False

        if LOG and self.left_paddle.score > 10 or self.right_paddle.score > 10:
            # game over
            # log accumulated reward
            self.reward_per_match.append(self.left_paddle.ai.accumulated_reward)

            #rolling average
            self.rolling_average = sum(self.reward_per_match[-50:]) / 50

            self.left_paddle.ai.accumulated_reward = 0
            self.left_paddle.score = 0
            self.right_paddle.score = 0


        text = self.font.render(str(self.rolling_average), True, (255, 255, 255), (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (WIDTH // 2, 50)
        self.screen.blit(text, textRect)
        
        if len(self.reward_per_match) <= 0:
            return False
        
        print(self.reward_per_match, '                 ', end="\r")

        text = self.font.render(str(self.reward_per_match[-1]), True, (255, 255, 255), (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (WIDTH // 2, 150)
        self.screen.blit(text, textRect)


        if len(self.reward_per_match) == MATCHES:
            return True
        else:
            return False

        
