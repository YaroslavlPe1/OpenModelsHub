import pygame
import random
from OpenModels.game_envs import FlappyBirdEnv

def main():
    env = FlappyBirdEnv()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # For demo purposes, we'll just flap every frame
        action = 1  # Flap
        state, reward, done = env.step(action)
        env.render()

        if done:
            env.reset()

    env.close()

if __name__ == "__main__":
    main()
