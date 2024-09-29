import pygame
import random
from abc import ABC, abstractmethod


class GameEnv(ABC):
    def __init__(self, width: int, height: int):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

    @abstractmethod
    def reset(self) -> None:
        """Сброс состояния игры."""
        pass

    @abstractmethod
    def step(self, action: int) -> tuple:
        """Шаг в игре на основе действия."""
        pass

    @abstractmethod
    def render(self) -> None:
        """Отображение состояния игры."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Закрытие игры."""
        pass


class FlappyBirdEnv(GameEnv):
    def __init__(self, width: int = 400, height: int = 600):
        super().__init__(width, height)
        self.reset()

    def reset(self) -> None:
        self.bird_y = self.height // 2
        self.bird_speed = 0
        self.pipes = []
        self.score = 0
        self.game_over = False
        self._generate_pipes()

    def step(self, action: int) -> tuple:
        if action == 1:  # 1 = flap
            self.bird_speed = -10

        # Update bird position
        self.bird_speed += 1
        self.bird_y += self.bird_speed

        # Update pipes
        self.pipes = [(x - 5, y) for x, y in self.pipes]
        if self.pipes and self.pipes[0][0] < -50:
            self.pipes.pop(0)
            self._generate_pipes()
            self.score += 1

        # Collision detection
        for pipe_x, pipe_y in self.pipes:
            if self.bird_y < pipe_y or self.bird_y > pipe_y + 150:
                if pipe_x < 50 and pipe_x > 0:
                    self.game_over = True

        if self.bird_y > self.height or self.bird_y < 0:
            self.game_over = True

        reward = 1 if not self.game_over else -100
        return (self.bird_y, self.bird_speed, self.pipes), reward, self.game_over

    def render(self) -> None:
        self.screen.fill((135, 206, 235))  # Sky blue

        # Draw bird
        pygame.draw.rect(self.screen, (255, 255, 0), (50, self.bird_y, 30, 30))

        # Draw pipes
        for pipe_x, pipe_y in self.pipes:
            pygame.draw.rect(self.screen, (0, 255, 0), (pipe_x, 0, 50, pipe_y))
            pygame.draw.rect(self.screen, (0, 255, 0), (pipe_x, pipe_y + 150, 50, self.height))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self) -> None:
        pygame.quit()

    def _generate_pipes(self) -> None:
        gap = 150
        pipe_height = random.randint(50, self.height - 200)
        self.pipes.append((self.width, pipe_height))
