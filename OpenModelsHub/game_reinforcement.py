import numpy as np
import random
from .base import BaseModel


class GameReinforcementLearningModel(BaseModel):
    def __init__(self, action_space, state_space, model_name="Game Reinforcement Model"):
        super().__init__(model_name)
        self.action_space = action_space
        self.state_space = state_space
        self.q_table = np.zeros((state_space, action_space))

    def train(self, environment, episodes, alpha=0.1, gamma=0.6, epsilon=0.1):
        """
        Метод для обучения агента в игровом окружении.

        environment: Объект игры, который должен иметь методы reset(), step() и state()
        episodes: Количество эпизодов для обучения
        alpha: Скорость обучения
        gamma: Дисконтирующий коэффициент (вес будущих вознаграждений)
        epsilon: Параметр для epsilon-жадной стратегии (исследование или использование знаний)
        """
        for episode in range(episodes):
            state = environment.reset()
            done = False
            total_loss = 0

            while not done:
                # Epsilon-жадная стратегия для выбора действия
                if random.uniform(0, 1) < epsilon:
                    action = random.randint(0, self.action_space - 1)
                else:
                    action = np.argmax(self.q_table[state])

                # Выполнение действия в окружении
                new_state, reward, done = environment.step(action)

                # Обновление Q-таблицы
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[new_state])

                # Обновленное значение Q
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                loss = abs(new_value - old_value)
                total_loss += loss

                # Запись нового значения
                self.q_table[state, action] = new_value

                # Переход к новому состоянию
                state = new_state

            self.loss_history.append(total_loss)
            print(f"Episode {episode + 1}/{episodes} - Total Loss: {total_loss}")

    def play(self, environment, episodes):
        """
        Метод для тестирования агента в игровом окружении.

        environment: Объект игры, который должен иметь методы reset(), step() и state()
        episodes: Количество эпизодов для тестирования
        """
        for episode in range(episodes):
            state = environment.reset()
            done = False
            total_reward = 0

            while not done:
                # Выбор действия на основе Q-таблицы
                action = np.argmax(self.q_table[state])

                # Выполнение действия в игре
                new_state, reward, done = environment.step(action)
                total_reward += reward
                state = new_state

            print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward}")

    def save_model(self, filename):
        np.save(filename, self.q_table)
        print(f"Q-table saved to {filename}.npy")

    def load_model(self, filename):
        self.q_table = np.load(filename)
        print(f"Q-table loaded from {filename}.npy")
