import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class GameReinforcementLearningModel:
    def __init__(self, action_space, state_space, model_name="Game RL Model"):
        self.action_space = action_space
        self.state_space = state_space
        self.q_table = np.zeros((state_space, action_space))  # Для Q-Learning и SARSA
        self.model_name = model_name

    # Q-Learning (табличный)
    def q_learning(self, environment, episodes, alpha=0.1, gamma=0.6, epsilon=0.1):
        for episode in range(episodes):
            state = environment.reset()
            done = False
            total_loss = 0
            while not done:
                # Epsilon-жадная стратегия
                if random.uniform(0, 1) < epsilon:
                    action = random.randint(0, self.action_space - 1)
                else:
                    action = np.argmax(self.q_table[state])
                new_state, reward, done = environment.step(action)

                # Q-Learning обновление
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[new_state])
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                self.q_table[state, action] = new_value
                state = new_state

    # SARSA
    def sarsa(self, environment, episodes, alpha=0.1, gamma=0.6, epsilon=0.1):
        for episode in range(episodes):
            state = environment.reset()
            done = False
            action = self.epsilon_greedy_action(state, epsilon)
            while not done:
                new_state, reward, done = environment.step(action)
                new_action = self.epsilon_greedy_action(new_state, epsilon)

                # SARSA обновление
                old_value = self.q_table[state, action]
                next_value = self.q_table[new_state, new_action]
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_value)
                self.q_table[state, action] = new_value
                state, action = new_state, new_action

    def epsilon_greedy_action(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            return np.argmax(self.q_table[state])

    # Deep Q-Learning (DQN)
    class DQN(nn.Module):
        def __init__(self, state_space, action_space):
            super(GameReinforcementLearningModel.DQN, self).__init__()
            self.fc1 = nn.Linear(state_space, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, action_space)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    def dqn(self, environment, episodes, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=32):
        model = self.DQN(self.state_space, self.action_space)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        memory = deque(maxlen=2000)
        criterion = nn.MSELoss()
        target_model = self.DQN(self.state_space, self.action_space)

        for episode in range(episodes):
            state = environment.reset()
            done = False
            total_reward = 0
            state = torch.FloatTensor(state).unsqueeze(0)

            while not done:
                if np.random.rand() <= epsilon:
                    action = random.randrange(self.action_space)
                else:
                    with torch.no_grad():
                        action = torch.argmax(model(state)).item()

                next_state, reward, done = environment.step(action)
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                memory.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                if len(memory) > batch_size:
                    self.replay(memory, batch_size, model, target_model, optimizer, criterion, gamma)

            epsilon = max(epsilon_min, epsilon_decay * epsilon)

            if episode % 10 == 0:
                target_model.load_state_dict(model.state_dict())
            print(f"Episode {episode} - Total Reward: {total_reward}")

    def replay(self, memory, batch_size, model, target_model, optimizer, criterion, gamma):
        minibatch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            q_update = reward
            if not done:
                q_update = (reward + gamma * torch.max(target_model(next_state)).item())
            q_values = model(state)
            q_values[0][action] = q_update
            optimizer.zero_grad()
            loss = criterion(q_values, model(state))
            loss.backward()
            optimizer.step()

    # PPO (Proximal Policy Optimization)
    # Актуален для более сложных окружений, таких как игры с непрерывными состояниями.
    # Реализация может быть добавлена в зависимости от выбранных требований.

    # A2C/A3C (Advantage Actor-Critic)
    # Аналогично PPO, требует настройки нейронных сетей для обучения.
