from .base import BaseModel
import numpy as np

class ReinforcementLearningModel(BaseModel):
    def __init__(self, action_space, state_space, model_name="Reinforcement Model"):
        super().__init__(model_name)
        self.action_space = action_space
        self.state_space = state_space
        self.q_table = np.zeros((state_space, action_space))

    def train(self, episodes, alpha=0.1, gamma=0.6, epsilon=0.1):
        for episode in range(episodes):
            state = np.random.randint(0, self.state_space)
            done = False
            total_loss = 0

            while not done:
                if np.random.uniform(0, 1) < epsilon:
                    action = np.random.randint(0, self.action_space)
                else:
                    action = np.argmax(self.q_table[state])

                new_state, reward, done = self.take_action(state, action)
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[new_state])
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                loss = abs(new_value - old_value)
                total_loss += loss
                self.q_table[state, action] = new_value
                state = new_state

            self.loss_history.append(total_loss)

    def take_action(self, state, action):
        new_state = np.random.randint(0, self.state_space)
        reward = np.random.uniform(-1, 1)
        done = np.random.choice([True, False])
        return new_state, reward, done

    def predict(self, state):
        return np.argmax(self.q_table[state])
