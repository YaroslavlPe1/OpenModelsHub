import pickle

class BaseModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.loss_history = []

    def train(self, data, labels=None):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def predict(self, input_data):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def evaluate(self, test_data, test_labels=None):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        with open(filename, 'rb') as file:
            self.model = pickle.load(file)
        print(f"Model loaded from {filename}")

    def print_loss_history(self):
        for epoch, loss in enumerate(self.loss_history):
            print(f"Epoch {epoch+1}: Loss = {loss}")
