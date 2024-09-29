from .base import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SupervisedLearningModel(BaseModel):
    def __init__(self, model, model_name="Supervised Model"):
        super().__init__(model_name)
        self.model = model

    def train(self, data, labels):
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
        self.model.fit(X_train, y_train)
        self.loss_history.append(1 - accuracy_score(y_test, self.model.predict(X_test)))
        self.X_test, self.y_test = X_test, y_test

    def predict(self, input_data):
        return self.model.predict(input_data)

    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, predictions)
