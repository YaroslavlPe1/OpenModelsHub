from .base import BaseModel
from sklearn.cluster import KMeans

class UnsupervisedLearningModel(BaseModel):
    def __init__(self, model=KMeans(n_clusters=3), model_name="Unsupervised Model"):
        super().__init__(model_name)
        self.model = model

    def train(self, data):
        self.model.fit(data)
        self.loss_history.append(self.model.inertia_)

    def predict(self, input_data):
        return self.model.predict(input_data)
