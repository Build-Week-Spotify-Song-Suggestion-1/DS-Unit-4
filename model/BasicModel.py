import pickle

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors


class BasicModel:
    def __init__(self):
        self.model = None

    def build(self, n_suggestions):
        self.model = Model(n_suggestions)

    def save(self, file_path):
        assert self.model is not None, 'model not built'
        pickle.dump(self.model, file_path)

    def load(self, file_path):
        self.model = pickle.load(file_path)

    def fit(self, X):
        assert self.model is not None, 'model not built'
        return self.model.fit(X)

    def predict(self, x):
        assert self.model is not None, 'model not built'
        return self.model.predict(x)


class Model:
    def __init__(self, n_suggestions):
        self.scaler = StandardScaler()
        self.model = NearestNeighbors(n_neighbors=n_suggestions)

    def fit(self, X):
        x_process = self.scaler.fit_transform(X)
        m = self.model.fit(x_process)
        return m

    def predict(self, x):
        x_process = x.reshape(1, -1)
        x_process = self.scaler.transform(x_process)
        scores, results = self.model.kneighbors(x_process)
        
        return scores, results
