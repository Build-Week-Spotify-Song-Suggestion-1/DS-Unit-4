import pickle

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors


class BasicModel:
    '''
    This is a 'wrapper' for the actual model. It allows for 
    the model to be instantiated, trained, saved, and reloaded.
    '''
    def __init__(self):
        self.model = None

    def build(self, n_suggestions):
        '''
        Build the model.
        Params:
            n_suggestions - the number of similar items to be
            returned.
        '''
        self.model = Model(n_suggestions)

    def save(self, file_path):
        '''
        Save the trained model
        Params:
            file_Path - the path to which file is to be written
        '''
        assert self.model is not None, 'model not built'
        pickle.dump(self.model, file_path)

    def load(self, file_path):
        '''
        Re-load the trained model (instead of building)
        Params:
            file_Path - the path from which to load
        '''
        self.model = pickle.load(file_path)

    def fit(self, X):
        '''
        Fit the model
        Params:
            X -- training data. Expected to be a pandas dataframe.
        '''
        assert self.model is not None, 'model not built'
        return self.model.fit(X)

    def predict(self, x):
        '''
        Predict similar to example
        Params:
            x -- training data. Expected to be a 2-dim array.
        '''
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
        x_process = self.scaler.transform(x)
        scores, results = self.model.kneighbors(x_process)
        
        return scores, results
