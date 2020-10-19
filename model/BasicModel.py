import pickle

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors


class BaseModel:
    '''
    BaseModel: a plain scikit-learn nearest neighbors model.
    Params:
        n_suggestions - number of similar examples to select
        model_path - path for pickled model to reload. Default =
                     None, in which case a new model is instantiated.
    '''
    def __init__(self, n_suggestions, model_path=None):
        self.scaler = StandardScaler()

        if model_path is not None:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = NearestNeighbors(n_neighbors=n_suggestions)
            
    def fit(self, X):
        '''
        Train the model. X is expected to be a pandas dataframe
        or 2-dim array.
        '''
        x_process = self.scaler.fit_transform(X)
        m = self.model.fit(x_process)
        return m

    def predict(self, x):
        '''
        Find similar examples. X is expected to be a 2-dim array.
        '''
        x_process = self.scaler.transform(x)
        scores, results = self.model.kneighbors(x_process)
        return scores, results

    def save(self, model_path):
        '''
        Save the model. 
        '''
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
