import pickle
import os

from tensorflow.keras.models import load_model

class Model:
    def __init__(self):
        scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pickle')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        enc_path = os.path.join(os.path.dirname(__file__), 'encoder.h5')
        self.encoder = load_model(enc_path, compile=False)

        nn_path = os.path.join(os.path.dirname(__file__), 'nearest.pickle')
        with open(nn_path, 'rb') as f:
            self.model = pickle.load(f)   

    def predict(self, x):
        x_process = self.scaler.transform(x)
        x_process = self.encoder.predict(x_process)
        scores, results = self.model.kneighbors(x_process)
        
        return scores, results
