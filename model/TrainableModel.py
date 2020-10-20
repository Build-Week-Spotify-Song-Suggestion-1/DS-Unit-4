import pickle
import os

from tensorflow.keras.models import Model as TF_Model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


'''
This is the complete pipeline: 
    input --> scaler --> encoder --> nearest neighbors --> results

This class will load a full version, make predictions, and also build/train
a new version if desired. It will still expect the same basic distribution as
the original track dataset. 

It assumes a pre-trained model, existing files in the same directory:
    scaler.pickle
    encoder.h5
    nearest.pickle

The fit method will build a new model, and the save method will overwrite
the original files with it, except for the encoder. That can be rebuilt using
the Encoder class. The pretrained one should work as long as the data has the
same distribution and number of features. Otherwise, one would rebuild that
first before rebuilding the pipeline.
'''

class Model:
    def __init__(self, save_prompt=True):
        '''
        Constructor. Loads the pretrained models from same directory.

        Param:

            save_prompt - This should only be changed if being used in an 
                          automated script, i.e., there is not a user to be 
                          prompted when saving a new version of the model. 
                          If used interactively it should remain at default 
                          (True).
        '''
        self.save_prompt = save_prompt

        scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pickle')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        enc_path = os.path.join(os.path.dirname(__file__), 'encoder.h5')
        self.encoder = load_model(enc_path, compile=False)

        nn_path = os.path.join(os.path.dirname(__file__), 'nearest.pickle')
        with open(nn_path, 'rb') as f:
            self.model = pickle.load(f)

    def fit(self, x, n_neighbors=10):
        '''
        Trains new instances of the scaler and the nearest
        neighbors layers. Does *not* retrain the encoder layer. Does not
        save the new model, but use the save method to replace the existing
        pretrained layers.

        Params:

            x - the training data 
            n_neighbors - number of similar items to find, default = 10

        Returns:

            the nearest neighbors model.
        '''
        
        # Make new instances of scaler and nearest neighbors
        self.scaler = StandardScaler()
        self.model = NearestNeighbors(n_neighbors=n_neighbors)

        # train them, using the pretrained encoder
        x_process = self.scaler.fit_transform(x)
        x_process = self.encoder.predict(x_process)
        return self.model.fit(x_process)

    def predict(self, x):
        '''
        Predict method.

        Params:

            x - example to be find similars to, 2-d array.

        Returns:

            score - distances bewtween x and the similars
            indices - indexes into the original dataset
        '''
        x_process = self.scaler.transform(x)
        x_process = self.encoder.predict(x_process)
        scores, results = self.model.kneighbors(x_process)
        return scores, results

    def save(self):
        '''
        Save new scaler and nearest neighbors layers. It will replace the
        existing files, so BE SURE YOU WANT THIS. ;)
        '''
        
        if self.save_prompt:
            print('WARNING: this will replace the existing layers, and is irreversible.')
            ans = input('Are you sure? (yes, no): ')
            if ans != 'yes':
                return

        scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pickle')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        nn_path = os.path.join(os.path.dirname(__file__), 'nearest.pickle')
        with open(nn_path, 'wb') as f:
            pickle.dump(self.model, f)

        
'''
Encoder class. This is to be used to rebuild the encoder.

Useage:

    To build a new encoder (which should not be necessary if the data has
    the same distribution and number of features):

        1) Instantiate the class
        2) Build the autoencoder, with the build method
        3) Fit it (fit method)
        4) Save method

    To test the encoder before saving it, one can call it simply to call
    it as, e.g.,

        encoder_object.encoder.predict(train_data)
'''

class Encoder:
    def __init__(self, save_prompt=True):
        '''
        Constructor. 

        Param:

            save_prompt - This should only be changed if being used in an 
                          automated script, i.e., there is not a user to be 
                          prompted when saving a new version of the model. 
                          If used interactively it should remain at default 
                          (True).
        '''
        self.encoder = None
        self.save_prompt = save_prompt

    def build(self, data_dim):
        '''
        Builds the autoencoder.

        Param:

            data_dim - the data shape (i.e., the number of samples) If a
                       dataframe or similar array, df, that would be 
                       df.shape[0].
        '''
        self.encoder = Sequential([Dense(data_dim, name='encode_1', 
                                         activation='tanh', 
                                         input_shape=(data_dim,)),
                                   Dense(data_dim // 1.25, name='encode_2', 
                                        activation='tanh'),
                                   Dense(data_dim // 2, name='encode_3', 
                                        activation='tanh')])
        self.encoder.compile(optimizer='adam', loss='mse')

        self.decoder = Sequential([Dense(data_dim // 2, name='decode_1', 
                                         activation='tanh', 
                                         input_shape=(data_dim // 2,)),
                                   Dense(data_dim // 1.25, name='decode_2', 
                                         activation='tanh'),
                                    Dense(data_dim, name='decode_3')])
        self.decoder.compile(optimizer='adam', loss='mse')

        input_layer = Input(shape=(data_dim,))
        encoder_output = self.encoder(input_layer)
        decoder_output = self.decoder(encoder_output)
        self.autoencoder = TF_Model(input_layer, decoder_output)
        self.autoencoder.compile(optimizer='adam', loss='mse')
    
    def fit(self, x, epochs=200, batch_size=128):
        '''
        Trains the encoder.

        Params:

            x - the training data (dataframe or 2 dimensional array)
            epochs - maximum number of epochs to train. Default 200
                     (It usually stops sooner).
            batch_size: batch size. Default is 128.

        Return:

            history - metrics on the training processes (e.g, loss at
                      each epoch)

        '''
        if self.encoder is None:
            raise Exception('Encoder not built or loaded')

        # pre-process training data by scaling
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        # stop training when losses level out (convergence)
        stop = EarlyStopping(monitor='loss', 
                             patience=5,
                             min_delta=0.0001,
                             restore_best_weights=True)

        # train the model
        history = self.autoencoder.fit(x_scaled, 
                                     x_scaled, 
                                     epochs=epochs, 
                                     batch_size=batch_size,
                                     callbacks=[stop])
        return history

    def save(self):
        '''
        Save the encoder. Really, in the end, that is all we want for use in the full-stack
        model. Saving it will replace any existing encoder, so BE SURE YOU WANT THAT, as
        it is irreversible.
        '''
        if self.save_prompt:
            print('WARNING: this will replace the existing encoder, and is irreversible.')
            ans = input('Are you sure? (yes, no): ')
            if ans != 'yes':
                return

        enc_path = os.path.join(os.path.dirname(__file__), 'encoder.h5')
        self.encoder.load_model(enc_path)
    
