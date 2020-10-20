## The complete model for Songster

### Files:

#### Classes

*BasicModel.py*: builds/trains/predicts on a plain scikit-learn Nearest Neighbors 
model.

*FullModel*: loads and predicts with a pretrained full pipeline:

input --> scaler --> encoder --> nearest neighbors --> results

*TrainableModel*: to rebuild the full pipeline.

#### Other files

*scaler.pickle*: Pickled, trained scikit-learn StandardScaler

*encoder.h5*: Trained TensorFlow/Keras encoder

*nearest.pickle*: Pickled, trained scikit-learn NearestNeighbors model


### Useage:

#### Make predictions with pipeline
```
from model.FullModel import Model

model = Model()
scores, indices = model.predict(data)
```

#### Rebuild pipeline
```
from model.TrainableModel import Encoder, Model

### If number of features changed ###
encoder = Encoder()

# build a new one
encoder.build(data_dim=*<number of features>*)
encoder.fit(traning_data)

# save it
encoder.save()

### Train new scaler and nearest model
model = Model()

# fit it
history = model.fit(training_data)

# You can plot the history.history['loss'] for example if you like

# make a prediction
# example needs be 2-dimensional array

x = x.reshape(-1, 1)
scores, indices = model.predict(x)

# save the new model (replaces previous, irrereversible)
model.save()
```

