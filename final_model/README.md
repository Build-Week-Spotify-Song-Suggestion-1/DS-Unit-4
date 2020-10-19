## Predictive Model for Songster
### Final Model version 1.0

This is a ready to go, fully trained model to predict similar items to a
single example. To use:

### Required imports

```
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '..')  # this may be needed if called from a sibling directory, 
                          # otherwise omit.
from final_model.FullModel import Model
```

### Instantiate the model

```
model = Model()
```

### Predict on an item from the dataset

You will need to pick an item from the dataset, and transform that row into a
2 dimensional array. The model is trained on 11 features:

```
features = ['acousticness', 'danceability','energy',
            'instrumentalness', 'key', 'liveness', 'loudness',
            'mode','speechiness', 'tempo',
            'valence']
```

So, for example, with an example chosen from the dataset:

```
test = np.array(df.iloc[20215][features])
test = test.reshape(1, -1)

scores, indices = model.predict(test)
```

And that is about it. Scores are the distances between the items (as vectors in
an 11th dimensional space), and indices are the indexes of the selected items in the
original dataset.

See also [Test model.ipynb](Test_model.ipynb) for reference.

