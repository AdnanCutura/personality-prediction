import pickle
import numpy as np
import h5py
import pandas as pd

data = h5py.File('data/data-normalized_prediction.h5', 'r')
X = pd.DataFrame(np.array(data.get('X')))
data.close()

with open('./model.sav', 'rb') as file:
    model = pickle.load(file)
  
prediction = model.predict(X)

print(np.argmax(prediction,axis=1))