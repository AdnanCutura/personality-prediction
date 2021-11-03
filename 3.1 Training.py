import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from xgboost import XGBClassifier

data = h5py.File('data/data_normalized.h5', 'r')
X = pd.DataFrame(np.array(data.get('X')))
y = pd.DataFrame(np.array(data.get('y')))
data.close()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=7)

model_xgb=XGBClassifier(max_depth=5,n_estimators=50,learning_rate=0.1)
model_xgb.fit(X_train, y_train.values.ravel())


print('Train classification report \n ',metrics.classification_report(y_train,model_xgb.predict(X_train)))
print('Test classification report \n ',metrics.classification_report(y_test,model_xgb.predict(X_test)))
