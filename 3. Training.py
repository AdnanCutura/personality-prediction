import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import pickle

data = h5py.File('data/data-normalized.h5', 'r')
X = pd.DataFrame(np.array(data.get('X')))
y = pd.DataFrame(np.array(data.get('y')))
data.close()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=7)

# Logistic Regression
logreg = LogisticRegression(max_iter=3000, C=0.5, n_jobs=-1)
logreg.fit(X_train, y_train.values.ravel())

Y_pred = logreg.predict(X_test)
predictions = [round(value) for value in Y_pred]

# Evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Confusion matrix

score = logreg.score(X_test, y_test)
cm = metrics.confusion_matrix(y_test, predictions)

plt.figure(figsize=(15,15))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);

pickle.dump(logreg, open('model.sav', 'wb'))