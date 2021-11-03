import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import h5py

df_preprocessed = pd.read_csv("data/data-preprocessed.csv")

# Vectorizing - featurization
posts = df_preprocessed['posts'].values.astype('U')
types = df_preprocessed['type']

vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(posts)
feature_names = vectorizer.get_feature_names_out()

X_vectorized = vectorizer.transform(posts)
X = pd.DataFrame(X_vectorized.todense().tolist(), columns=feature_names)

enc = LabelEncoder()
Y = enc.fit_transform(types)
Y = pd.DataFrame(Y)

hf = h5py.File('data/data-normalized.h5', 'w')
hf.create_dataset('X', data=X)
hf.create_dataset('y', data=Y)
hf.close()