import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Ucitavanje podataka
df = pd.read_csv("data/products.csv")
df = df[['Product Title', 'Category Label']].dropna()
df['Product Title'] = df['Product Title'].str.lower()

X = df['Product Title']
y = df['Category Label']

# Model
model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=50000)),
    ('clf', LogisticRegression(max_iter=1000))
])

model.fit(X, y)

# Cuvanje modela
with open("models/product_category_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model uspesno treniran i sacuvan.")
