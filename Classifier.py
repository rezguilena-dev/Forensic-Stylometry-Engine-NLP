import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class Classifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.model = MultinomialNB(alpha=0.1)

    def train(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        print("Modèle entraîné avec succès.")
        
    def predict(self, text):
        X_vect = self.vectorizer.transform([text])
        return self.model.predict(X_vect)[0]