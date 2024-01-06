import pandas as pd
import numpy as np
import spacy

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

class FakeNewsDectector:
    
    def __init__(self, path_dict):
        self.nlp = spacy.load("en_core_web_sm")
        self.knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
        self.paths = path_dict
        
    def loadNews (self, bool):
        if bool:
            return pd.read_csv(self.paths["fake"])#[0:5000]
        else:
            return pd.read_csv(self.paths["true"])#[0:5000]
        
    def mergeData (self, category):
        df = pd.concat(category, ignore_index=True)
        return df.sample(frac=1, random_state=42)
            
    def MLCategorizer (self, dataset, x, y):
        return train_test_split(dataset[x], dataset[y], test_size=.25)
    
    def classifier (self, X_train, y_train):
        self.clf = Pipeline([
            ("vectorizer", CountVectorizer()),
            ("np", MultinomialNB())
        ])
        self.clf.fit(X_train, y_train)
        return self.clf
    
    def analizeReport (self, X_test, y_test):
        pre_training = self.clf.predict(X_test)
        return classification_report(y_test, pre_training)
    
    def predictText (self, text):
        return self.clf.predict([text])
    
    def convertTextToVec (self, text):
        doc = self.nlp(text)
        return doc.vector
    
    def knnClassifier (self, X_train, y_train):
        self.knnClf = self.knn.fit(X_train, y_train)
        return self.knnClf
    
    def stackVector (self, vector):
        return np.stack(vector)
    
    def predictTextWithKNN (self, vector):
        return self.knnClf.predict(vector)
    
    def analizeTextWithKNN (self, pred, data):
        return classification_report(data, pred)