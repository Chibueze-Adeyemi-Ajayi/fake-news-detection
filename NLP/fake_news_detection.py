import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

class FakeNewsDectector:
    
    def __init__(self, path_dict):
        self.paths = path_dict
        
    def loadNews (self, bool):
        if bool:
            return pd.read_csv(self.paths["fake"])
        else:
            return pd.read_csv(self.paths["true"])
        
    def mergeData (self, category):
        df = pd.concat(category, ignore_index=True)
        return df.sample(frac=1, random_state=42)
            
    def MLCategorizer (self, dataset):
        return train_test_split(dataset["text"], dataset["type"], test_size=.25)
    
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