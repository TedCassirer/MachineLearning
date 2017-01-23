import pandas as pd
import numpy as np
from scipy import stats as st

class Classifier:
    def load_data(self, ratio = 0.5):
        df = pd.read_csv(self.data_path)
        train_size = int(df.shape[0] * ratio)
        self.train = df.sample(train_size)
        self.test = df.drop(self.train.index)

    def score(self):
        pred = self.predict(self.test.iloc[:, :-1], self.labels)
        actual = self.test.label
        accuracy = (pred == actual).sum()/len(actual) * 100
        print(('Accuracy: {0}%').format(round(accuracy, 2)))
        
