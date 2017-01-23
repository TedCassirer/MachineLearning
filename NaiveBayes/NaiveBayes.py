import pandas as pd
from scipy import stats as st
import sys
sys.path.append('../')
sys.path.append('../tools/')
from Classifier import Classifier
from PrettyPlotter import prettyPicture

class NaiveBayes(Classifier):
    def __init__(self, data_path = '../data/pima-indians-diabetes.data'):
        self.data_path = data_path
        
    def fit(self):
        separated = []
        self.mean = self.train.groupby('label').mean()
        self.std = self.train.groupby('label').std()
        self.labels = self.mean.index
        
    def predict(self, df, labels = None):
        p = pd.DataFrame(index = df.index, columns = labels)
        for i in p.columns:
            p[i] = st.norm(self.mean.loc[i], self.std.loc[i]).pdf(df).prod(axis=1)
        # p = p.div(p.sum(axis=1), axis=0) #Normilize
        return p.idxmax(axis=1)

a = NaiveBayes()
a.load_data(0.5)
a.fit()
a.score()
