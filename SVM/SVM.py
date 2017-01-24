import numpy as np
from scipy import stats as st
from scipy import optimize as op
import pandas as pd
from sys import path
from matplotlib import pyplot as plt
path.append('../')

n = 50
u1 = [3,3]
std1 = [[2,0], [0,30]]
u2 = [15,5]
std2 = [[3,-12],[3,2]]
train1 = np.array([list(np.random.multivariate_normal(u1, std1)) + [-1] for i in range(n//2)])
train2 = np.array([list(np.random.multivariate_normal(u2, std2)) + [1] for i in range(n//2)])
train = np.concatenate((train1, train2))


cs = pd.Series(train[:,-1]).map({-1 : 'red', 1 : 'blue'})
x_data = train[:,0]
y_data = train[:,1]
plt.scatter(x_data, y_data, c=cs)
plt.axis('equal')
plt.show()
#class SVM(Classifier):
    
def get_equation(a):
    X = train[:,:-1]
    y = train[:,-1]
    H = get_H(X, y)
    return (sum(a)- 1/2*np.matmul(np.matmul(a.T, H), a), min(a) > 0, sum(np.inner(a, y)))
    

def get_H(X, y):
    n_samples, n_features = X.shape
    K = np.zeros((n_samples, n_samples))
    # TODO(tulloch) - vectorize
    for i, x_i in enumerate(X):
        for j, x_j in enumerate(X):
            K[i, j] = np.dot(x_i, x_j)
    return K
