import pylab as pl
import matplotlib.pyplot as plt

import numpy as np
'''
'df needs to have two features, and a column named label
'''
def prettyPicture(clf, df):
    df = df.T
    x_min = df.iloc[:, 0].min(); x_max = df.iloc[:, 0].max() 
    y_min = df.iloc[:, 1].min(); y_max = df.iloc[:, 1].max() 
    xd = (x_max - x_min) * 0.2
    x_max += xd
    x_min -= xd
    yd = (y_max - y_min) * 0.2
    y_max += yd
    y_min -= yd
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    h = xd*0.05  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)

    # Plot also the test points
    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]

    plt.scatter(grade_sig, bumpy_sig, color = "b", label="fast")
    plt.scatter(grade_bkg, bumpy_bkg, color = "r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")
    plt.show()
    #plt.savefig("test.png")
