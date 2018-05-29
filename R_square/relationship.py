import numpy as np
import math
import matplotlib.pylab as plt
from bokeh.charts.builders.scatter_builder import Scatter


def computeCorrelation(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    varX = 0
    varY = 0
    SSR = 0
    for i in range(len(X)):
        diffXXBar = X[i] - xBar
        diffYYbar = Y[i] - yBar

        SSR += diffXXBar * diffYYbar
        varX += diffXXBar ** 2
        varY += diffYYbar ** 2
    SST = math.sqrt(varX * varY)
    return SSR / SST


def ployfit(x, y, degree):
    result = {}
    coffs = np.polyfit(x, y, degree)
    result['polynomial'] = coffs.tolist()
    #     print coffs
    p = np.poly1d(coffs)
    #     print p
    yhat = p(x)
    #     print yhat," ----"
    fig.scatter(x, yhat)
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    result['determination'] = ssreg / sstot
    return result


fig = plt.subplot()
testX = [1, 3, 8, 7, 9]
testY = [10, 12, 24, 21, 34]
r = computeCorrelation(testX, testY)
print('r:', r)
print("r*r:", r * r)
result = ployfit(testX, testY, 1)
print(result)
fig.scatter(testX, testY, color="green")
plt.show()