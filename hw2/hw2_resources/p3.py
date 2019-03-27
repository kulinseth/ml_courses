import numpy as np
from plotBoundary import *
import pylab as pl
# import your LR training code



# Carry out training.
### TODO %%%
train = np.loadtxt('data/data3_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

wt = np.zeros(X.shape[1])
b = np.zeros(1)

# Define the predict_linearSVM(x) function, which uses global trained parameters, w
def predict_linearSVM(x):
    if (np.dot(wt, x + b) < 1) :
        return -1;
    else:
        return 1;

def pegasos(X, y, lambd, max_epochs) :
    t = 0
    epoch = 1
    wt = np.zeros(X.shape[1])
    while (epoch < max_epochs) :
        for i in range(X.shape[0]):
            t = t+1
            eta = 1/(t*lambd)
            if (y[i]*(np.dot(wt.T, X[i]))) < 1:
                wt = (1 - eta*lambd)*wt + eta*y[i]*X[i]
            else:
                wt = (1 - eta*lambd)*wt
        epoch = epoch + 1
    return wt

if __name__ == '__main__':
# load data from csv files
    W = pegasos(X, Y, 0.5, 10)
    print W
    #plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear SVM')
    #pl.show()

