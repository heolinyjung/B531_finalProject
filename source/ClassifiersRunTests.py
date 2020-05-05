import Classifiers as cl
import json
import numpy as np
import random as rn

with open('train.json') as f:
    train = json.load(f)
        # train is now a dictionary with keys : 'id', 'cuisine', 'ingredients'
with open('test.json') as f2:
    test = json.load(f2)

def runFullTests():
    X, y = cl.makeData(train)

    cl.SVM(X, y)
    cl.NearestNeighbor(X, y)
    cl.MLP(X, y)

def quickDemo():
    X, y = cl.makeData(train)
    index = rn.sample(range(0,39739), 5000)
    smallX = np.empty((5000, len(X[0])))
    smally = np.empty(5000,  dtype = '|S30')
    count = 0
    for i in index:
        smallX[count] = X[i]
        smally[count] = y[i]
        count = count + 1
    cl.SVM(smallX, smally)
    cl.NearestNeighbor(smallX, smally)
    cl.MLP(smallX, smally)

quickDemo()