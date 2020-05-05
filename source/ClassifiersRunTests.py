import Classifiers as cl
import json
import numpy as np
import random as rn
import time

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
    trials = 5000
    X, y = cl.makeData(train)
    index = rn.sample(range(0,39739), trials)
    smallX = np.empty((trials, len(X[0])))
    smally = np.empty(trials,  dtype = '|S30')
    count = 0
    for i in index:
        smallX[count] = X[i]
        smally[count] = y[i]
        count = count + 1
    start = time.time()
    cl.SVM(smallX, smally)
    print((time.time() - start)/5)
    start = time.time()
    cl.NearestNeighbor(smallX, smally)
    print((time.time() - start)/5)
    start = time.time()
    cl.MLP(smallX, smally)
    print((time.time() - start)/5)

quickDemo()


#For 1000 samples: SVM: 46.9% time = 0.56 seconds, NN: 30.7% time = 0.09 seconds, MLP: 46% time = 1.11 seconds
#For 2000 samples: SVM: 51.55% time = 1.87 seconds, NN: 35.15% time = 0.27 seconds, MLP: 52.25% time = 3.06 seconds
#For 4000 samples: SVM: 53.7% time = 6.43 seconds, NN: 38.85% time = 1.22 seconds, MLP: 54.275% time = 6.45 seconds 
#For 5000 samples: SVM: 56.04% time = 9.33 seconds, NN: 43.74% time = 2.09 seconds, MLP: 55.3% time = 7.01 seconds