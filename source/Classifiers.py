import json
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier





#Takes in a dictionary from our dataset and turns it into testable features and targets
#Returns features, targets
def makeData(train):
    cuisineCounts = dict()
    ingredientCounts = dict()

    for item in train:
        ingredientList = item.get("ingredients")
        for ingredient in ingredientList:
            if ingredient in ingredientCounts:
                ingredientCounts[ingredient] += 1
            else:
                ingredientCounts[ingredient] = 1

     # makes a dictionary of all the cuisine type and how many times they appear in the json object
    for item in train:
        cuisine = item.get("cuisine")
        if cuisine in cuisineCounts:
            cuisineCounts[cuisine] += 1
        else:
            cuisineCounts[cuisine] = 1

    ingredientTotal = 0
    for item in ingredientCounts.values():
        ingredientTotal += item

    #Get features to use for classification
    features = []
    for ingredient, amount in ingredientCounts.items():
        if amount > 500 and amount < 10000:
            features.append(ingredient)


    #Build the training features and targets
    y = np.empty(len(train), dtype = '|S30')
    X = np.zeros((len(train), len(features)))

    cuisineList = list(cuisineCounts.keys())

    for i in range(len(train)):
        ingredients = train[i].get("ingredients")
        cuisine = train[i].get("cuisine")
        y[i] = cuisine
        for x in ingredients:
            if x in features:
                X[i][features.index(x)] = 1  

    return X, y


def MLP(X, y):
    classifier1 = MLPClassifier(hidden_layer_sizes = (10,))

    folding = KFold(n_splits=5)

    for train, test in folding.split(X):
        classifier1.fit(X[train], y[train])
        print("MLP Prediction accuracy", classifier1.score(X[test], y[test]))

def SVM(X, y):
    classifier2 = SVC(kernel = 'rbf')

    folding = KFold(n_splits=5)

    for train, test in folding.split(X):
        classifier2.fit(X[train], y[train])
        print("SVM Prediction accuracy", classifier2.score(X[test], y[test]))

def NearestNeighbor(X, y):
    classifier1 = KNeighborsClassifier(n_neighbors = 25, weights = 'distance', algorithm = 'kd_tree')

    folding = KFold(n_splits=5)

    for train, test in folding.split(X):
        classifier1.fit(X[train], y[train])
        print("Nearest Neighbors Prediction accuracy", classifier1.score(X[test], y[test]))




