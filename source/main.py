from source.decisionTree import *
from source.RandomForest import *
import json
import timeit
import numpy as np
import sys


def dTreeTest(trainFile, testFile):

    with open(trainFile) as f:
        train = json.load(f)

    # need to use when doing full dataset or else will hit max recursion depth
    # doesnt cause crash on my machine but may on yours, care
    limit = 5
    sys.setrecursionlimit(10 ** limit)

    # can mess with the filter, is the percentage of the lowest occuring ingredients that will be removed
    # seems to be at about 75% we start to lose accuracy with medium set
    # 80 seems to be the cutoff for the large dataset but can go to 90 and only lose about a percent
    filter = 90
    filterIngredients(train, filter)
    print(filter)
    # removes the recipes with now empty ingredient lists bc they don't help but may negativly effect acc
    train = removeEmptyRecipes(train)
    # put all the ingredent lists in sets, speeds up everything alot
    putIngredientsInSets(train)

    root1 = decisionTreeNode()

    starttime = timeit.default_timer()
    root1.makeDecisionTree(train)
    print("The time difference is :", timeit.default_timer() - starttime)

    with open(testFile) as f:
        test = json.load(f)

    total = 0
    correct = 0
    for recipe in test:
        total += 1
        result = root1.test_point(recipe)
        if result == recipe['cuisine']:
            correct += 1
    print("Percentage correct = " + str((correct / total) * 100) + "%")


def forestTest(trainFile, testFile, forestSize):
    with open(trainFile) as f:
        train = json.load(f)
    with open(testFile) as y:
        test = json.load(y)

    # need to use when doing full dataset or else will hit max recursion depth
    # doesnt cause crash on my machine but may on yours, care
    limit = 5
    sys.setrecursionlimit(10**limit)

    # can mess with the filter, is the percentage of the lowest occuring ingredients that will be removed
    # seems to be at about 75% we start to lose accuracy with medium set
    # 80 seems to be the cutoff for the large dataset but can go to 90 and only lose about a percent
    filter = 90
    filterIngredients(train, filter)
    print(filter)
    # removes the recipes with now empty ingredient lists bc they don't help but may negativly effect acc
    train = removeEmptyRecipes(train)
    # put all the ingredent lists in sets, speeds up everything alot
    putIngredientsInSets(train)

    starttime = timeit.default_timer()

    testForest = RandomForest(forestSize, train)
    print("The time difference is :", timeit.default_timer() - starttime)

    total = 0
    correct = 0
    for recipe in test:
        total += 1
        result = testForest.test_point(recipe)
        if result == recipe['cuisine']:
            correct += 1

    print("Percentage correct = " + str((correct / total) * 100) + "%")


if __name__ == '__main__':

    # change these to the right strings for your system
    trainSmall = 'trainSmall.json'
    testSmall = 'testSmall.json'
    trainMedium = 'trainMedium.json'
    testMedium = 'testMedium.json'
    trainLarge = 'trainLarge.json'
    testLarge = 'testLarge.json'

    print("\n---- Welcome to the Recipe Classifier ----\nBy: Mary Ann Hazuga, Heoliny Jung, Joe Soellner\n")
    isForest = input("Would you like to test a single decision tree or a random forest?\n(tree = 'T', forest = 'F')\n")
    if isForest == 'F':
        forest_size = int(input("Please enter the size of your forest:\n"))
    selection = input("Please select a dataset size:\nSmall: 'S'\nMedium: 'M'\nLarge: 'L'\n")
    if selection == 'S':
        if isForest == 'F':
            forestTest(trainSmall, testSmall, forest_size)
        else:
            dTreeTest(trainSmall, testSmall)
    if selection == 'M':
        if isForest == 'F':
            forestTest(trainMedium, testMedium, forest_size)
        else:
            dTreeTest(trainMedium, testMedium)
    if selection == 'L':
        if isForest == 'F':
            forestTest(trainLarge, testLarge, forest_size)
        else:
            dTreeTest(trainLarge, testLarge)
