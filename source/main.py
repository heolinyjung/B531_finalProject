from source.decisionTree import *
from source.RandomForest import *
import json
import timeit
import numpy as np
import sys
import random
from source.datasetStats import getStats


def foldData(data, numTrain, numTest):
    if numTrain + numTest < len(data):
        total = random.sample(data, numTrain + numTest)
        train = total[:numTrain]
        test = total[numTrain:]
        result = (train, test)
        return result
    else:
        # if this triggers, the desired size of the training + testing sets is larger than the size of the original dataset
        raise ArithmeticError


def dTreeTest(train, test):

    root1 = decisionTreeNode()

    starttime = timeit.default_timer()
    root1.makeDecisionTree(train)
    duration = timeit.default_timer() - starttime
    print("Test duration:", duration)

    total = 0.0
    correct = 0.0
    for recipe in test:
        total += 1
        result = root1.test_point(recipe)
        if result == recipe['cuisine']:
            correct += 1
    percentage = (correct / total) * 100
    print("Percentage correct = " + str(percentage) + "%")
    results = (duration, percentage)
    return results


def forestTest(train, test, forestSize):

    starttime = timeit.default_timer()

    testForest = RandomForest(forestSize, train)
    duration = timeit.default_timer() - starttime
    print("Test duration:", duration)

    total = 0.0
    correct = 0.0
    for recipe in test:
        total += 1
        result = testForest.test_point(recipe)
        if result == recipe['cuisine']:
            correct += 1
    percentage = (correct / total) * 100
    print("Percentage correct = " + str(percentage) + "%")
    results = (duration, percentage)
    return results


def filterData(train):
    # need to use when doing full dataset or else will hit max recursion depth
    # doesnt cause crash on my machine but may on yours, care
    limit = 5
    sys.setrecursionlimit(10 ** limit)

    # can mess with the filter, is the percentage of the lowest occuring ingredients that will be removed
    # seems to be at about 75% we start to lose accuracy with medium set
    # 80 seems to be the cutoff for the large dataset but can go to 90 and only lose about a percent
    filter = 80
    filterIngredients(train, filter)
    print("Filter percentage: " + str(filter))
    # removes the recipes with now empty ingredient lists bc they don't help but may negativly effect acc
    return removeEmptyRecipes(train)


def testWithoutFilter(train, test):
    # ------test without filter------
    print("Unfiltered training data stats:")
    train1 = train.copy()
    getStats(train1, test)
    print()
    # put all the ingredent lists in sets, speeds up everything alot
    putIngredientsInSets(train1)
    print("Unfiltered training data decision tree test...")
    dTreeDur, dTreePer = dTreeTest(train1, test)
    print()
    print("Unfiltered training data random forest test...")
    forestDur, forestPer = forestTest(train1, test, 15)
    print()
    results = (dTreeDur, dTreePer, forestDur, forestPer)
    return results


def testWithFilter(train, test, numTrees):
    # ------test with filter------
    train = filterData(train)
    print()
    print("Filtered training data stats:")
    getStats(train, test)
    print()
    # put all the ingredent lists in sets, speeds up everything alot
    putIngredientsInSets(train)
    print("Filtered training data decision tree test...")
    dTreeDur, dTreePer = dTreeTest(train, test)
    print()
    print("Filtered training data random forest test...")
    forestDur, forestPer = forestTest(train, test, numTrees)
    print()
    results = (dTreeDur, dTreePer, forestDur, forestPer)
    return results


def findAverages(data, forestSize, trials, train_size, test_size):
    avgTreeDur = 0.0
    avgTreePer = 0.0
    avgForestDur = 0.0
    avgForestPer = 0.0
    maxTreePer = 0.0
    minTreePer = 100.0
    maxForestPer = 0.0
    minForestPer = 100.0

    for i in range(trials):
        print("-------------Trial "+str(i+1)+"-------------")
        train, test = foldData(data, train_size, test_size)

        # results = testWithoutFilter(train, test)
        # uncomment above and comment below to test without filter
        results = testWithFilter(train, test, forestSize)
        avgTreeDur += results[0]
        avgTreePer += results[1]
        avgForestDur += results[2]
        avgForestPer += results[3]

        maxTreePer = max(maxTreePer, results[1])
        minTreePer = min(minTreePer, results[1])
        maxForestPer = max(maxForestPer, results[3])
        minForestPer = min(minForestPer, results[3])

    avgTreeDur /= trials
    avgTreePer /= trials
    avgForestDur /= trials
    avgForestPer /= trials

    print("Trials:", trials)
    print("Average decision tree duration:", avgTreeDur)
    print("Average decision tree percent correct: " + str(avgTreePer) + "%")
    print("Decision tree deviation:", maxTreePer - minTreePer)
    print("Forest size:", forestSize)
    print("Average random forest duration:", avgForestDur)
    print("Average random forest percent correct: " + str(avgForestPer) + "%")
    print("Random forest deviation:", maxForestPer - minForestPer)


if __name__ == '__main__':

    with open('train.json') as f:
        data = json.load(f)

    # desired forest size
    forestSize = 3

    # ------full test------

    train_size = len(data) - int(len(data)/5)
    test_size = int(len(data)/5) - 1
    train, test = foldData(data, train_size, test_size)
    testWithFilter(train, test, forestSize)

    # ---------------------

    # desired train size
    train_size = 3600
    # desired test size
    test_size = 800

    # ------same data, with or without filter------

    # train, test = foldData(data, train_size, test_size)
    # testWithoutFilter(train, test, forestSize)
    # testWithFilter(train, test, forestSize)

    # ------averages over multiple different datasets------

    # desired number of trials
    trials = 10

    # findAverages(data, forestSize, trials, train_size, test_size)

    # Seems to me like the filter can hugely affect the single decision tree either in a good way or bad while
    # the random forest stays within +-2% with or without filter, meanwhile their execution times are affected
    # in similar ways

    # random forest outcomes are much too unpredictable at anything less than size 10/15. Deviations are almost always
    # higher than their decision tree counterparts

    # Set feature randomness factor to .5
    # With filter, @ 800/200, 20 trees, 20 trials = 38.525% acc, avg dur = 20.05 (control = 32.075% acc, 1.29)
    # With filter, @ 800/200, 30 trees, 20 trials = 39.975% acc, avg dur = 31.57 (control = 32.625% acc, 1.33)
    # With filter, @ 1600/400, 20 trees, 10 trials = 45.1% acc, avg dur = 69.3, dev = 5.25 (control = 36.95% acc, 4.35)
    # With filter, @ 1600/400, 30 trees, 10 trials = 42.65% acc, avg dur = 103.9, dev = 9.5 (control = 37.05% acc, 4.42)
    # With filter, @ 1600/400, 50 trees, 10 trials = 44.75% acc, avg dur = 174.14, dev = 6.5 (control = 37.125% acc, 4.44)
    # ^^ Only difference between number of trees is deviation (maybe) ^^
    # With filter, @ 3200/800, 20 trees, 10 trials = 42.65% acc, avg dur = 103.9 (control = 37.05% acc, 4.42)




    '''
    # change these to the right strings for your system
    trainSmall = 'trainSmall.json'
    testSmall = 'testSmall.json'
    trainMedium = 'trainMedium.json'
    testMedium = 'testMedium.json'
    trainLarge = 'trainLarge.json'
    testLarge = 'testLarge.json'

    print("\n---- Welcome to the Recipe Classifier ----\nBy: Mary Ann Hazuga, Heoliny Jung, Joe Soellner\n")
    cont = False
    forest_size = 0
    isForest = ''
    while not cont:
        isForest = input("Would you like to test a single decision tree or a random forest?\n(tree = 'T', forest = 'F')\n")
        if isForest == 'F':
            forest_size = int(input("Please enter the size of your forest:\n"))
            cont = True
        elif isForest == 'T':
            cont = True
        else:
            print("**Not a valid input, try again**")

    cont = False
    while not cont:
        selection = input("Please select a dataset size:\nSmall: 'S'\nMedium: 'M'\nLarge: 'L'\n")
        if selection == 'S':
            cont = True
            if isForest == 'F':
                forestTest(trainSmall, testSmall, forest_size)
            else:
                dTreeTest(trainSmall, testSmall)
        elif selection == 'M':
            cont = True
            if isForest == 'F':
                forestTest(trainMedium, testMedium, forest_size)
            else:
                dTreeTest(trainMedium, testMedium)
        elif selection == 'L':
            cont = True
            if isForest == 'F':
                forestTest(trainLarge, testLarge, forest_size)
            else:
                dTreeTest(trainLarge, testLarge)
        else:
            print("**Not a valid input, try again**")
    '''
