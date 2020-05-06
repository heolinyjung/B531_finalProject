from RandomForest import *
from decisionTree import *
import decisionTree
import json
import timeit
import sys
import main
import pickle

if __name__ == "__main__":
    # opens and tests pickled model
    """
    with open('source/75Filter30TreeForest.pickle', "rb") as f:
       forest1 = pickle.load(f)
    with open('source/75Filter30TreeForestTest.pickle', "rb") as f:
        testForest1 = pickle.load(f)

    total = 0
    correct = 0
    for recipe in testForest1:
        total += 1
        result = forest1.test_point(recipe)
        if result == recipe['cuisine']:
            correct += 1
    print("Random forest percentage correct = " + str((correct / total) * 100) + "%")

    testRecipe = {"ingredients" : ["flour", "eggs", "milk", "salt"]}
    print(forest1.test_point(testRecipe))
    
    """

    with open("source/train.json") as f:
        data = json.load(f)
    
    train, testForest1 = main.foldData(data, 31818, 7955)

    # need to use when doing full dataset or else will hit max recursion depth
    # doesnt cause crash on my machine but may on yours, care
    limit = 7
    sys.setrecursionlimit(10**limit)

    # can mess with the filter, is the percentage of the lowest occuring ingredients that will be removed
    # seems to be at about 75% we start to lose accuracy with medium set
    # 80 seems to be the cutoff for the large dataset but can go to 90 and only lose about a percent
    filter = 75
    filterIngredients(train, filter)
    print("filter:", filter)
    # removes the recipes with now empty ingredient lists bc they don't help but may negativly effect acc
    train = removeEmptyRecipes(train)
    # put all the ingredent lists in sets, speeds up everything alot
    putIngredientsInSets(train)

    # makes a forest with so many trees
    trees = 30
    print("trees:", trees)
    starttime = timeit.default_timer()
    forest1 = RandomForest(trees, train)
    print("The time difference is :", timeit.default_timer() - starttime)

    # tests forest
    total = 0
    correct = 0
    for recipe in testForest1:
        total += 1
        result = forest1.test_point(recipe)
        if result == recipe['cuisine']:
            correct += 1
    print("Random forest percentage correct = " + str((correct / total) * 100) + "%")

    # pickles model for later use
    """
    with open('source/TreeForest.pickle', 'wb') as f:
        pickle.dump(forest1, f)
    with open('source/ForestTest.pickle', 'wb') as f:
        pickle.dump(testForest1, f)
    """
    