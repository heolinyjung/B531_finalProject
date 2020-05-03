from RandomForest import *
from decisionTree import *
import decisionTree
import json
import timeit
import sys

if __name__ == "__main__":

    with open("trainLarge.json") as f:
        train = json.load(f)
    with open("testLarge.json") as y:
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
    # change number 10 to other numbers to increase or decrease the size of the forest
    forestSize = 10
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