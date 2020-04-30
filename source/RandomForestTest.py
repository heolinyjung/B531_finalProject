from RandomForest import *
import decisionTree
import json
import timeit

if __name__ == "__main__":

    with open("source/trainSmall.json") as f:
        train = json.load(f)
    with open("source/testSmall.json") as y:
        test = json.load(y)

    # decisionTree.filterIngredients(train, 5)
    # decisionTree.putIngredientsInSets(train)

    starttime = timeit.default_timer()
    print("The start time is :", starttime)
    testForest = RandomForest(15, train)
    print("The time difference is :", timeit.default_timer() - starttime)

    total = 0
    correct = 0
    for recipe in test:
        total += 1
        result = testForest.test_point(recipe)
        if result == recipe['cuisine']:
            correct += 1

    print("Percentage correct = " + str((correct / total) * 100) + "%")