from source.RandomForest import *
import json

if __name__ == "__main__":

    with open("trainSmall.json") as f:
        train = json.load(f)
    with open("testSmall.json") as y:
        test = json.load(y)

    testForest = RandomForest(5, train)

    total = 0
    correct = 0
    for recipe in test:
        total += 1
        result = testForest.test_point(recipe)
        if result == recipe['cuisine']:
            correct += 1

    print("Percentage correct = " + str((correct / total) * 100) + "%")