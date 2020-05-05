import math
import json
from decisionTree import *
import timeit
import numpy as np
import sys
import main
import pickle

if __name__ == '__main__':
    # ---------------------------------------------- Decision Tree Tests -----------------------------------------------------
    # loads pickled models 
    """
    with open('tree.pickle', "rb") as f:
        root1 = pickle.load(f)
    with open('testTree.pickle', "rb") as f:
        test = pickle.load(f)
    """
    
    with open('source/train.json') as f:
        data = json.load(f)

    # need to use when doing full dataset or else will hit max recursion depth
    # doesnt cause crash on my machine but may on yours, care
    limit = 5
    sys.setrecursionlimit(10**limit)

    # can mess with the filter, is the percentage of the lowest occuring ingredients that will be removed
    # seems to be at about 75% we start to lose accuracy with medium set
    # 80 seems to be the cutoff for the large dataset but can go to 90 and only lose about a percent
    train, test = main.foldData(data, 31818, 7955)
    # removes the recipes with now empty ingredient lists bc they don't help but may negativly effect acc
    filter = 95
    filterIngredients(train, filter)
    # filterIngredients(train2, filter)
    train = removeEmptyRecipes(train)
    # train2 = removeEmptyRecipes(train2)
    # put all the ingredent lists in sets, speeds up everything alot
    putIngredientsInSets(train)
    # putIngredientsInSets(train2)

    root1 = decisionTreeNode()

    # makes the tree and times it
    starttime = timeit.default_timer()
    root1.makeDecisionTree(train)
    print("The time difference is :", timeit.default_timer() - starttime)

    # tests trained model
    total = 0
    correct = 0
    for recipe in test:
        total += 1
        result = root1.test_point(recipe)
        if result == recipe['cuisine']:
            correct += 1
    print("Decision tree percentage correct = " + str((correct/total) * 100) + "%")

    # pickles models for later use
    """
    with open('tree.pickle', 'wb') as f:
        pickle.dump(root1, f)
    with open('testTree.pickle', 'wb') as f:
        pickle.dump(test, f)
    """
    
    # ---------------------------------------------- Decision Tree Tests -----------------------------------------------------
    """
    # ---------------------------------------------- Entropy Tests -----------------------------------------------------
    testEntropyAllSame = [
        {
        "id": 10259,
        "cuisine": "italian",
        "ingredients": [
            "romaine lettuce"
        ]
        },
        {
        "id": 25693,
        "cuisine": "italian",
        "ingredients": [
            "plain flour"
        ]
        },
        {
        "id": 20130,
        "cuisine": "italian",
        "ingredients": [
            "eggs"
        ]
        }
    ]
    testEntropyMaxEntropy = [
        {
        "id": 10259,
        "cuisine": "italian",
        "ingredients": [
            "romaine lettuce"
        ]
        },
        {
        "id": 25693,
        "cuisine": "italian",
        "ingredients": [
            "plain flour"
        ]
        },
        {
        "id": 20130,
        "cuisine": "greek",
        "ingredients": [
            "eggs"
        ]
        },
        {
        "id": 201,
        "cuisine": "greek",
        "ingredients": [
            "eggs"
        ]
        }
    ]
    testEntropyMaxEntropyThreeClasses = [
        {
        "id": 10259,
        "cuisine": "italian",
        "ingredients": [
            "romaine lettuce"
        ]
        },
        {
        "id": 25693,
        "cuisine": "italian",
        "ingredients": [
            "plain flour"
        ]
        },
        {
        "id": 20130,
        "cuisine": "greek",
        "ingredients": [
            "eggs"
        ]
        },
        {
        "id": 201,
        "cuisine": "greek",
        "ingredients": [
            "eggs"
        ]
        },
        {
        "id": 20130,
        "cuisine": "french",
        "ingredients": [
            "eggs"
        ]
        },
        {
        "id": 201,
        "cuisine": "french",
        "ingredients": [
            "eggs"
        ]
        }
    ]
    testEntropyThreeClasses = [
        {
        "id": 10259,
        "cuisine": "italian",
        "ingredients": [
            "romaine lettuce"
        ]
        },
        {
        "id": 25693,
        "cuisine": "italian",
        "ingredients": [
            "plain flour"
        ]
        },
        {
        "id": 20130,
        "cuisine": "italian",
        "ingredients": [
            "eggs"
        ]
        },
        {
        "id": 22213,
        "cuisine": "french",
        "ingredients": [
            "water"
        ]
        },
        {
        "id": 13162,
        "cuisine": "french",
        "ingredients": [
            "black pepper"
        ]
        },
        {
        "id": 6602,
        "cuisine": "french",
        "ingredients": [
            "plain flour"
        ]
        },
        {
        "id": 42779,
        "cuisine": "greek",
        "ingredients": [
            "olive oil"
        ]
        },
        {
        "id": 427,
        "cuisine": "greek",
        "ingredients": [
        "olive oil"
        ]
        },
        {
        "id": 4277,
        "cuisine": "greek",
        "ingredients": [
        "olive oil"
        ]
        },
        {
        "id": 3124,
        "cuisine": "greek",
        "ingredients": [
        "olive oil"
        ]
        }
    ]

    assert(calculateEntropy(testEntropyThreeClasses) == 1.5709505944546684), calculateEntropy(testEntropyThreeClasses)
    assert(calculateEntropy(testEntropyAllSame) == 0), calculateEntropy(testEntropyAllSame)
    assert(calculateEntropy(testEntropyMaxEntropy) == 1), calculateEntropy(testEntropyMaxEntropy)
    assert(calculateEntropy(testEntropyMaxEntropyThreeClasses) == 1.584962500721156), calculateEntropy(testEntropyMaxEntropyThreeClasses)
    """
    # ---------------------------------------------- Entropy Tests -----------------------------------------------------

    # ---------------------------------------------- Info Gain Tests -----------------------------------------------------
    """
    testInfoGain = [
        {
        "id": 10259,
        "cuisine": "italian",
        "ingredients": [
            "plain flour",
            "lettuce",
            "eggs",
            "salt"
        ]
        },
        {
        "id": 25693,
        "cuisine": "italian",
        "ingredients": [
            "plain flour",
            "lettuce",
            "eggs"
        ]
        },
        {
        "id": 20130,
        "cuisine": "italian",
        "ingredients": [
            "plain flour",
            "lettuce",
            "salt"
        ]
        },
        {
        "id": 22213,
        "cuisine": "french",
        "ingredients": [
            "flour",
            "lettuce",
            "eggs"
        ]
        },
        {
        "id": 13162,
        "cuisine": "french",
        "ingredients": [
            "eggs",
            "flour",
            "lettuce"
        ]
        },
        {
        "id": 6602,
        "cuisine": "french",
        "ingredients": [
            "eggs"
        ]
        },
        {
        "id": 42779,
        "cuisine": "greek",
        "ingredients": [
            "eggs",
            "flour",
            "lettuce"
        ]
        },
        {
        "id": 427,
        "cuisine": "greek",
        "ingredients": [
            "olive oil",
            "flour"
        ]
        },
        {
        "id": 4277,
        "cuisine": "greek",
        "ingredients": [
            "olive oil",
            "lettuce"
        ]
        },
        {
        "id": 3124,
        "cuisine": "greek",
        "ingredients": [
            "olive oil"
        ]
        }
    ]
    testInfoGain2 = [
        {
        "id": 22213,
        "cuisine": "french",
        "ingredients": [
            "flour",
            "lettuce",
            "eggs"
        ]
        },
        {
        "id": 13162,
        "cuisine": "french",
        "ingredients": [
            "eggs",
            "flour",
            "lettuce"
        ]
        },
        {
        "id": 6602,
        "cuisine": "french",
        "ingredients": [
            "eggs"
        ]
        },
        {
        "id": 42779,
        "cuisine": "greek",
        "ingredients": [
            "eggs",
            "flour",
            "lettuce"
        ]
        },
        {
        "id": 427,
        "cuisine": "greek",
        "ingredients": [
            "olive oil",
            "flour"
        ]
        },
        {
        "id": 4277,
        "cuisine": "greek",
        "ingredients": [
            "olive oil",
            "lettuce"
        ]
        },
        {
        "id": 3124,
        "cuisine": "greek",
        "ingredients": [
            "olive oil"
        ]
        }
    ]

    testIt = getUniqueIngredients(testInfoGain)
    counts = getIngredientCounts(testInfoGain)

    for ing in testIt:
        print("bad:", calculateInformationGainBad(testInfoGain, ing), "good", calculateInformationGain(testInfoGain, ing), "ing:", ing, "count:", counts.get(ing))
    

    # assert(calculateInformationGain(testInfoGain, "eggs") == 0.9709505944546685), calculateInformationGain(testInfoGain, "eggs")
    assert(calculateInformationGain(testInfoGain2, "eggs") == calculateInformationGain(testInfoGain2, "olive oil"))
    """
    # ---------------------------------------------- Info Gain Tests -----------------------------------------------------
    
    # ---------------------------------------------- Decision Tree Train Tests -----------------------------------------------------
    """
    with open('source/trainSmall.json') as f:
        train = json.load(f)

    # testing the test_point() function
    testTree = decisionTree.decisionTreeNode()
    testTree.makeDecisionTree(testInfoGain)
    newPoint = {
        "id": 3123,
        "cuisine": "greek",
        "ingredients": [
            "olive oil",
            "eggs"
        ]}
    print(testTree.test_point(newPoint))

    # testing the tree with full datasets
    with open('source/test.json') as f:
        test = json.load(f)

    total = 0
    correct = 0
    for recipe in testInfoGain:
        total += 1
        result = root.test_point(recipe)
        if result == recipe['cuisine']:
            correct += 1

    print("Percentage correct = " + str((correct/total) * 100) + "%")
    """