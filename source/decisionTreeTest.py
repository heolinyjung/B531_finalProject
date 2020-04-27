import math
import json
import decisionTree
import timeit
import numpy as np

if __name__ == '__main__':
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

    assert(decisionTree.calculateEntropy(testEntropyThreeClasses) == 1.5709505944546684), decisionTree.calculateEntropy(testEntropyThreeClasses)
    assert(decisionTree.calculateEntropy(testEntropyAllSame) == 0), decisionTree.calculateEntropy(testEntropyAllSame)
    assert(decisionTree.calculateEntropy(testEntropyMaxEntropy) == 1), decisionTree.calculateEntropy(testEntropyMaxEntropy)
    assert(decisionTree.calculateEntropy(testEntropyMaxEntropyThreeClasses) == 1.584962500721156), decisionTree.calculateEntropy(testEntropyMaxEntropyThreeClasses)
    # ---------------------------------------------- Entropy Tests -----------------------------------------------------

    # ---------------------------------------------- Info Gain Tests -----------------------------------------------------
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

    assert(decisionTree.calculateInformationGain(testInfoGain, "eggs") == 0.9709505944546685), decisionTree.calculateInformationGain(testInfoGain, "eggs")
    print(decisionTree.calculateInformationGain(testInfoGain2, "olive oil"))
    print(decisionTree.calculateInformationGain(testInfoGain2, "eggs"))
    # ---------------------------------------------- Info Gain Tests -----------------------------------------------------

    # ---------------------------------------------- Decision Tree Train Tests -----------------------------------------------------
    with open('testDTree.json') as f:
        train = json.load(f)

    """
    filter = 5
    decisionTree.filterIngredients(train, filter)
    ingredientCounts = decisionTree.getIngredientCounts(train)
    dick = dict()
    for x in ingredientCounts:
        if ingredientCounts.get(x) < filter:
            dick[x] = ingredientCounts.get(x)

    ingredientAmountsBiggerThanFilter = 0
    for ingredient in ingredientCounts:
        if ingredientCounts.get(ingredient) < filter:
            ingredientAmountsBiggerThanFilter += 1

    print("# of unique ingredients with amounts more than or equal to", filter, ":", ingredientAmountsBiggerThanFilter)
    """
    decisionTree.filterIngredients(train, 5)
    decisionTree.putIngredientsInSets(train)
    root = decisionTree.dTreeNode()
    
    # convert to numpy arrays instead of python lists maybe?
    starttime = timeit.default_timer()
    print("The start time is :", starttime)
    root.decisionTree(train)
    print("The time difference is :", timeit.default_timer() - starttime)

    # ---------------------------------------------- Decision Tree Train Tests -----------------------------------------------------

    # ---------------------------------------------- Decision Tree Test Tests -----------------------------------------------------

    testTree = decisionTree.dTreeNode()
    testTree.decisionTree(testInfoGain)
    newPoint = {
        "id": 3123,
        "cuisine": "greek",
        "ingredients": [
            "olive oil",
            "eggs"
        ]}
    print(testTree.test_point(newPoint))

    # ---------------------------------------------- Decision Tree Test Tests -----------------------------------------------------
