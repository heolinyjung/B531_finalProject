import math
import json
from decisionTree import *
import timeit
import numpy as np
import sys

if __name__ == '__main__':
    # ---------------------------------------------- Decision Tree Tests -----------------------------------------------------
    with open('source/trainMedium.json') as f:
        train = json.load(f)

    # can mess with the filter, the higher it is the quicker the trees are to make
    # when lower doesn't really impact accuracy
    filterIngredients(train, 10)
    # put all the ingredent lists in sets, speeds everything alot
    putIngredientsInSets(train)

    root1 = decisionTreeNode()
    root2 = decisionTreeNode()
    root3 = decisionTreeNode()
    
    starttime = timeit.default_timer()
    root1.makeDecisionTree(train)
    print("The time difference is :", timeit.default_timer() - starttime)
    
    starttime = timeit.default_timer()
    root3.makeDecisionTree2(train)
    print("The time difference is :", timeit.default_timer() - starttime)

    starttime = timeit.default_timer()
    root2.makeDecisionTreeWithEntropy(train)
    print("The time difference is :", timeit.default_timer() - starttime)
    
    with open('source/testMedium.json') as f:
        test = json.load(f)
    
    total = 0
    correct = 0
    for recipe in test:
        total += 1
        result = root1.test_point(recipe)
        if result == recipe['cuisine']:
            correct += 1
    print("Percentage correct = " + str((correct/total) * 100) + "%")

    total = 0
    correct = 0
    for recipe in test:
        total += 1
        result = root3.test_point(recipe)
        if result == recipe['cuisine']:
            correct += 1
    print("Percentage correct = " + str((correct/total) * 100) + "%")

    total = 0
    correct = 0
    for recipe in test:
        total += 1
        result = root2.test_point(recipe)
        if result == recipe['cuisine']:
            correct += 1
    print("Percentage correct = " + str((correct/total) * 100) + "%")

    """
    # need to use when doing full dataset or else will hit max recusion depth
    # doesnt cause crash on my machine but may on yours, care
    limit = 8
    sys.setrecursionlimit(10**limit)
    print(limit)
    """

    """
    decisionTree.calculateInformationGainV2(train, "soy sauce")

    recipesWithIngredient = []
    recipesWithoutIngredient = []
    for recipe in train:
        if "soy sauce" in recipe.get("ingredients"):
            recipesWithIngredient.append(recipe)
        else:
            recipesWithoutIngredient.append(recipe)

    decisionTree.calculateInformationGainV2(recipesWithIngredient, "sesame oil")
    decisionTree.calculateInformationGainV2(recipesWithoutIngredient, "jalapeno chilies")
    """

    """
    # filter function testing
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