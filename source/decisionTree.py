import math
import json
import numpy as np
import random
import re

"""
recipes param is a list of dictionaries
each dict in the list is a recipe with keys id, cuisine, and ingredients (strings)
id is assigned to the recipe id (int)
cuisine is assigned to the cuisine type (string)
ingredients is assigned to the list of ingredients (list of strings)
example structure of each dictionary in the list:
{ "id" : 1234,
  "cuisine" : "mexican",
  "ingredients" : ["apples", "flour", "sugar"]
}
"""

featureRandomnessFactor = .5
# using filtered data
# .7 @ 800/200, 10 trees, 20 trials = 37.15% acc, avg dur = 11.44 (control = 32.1% acc, 1.51)
# .5 @ 800/200, 10 trees, 20 trials = 37.75% acc, avg dur = 11.99 (control = 32.0% acc, 1.55)
# .3 @ 800/200, 10 trees, 20 trials = 37.175% acc, avg dur = 11.66 (control = 33.225% acc, 1.57)
# .7 @ 800/200, 15 trees, 10 trials = 38.5% acc, avg dur = 18.37 (control = 32.0% acc, 1.55)
# .5 @ 800/200, 15 trees, 10 trials = 38.4% acc, avg dur = 16.63, dev = 10 (control = 32.4% acc, 1.46, 10)
# .3 @ 800/200, 15 trees, 10 trials = 39.1% acc, avg dur = 18.26, dev = 12.5 (control = 32.0% acc, 1.55, 16)
# ^^^ VERY INCONSISTENT ^^^, but shows from 10-15 trees, about +7 sec.
# conclusion: featureRandomnessFactor does not have a huge amount of impact to the algorithm, .5 is a valid choice

# puts all the ingredients lists in sets instead of lists
# MAJOR KEY, REDUCES TIMES BY ALOT BC CONSTANT CHECKS IF SOMETHING IN A SET
# O(ingredients)
def putIngredientsInSets(recipes):
    for recipe in recipes:
        ingredientList = recipe.get("ingredients")
        recipe["ingredients"] = set(ingredientList)

    return recipes

# removes a percentage of ingredients based on how many times they occur in the recipes
# i.e. the if the param bottomPercentToRemove is 10 then will remove ten percent of the ingredients with the lowest number
# of occurences, same for param topPercentToRemove but will remove the ingredients with the most occurences
# source for sorting a dictionary - https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
def filterIngredients(recipes, bottomPercentToRemove=-1, topPercentToRemove=-1):
    if topPercentToRemove == -1 and bottomPercentToRemove == -1:
        return recipes
    # idiot proofing, aka making sure i don't break anything when i eventually do something stupid
    elif bottomPercentToRemove >= 100 or topPercentToRemove >= 100 or bottomPercentToRemove + topPercentToRemove >= 100:
        return list()
    else:
        sortedIngredients = sorted(getIngredientCounts(recipes).items(), key=lambda x: x[1])
        setOfIngreToRemove = set() 

        if bottomPercentToRemove > 0:
            numOfBottomIngreToRemove = int(len(sortedIngredients) * (bottomPercentToRemove / 100))
            for i in range(numOfBottomIngreToRemove):
                setOfIngreToRemove.add((sortedIngredients[i])[0])
        
        if topPercentToRemove > 0:
            numOfTopIngreToRemove = int(len(sortedIngredients) * (topPercentToRemove / 100))
            sortedIngredientsLength = len(sortedIngredients) - 1
            for i in range(numOfTopIngreToRemove):
                setOfIngreToRemove.add((sortedIngredients[sortedIngredientsLength - i])[0])

        for recipe in recipes:
            newIngredientList = list()
            ingredientList = recipe.get("ingredients")
            for ingredient in ingredientList:
                if ingredient not in setOfIngreToRemove:
                    newIngredientList.append(ingredient)

            recipe["ingredients"] = newIngredientList

        return recipes

# removes all the recipes that have no ingredients, usually used after filtering ingredients
def removeEmptyRecipes(recipes):
    newRecipes = list()
    for recipe in recipes:
        ingredientList = recipe.get("ingredients")
        if len(ingredientList) > 0:
            newRecipes.append(recipe)

    return newRecipes

# returns a dictionary of cuisine types (strings) mapped to
# the number of times the cuisine appears in the recipes parameter (int)
# e.g. { "mexican" : 23, "chinese" : 56, "british" : 5 }
# O(recipes)
def getCuisineAmounts(recipes):
    cuisineCounts = dict()
    for recipe in recipes:
        cuisine = recipe.get("cuisine")
        if cuisine in cuisineCounts:
            cuisineCounts[cuisine] += 1
        else:
            cuisineCounts[cuisine] = 1
    return cuisineCounts

# returns a set of all the ingredients in recipes (no duplicates)
# O(ingredients)
def getUniqueIngredients(recipes):
    ingredientsSoFar = set()
    for recipe in recipes:
        ingredientList = recipe.get("ingredients")
        for ingredient in ingredientList:
            if ingredient not in ingredientsSoFar:
                ingredientsSoFar.add(ingredient)

    return ingredientsSoFar

# returns a dictionary of each ingredient mapped to the number of times it occurs in the recipes
# e.g. {"ham" : 20, "salt" : 20000}
# O(ingredients)
def getIngredientCounts(recipes):
    ingredientCounts = dict()

    for recipe in recipes:
        ingredientList = recipe.get("ingredients")
        for ingredient in ingredientList:
            if ingredient in ingredientCounts:
                ingredientCounts[ingredient] += 1
            else:
                ingredientCounts[ingredient] = 1

    return ingredientCounts

# returns a dictionary of each cuisine and the number of times that ingredient is in that cuisine type
# key is the cuisine type (string) and the value is the number of times the ingredient is in the cuisine (int)
# ingredients is a string
# O(ingredients)
def ingredientOccurencesPerCuisineType(recipes, ingredient):
    ingredientCounts = dict()
    for recipe in recipes:
        if ingredient in recipe.get("ingredients"):
            cuisine = recipe.get("cuisine")
            if cuisine in ingredientCounts:
                ingredientCounts[cuisine] += 1
            else:
                ingredientCounts[cuisine] = 1

    return ingredientCounts

# returns a dictionary of all unique ingredients mapped to a dictionary of each cuisine type mapped to the
# number of times that ingredients appears in a recipe of that cuisine type
# e.g. {carrots : {mexican : 3, chinese : 1, british : 6} }
# O(ingredients)
def getCuisineOccurenceForAllIngredients(recipes):
    cuisineOccurenceForAllIngredients = dict()
    
    for recipe in recipes:
        cuisineType = recipe.get("cuisine")
        ingredientList = recipe.get("ingredients")
        for ingredient in ingredientList:
            if ingredient in cuisineOccurenceForAllIngredients:
                ingredientsCuisineCounts = cuisineOccurenceForAllIngredients[ingredient]
                if cuisineType in ingredientsCuisineCounts:
                    ingredientsCuisineCounts[cuisineType] += 1
                else:
                    ingredientsCuisineCounts[cuisineType] = 1
            else:
                cuisineOccurenceForAllIngredients[ingredient] = dict()
                (cuisineOccurenceForAllIngredients[ingredient])[cuisineType] = 1

    return cuisineOccurenceForAllIngredients

"""
returns a tuple with two elements

the first element:
contains a dictionary with each unique ingredient (string) mapped to a tuple with two elements
the first element of the tuple is a list with all the recipes that the ingredient is in (list of strings)
the second element is a list with all the recipes that the ingredient is not in (list of strings)
e.g. 
    { "milk" : 
        (
            [{ "id" : 1234,
                "cuisine" : "mexican",
                "ingredients" : ["apples", "flour", "sugar", "milk"]},
                { "id" : 678,
                "cuisine" : "french",
                "ingredients" : ["milk", "ranch dressing"]}
            ],

            [{ "id" : 1234,
                "cuisine" : "chinese",
                "ingredients" : ["cheese", "bread", "ham"]},
                { "id" : 678,
                "cuisine" : "british",
                "ingredients" : ["lettuce", "ranch dressing"]}
            ]
            
        )
    }

the second element:
contains a dictionary of cuisine types (strings) mapped to
the number of times the cuisine appears in the recipes parameter (int)
e.g. { "mexican" : 23, "chinese" : 56, "british" : 5 }

time = O((recipes * unique ingredients) + ingredients + unique ingredients)
space = O(unique ingredients * recipes + cuisine types)
"""
def getIngredientsWithRecipesAndCuisineAmounts(recipes):
    ingredients = getUniqueIngredients(recipes)
    ingredientsAndRecipes = dict()
    # initializing the ingredient and recipes dictionary
    for ingredient in ingredients:
        ingredientsAndRecipes[ingredient] = ([], [])

    cuisineCounts = dict()
    for recipe in recipes:
        # gets the cuisine counts
        cuisine = recipe.get("cuisine")
        if cuisine in cuisineCounts:
            cuisineCounts[cuisine] += 1
        else:
            cuisineCounts[cuisine] = 1

        ingredientList = recipe.get("ingredients")
        # goes over all the unique ingredients, if it is in the recipe list then add the recipe
        # to the first list in the tuple other wise add it to the second
        for ingredient in ingredientsAndRecipes:
            recipeTuple = ingredientsAndRecipes.get(ingredient)
            if ingredient in ingredientList:
                recipeTuple[0].append(recipe)
            else:
                recipeTuple[1].append(recipe)

    return (ingredientsAndRecipes, cuisineCounts)

# introduces feature randomness by giving the tree a random set of len(cuisines)/2 choices (unique) from the original
# maybe try changing or randomizing the number of choices rather than just 1/2 the amount of choices
def cuisineCountsWithFeatureRandomness(cuisineOccurenceForAllIngredients):
    if len(cuisineOccurenceForAllIngredients) <= 1:
        return cuisineOccurenceForAllIngredients
    else:
        randomFeatures = random.sample(list(cuisineOccurenceForAllIngredients), int(featureRandomnessFactor * len(cuisineOccurenceForAllIngredients)))
        newSet = set()
        for i in randomFeatures:
            newSet.add(i)
        return newSet

# calculates the entropy of a list of recipes according to the cuisine types
# O(1) with cuisineCounts precomputed, O(recipes) without
def calculateEntropy(recipes, cuisineAmounts=None):
    totalRecipes = len(recipes)
    if cuisineAmounts == None:
        cuisineCounts = getCuisineAmounts(recipes)
    else:
        cuisineCounts = cuisineAmounts
    
    entropy = 0
    for cuisine in cuisineCounts:
        cuisineOccurences = cuisineCounts.get(cuisine)
        entropy -= (cuisineOccurences / totalRecipes) * math.log2(cuisineOccurences / totalRecipes)
        
    return entropy

# calculates the information gain of the set of recipes using the ingredient param as the target variable
def calculateInformationGain(recipes, ingredient, cuisineAmounts=None, recipeListsWithAndWithout=None):
    totalRecipes = len(recipes)

    # if this has been precomputed then use it, if not do it yourself lazy
    if cuisineAmounts == None:
        cuisineCounts = getCuisineAmounts(recipes)
    else:
        cuisineCounts = cuisineAmounts

    # if this has been precomputed then use it, if not do it yourself lazy
    # makes two lists, one with all the recipes that the the ingredient param and one with all the recipes that dont
    # also get the cuisine counts while going through recipes to save some time
    if recipeListsWithAndWithout == None:
        recipesWithIngredient = []
        recipesWithoutIngredient = []
        cuisineCountsOfRecipesWith = dict()
        cuisineCountsOfRecipesWithout = dict()
        for recipe in recipes:
            cuisine = recipe.get("cuisine")
            if ingredient in recipe.get("ingredients"):
                recipesWithIngredient.append(recipe)

                if cuisine in cuisineCountsOfRecipesWith:
                    cuisineCountsOfRecipesWith[cuisine] += 1
                else:
                    cuisineCountsOfRecipesWith[cuisine] = 1
            else:
                recipesWithoutIngredient.append(recipe)

                if cuisine in cuisineCountsOfRecipesWithout:
                     cuisineCountsOfRecipesWithout[cuisine] += 1
                else:
                    cuisineCountsOfRecipesWithout[cuisine] = 1
    else:
        recipesWithIngredient = recipeListsWithAndWithout[0]
        recipesWithoutIngredient = recipeListsWithAndWithout[1]
        cuisineCountsOfRecipesWith = None
        cuisineCountsOfRecipesWithout = None

    # the entropys of the two sets after the ingredient split times their size ratios
    weightedEntropyAfterSplit = ((len(recipesWithIngredient) / totalRecipes) * calculateEntropy(recipesWithIngredient, cuisineCountsOfRecipesWith)) + ((len(recipesWithoutIngredient) / totalRecipes) * calculateEntropy(recipesWithoutIngredient, cuisineCountsOfRecipesWithout))

    return calculateEntropy(recipes, cuisineCounts) - weightedEntropyAfterSplit

class decisionTreeNode:
    # trueBranch (dTreeNode) the Node all recipes that DO have the ingredient in ingredientSplit will go to
    # falseBranch (dTreeNode) the Node all recipes that DON'T have the ingredient in ingredientSplit will go to
    # ingredientSplit (string) is the ingredient being split at the Node
    # cuisineClassification (string) is the classification for the recipe reaches this Node, None for all non-leaf nodes
    def __init__(self, trueBranch = None, falseBranch = None, ingredientSplit = None, cuisineClassification = None):
        self.inForest = False
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.ingredientSplit = ingredientSplit
        self.cuisineClassification = cuisineClassification

    # classifies a recipe using a the self param decision tree and return the classification (string)
    # O(tree depth)
    def test_point(self, recipe):
        if self.cuisineClassification is not None:
            return self.cuisineClassification
        else:
            if self.ingredientSplit in recipe["ingredients"]:
                return self.trueBranch.test_point(recipe)
            else:
                return self.falseBranch.test_point(recipe)

    # writes a tree to a text file with the given file name
    def writeTreeToFile(self, fileName):
        file = open("source/" + fileName + ".txt", "w")
        file.write(self.treeToString())
        file.close()

    # turns a decision tree into a string
    # root[leftBranch[leftleftBranch[_leftleftleftBranch,_rightrightleftBranch],_rightleftBranch],_rightBranch]
    def treeToString(self):
        if self.cuisineClassification is None:
            return self.ingredientSplit + "[" + self.trueBranch.treeToString() + "," + self.falseBranch.treeToString() + "]"
        else:
            # marks that this a cuisine and not an ingredient
            return "_" + self.cuisineClassification

    # turns the given tree into the tree stored in the file with the given file name
    def loadTreeFromFile(self, fileName):
        file = open("source/" + fileName + ".txt", "r")
        treeString = file.readline()
        file.close()
        # self.makeTreeFromString(treeString)

    """
    def makeTreeFromString(self, treeString):
        if treeString[:1] == "_":
            return treeString.partition("[")
        else:
            self.ingredientSplit = (re.match("(.*?)]", treeString).group())
    """
        

    """
    return the root of a decision tree built using recipes in the recipes param
    """
    # uses information gain with no precomputation
    # time (not accurate anymore) - O(nodes(2recipes + ingredients + uIngredients))
    # space (not accurate anymore) - O(nodes * (2cuisine types + unique ingredients + recipes))
    def makeDecisionTree(self, recipes):
        cuisineCounts = getCuisineAmounts(recipes)
        uniqueIngredients = getUniqueIngredients(recipes)

        # introduce feature randomness if this is a forest decision tree
        if self.inForest:
            uniqueIngredients = cuisineCountsWithFeatureRandomness(uniqueIngredients)

        # if only one type of cuisine left then make a leaf node with that classification
        if len(cuisineCounts) == 1:
            onlyCuisine = list(cuisineCounts.keys())[0]
            return decisionTreeNode(cuisineClassification=onlyCuisine)
        else:
            # tuple that keeps track of the ingredient with highest info gain so far
            # (ingredient name, ingredient info gain)
            bestInfoGainIngredient = ("wazowski", -1)
            # gets the ingredient with the best info gain
            for ingredient in uniqueIngredients:
                infoGain = calculateInformationGain(recipes, ingredient, cuisineCounts)
                if infoGain > bestInfoGainIngredient[1]:
                    bestInfoGainIngredient = (ingredient, infoGain)

            # sets this node split to the ingredient with the best info gain
            self.ingredientSplit = bestInfoGainIngredient[0]

            # divides recipes based on if they have the ingredient being split at this Node
            recipesWithIngredient = []
            recipesWithoutIngredient = []
            for recipe in recipes:
                if self.ingredientSplit in recipe.get("ingredients"):
                    recipesWithIngredient.append(recipe)
                else:
                    recipesWithoutIngredient.append(recipe)
            
            # case excutes if no recipes are being split, all either have the ingredient or dont
            # this means all ingredients left are resulting in no info gain
            # so makes a leaf node with the classification of the cuisine that most of recipe have
            if len(recipesWithIngredient) == 0 or len(recipesWithoutIngredient) == 0:
                if len(recipesWithIngredient) == 0:
                    cuisineAmounts = getCuisineAmounts(recipesWithoutIngredient)
                else:
                    cuisineAmounts = getCuisineAmounts(recipesWithIngredient)

                # gets the cuisine that is majority and sets that as this node's classification
                majorityCuisine = ("wazowski", -1)
                for cuisine in cuisineAmounts:
                    if cuisineAmounts.get(cuisine) > majorityCuisine[1]:
                        majorityCuisine = (cuisine, cuisineAmounts.get(cuisine))
                return decisionTreeNode(cuisineClassification = majorityCuisine[0], ingredientSplit=None)
            else:
                # recures and on the ingredients that have the split ingredient and don't and then returns itself
                self.trueBranch = decisionTreeNode.makeDecisionTree(decisionTreeNode(), recipesWithIngredient)
                self.falseBranch = decisionTreeNode.makeDecisionTree(decisionTreeNode(), recipesWithoutIngredient)
                return self