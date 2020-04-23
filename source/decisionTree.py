import math
import json
import numpy as np

"""
fills a dictionary with all the cuisine type and how many times they appear in the training dataset

recipes param in a list of dictionaries
each dict in the list is a recipe with keys id, cuisine, and ingredients (strings)
id is assigned to the recipe id (int)
cuisine is assigned to the cuisine type (string)
ingredients is assigned to the list of ingredients (list of strings)

returns a dictionary of cuisine types (strings) mapped to
the number of times the cuisine appears in the recipes parameter (int)
"""
# O(n)
def getCuisineAmounts(recipes):
    cuisineCounts = dict()
    for recipe in recipes:
        cuisine = recipe.get("cuisine")
        if cuisine in cuisineCounts:
            cuisineCounts[cuisine] += 1
        else:
            cuisineCounts[cuisine] = 1
    return cuisineCounts

# calculates the entropy of a list of recipes according to the cuisine types
# recipes param same as getCuisineAmounts comment
# O(1) with cuisineCounts precomputed O(n) w/o
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

# returns a dictionary of each cuisine and the number of times that ingredient is in that cuisine type
# key is the cuisine type (string) and the value is the number of times the ingredient is in the cuisine (int)
# ingredients is a string
# recipes param same as getCuisineAmounts comment
# O(ingrdients)?
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

# calculates the information gain of the set of recipes using the ingredient param as the target variable
# recipes param same as getCuisineAmounts comment
# O(1) if have the precomputations else O(yikes)
def calculateInformationGain(recipes, ingredient, cuisineAmounts=None, ingredientinCuisineAmounts=None):
    totalRecipes = len(recipes)
    ingredientEntropy = 0

    # if these things have been precomputed then use them, if not do it yourself 
    if ingredientinCuisineAmounts == None:
        ingredientinCuisineCounts = ingredientOccurencesPerCuisineType(recipes, ingredient)
    else:
        ingredientinCuisineCounts = ingredientinCuisineAmounts

    if cuisineAmounts == None:
        cuisineCounts = getCuisineAmounts(recipes)
    else:
        cuisineCounts = cuisineAmounts

    for cuisine in cuisineCounts:
        # !!!!!!!!!!!!!!!!!!!!! try .items()
        cuisineOccurences = cuisineCounts.get(cuisine)
        ingredientOccurences = ingredientinCuisineCounts.get(cuisine)
        if ingredientOccurences == None:
            ingredientOccurences = 0

        if ingredientOccurences == 0 or cuisineOccurences - ingredientOccurences == 0:
            cuisineEntropy = 0
        else:
            cuisineEntropy = (-(ingredientOccurences / cuisineOccurences) * math.log2(ingredientOccurences / cuisineOccurences)) - ((cuisineOccurences - ingredientOccurences) / cuisineOccurences) * math.log2((cuisineOccurences - ingredientOccurences) / cuisineOccurences)
        ingredientEntropy += (cuisineOccurences / totalRecipes) * cuisineEntropy

    return calculateEntropy(recipes, cuisineCounts) - ingredientEntropy

# O(ingredients)
def getUniqueIngredients(recipes):
    ingredientsSoFar = set()
    for recipe in recipes:
        ingredientList = recipe.get("ingredients")
        for ingredient in ingredientList:
            if ingredient not in ingredientsSoFar:
                ingredientsSoFar.add(ingredient)

    return ingredientsSoFar

# O(ingredients)
# puts all the ingredients lists in sets instead of lists 
def putIngredientsInSets(recipes):
    for recipe in recipes:
        ingredientList = recipe.get("ingredients")
        recipe["ingredients"] = set(ingredientList)

    return recipes

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

# O(2 * ingredients)
def filterIngredients(recipes, filter):
    ingredientCounts = getIngredientCounts(recipes)

    for recipe in recipes:
        ingredientList = recipe.get("ingredients")
        for ingredient in ingredientList:
            if ingredientCounts.get(ingredient) < filter:
                ingredientList.remove(ingredient)

    return recipes

# returns a dictionary of all unique ingredients mapped to a dictionary of each cuisine type mapped to the
# number of times that ingredients appears in a recipe of that cuisine type
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

class dTreeNode:
    # maybe add depth?
    # trueBranch (dTreeNode) the Node all recipes that DO have the ingredient in ingredientSplit will go to
    # falseBranch (dTreeNode) the Node all recipes that DON'T have the ingredient in ingredientSplit will go to
    # ingredientSplit (string) is the ingredient being split at the Node
    # cuisineClassification (string) is the classification for whatever reaches this Node, None for all non-leaf nodes
    def __init__(self, trueBranch = None, falseBranch = None, ingredientSplit = None, cuisineClassification = None):
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.ingredientSplit = ingredientSplit
        self.cuisineClassification = cuisineClassification

    # O(nodes(2recipes + ingredients + uIngredients))
    def decisionTree(self, recipes):
        # can maybe combine get cuisineAmounts into getCuisineOccurenceForAllIngredients
        cuisineCounts = getCuisineAmounts(recipes)
        cuisineOccurenceForAllIngredients = getCuisineOccurenceForAllIngredients(recipes)
        if len(cuisineCounts) == 1:
            onlyCusisine = None
            for onlykey in cuisineCounts:
                onlyCusisine = onlykey
            return dTreeNode(cuisineClassification = onlyCusisine)
        else:
            # gets the ingredient with the best info gain
            bestInfoGainIngredient = ("wazowski", -1)
            for ingredient in cuisineOccurenceForAllIngredients:
                infoGain = calculateInformationGain(recipes, ingredient, cuisineCounts, cuisineOccurenceForAllIngredients.get(ingredient))
                if infoGain > bestInfoGainIngredient[1]:
                    bestInfoGainIngredient = (ingredient, infoGain)

            self.ingredientSplit = bestInfoGainIngredient[0]

            # divides recipes based on if they have the ingredient being split at this Node
            recipesWithIngredient = []
            recipesWithoutIngredient = []
            for recipe in recipes:
                if self.ingredientSplit in recipe.get("ingredients"):
                    recipesWithIngredient.append(recipe)
                else:
                    recipesWithoutIngredient.append(recipe)
            
            # if no information is being gained anymore then make this a leaf node with the cuisine
            # classification of the majority of recipes left

            # !!!!!!!!!!!!!!could maybe just use the cuisineOccurenceForAllIngredients
            if len(recipesWithIngredient) == 0 or len(recipesWithoutIngredient) == 0:
                if len(recipesWithIngredient) == 0:
                    cuisineAmounts = getCuisineAmounts(recipesWithoutIngredient)
                else:
                    cuisineAmounts = getCuisineAmounts(recipesWithIngredient)

                majorityCuisine = ("wazowski", -1)
                for cuisine in cuisineAmounts:
                    if cuisineAmounts.get(cuisine) > majorityCuisine[1]:
                        majorityCuisine = (cuisine, cuisineAmounts.get(cuisine))
                return dTreeNode(cuisineClassification = majorityCuisine[0])
            else:
                self.trueBranch = dTreeNode.decisionTree(dTreeNode(), recipesWithIngredient)
                self.falseBranch = dTreeNode.decisionTree(dTreeNode(), recipesWithoutIngredient)
                return self