try:
    from source.RandomForest import *
except ModuleNotFoundError:
    from RandomForest import *
import pickle

if __name__ == '__main__':
    try:
        with open('75Filter30TreeForest.pickle', "rb") as f:
            forest = pickle.load(f)
    except FileNotFoundError:
        with open('source/75Filter30TreeForest.pickle', "rb") as f:
            forest = pickle.load(f)
    
    print()
    print("Here is a list of the 20 cuisine classifications your cusisine can be classified as:")
    print("Greek, Southern US, Filipino, Indian, Jamaican, Spanish, Italian, Mexican, Chinese, British")
    print("Thai, Vietnamese, Cajun/Creole, Bazilian, French, Japanese, Irish, Korean, Moroccan, Russian")
    print()

    print("Please enter a list of ingredients separated by commas and then press enter the run the model on those ingredients.")
    ingredientList = input()
    ingredientList = ingredientList.lower()
    ingredientList = ingredientList.split(',')
    cleanIngredientList = []
    for ingredient in ingredientList:
        cleanIngredientList.append(ingredient.strip())

    print("Would you like to know the result of each individual tree? Option: Y/y or N/n")
    showEachTreeResult = "a"    
    while showEachTreeResult != 'Y' and showEachTreeResult != 'y' and showEachTreeResult != "N" and showEachTreeResult != "n":
        showEachTreeResult = input()
        if showEachTreeResult != 'Y' and showEachTreeResult != 'y' and showEachTreeResult != "N" and showEachTreeResult != "n":
            print("Please enter Y or y for yes and N or n for no.")
    print()

    mockRecipe = {"ingredients"  : cleanIngredientList}
    resultCounts = dict()
    total = 0
    for tree in forest.forest:
        total += 1
        cuisine = tree.test_point(mockRecipe)
        if showEachTreeResult == "Y" or showEachTreeResult == "y":
            print("Tree", total, "said the recipe is", cuisine)
        if cuisine in resultCounts:
            resultCounts[cuisine] += 1
        else:
            resultCounts[cuisine] = 1
    
    if showEachTreeResult == "Y" or showEachTreeResult == "y":
        print()
    
    sortedResults = sorted(resultCounts.items(), key=lambda x: x[1], reverse=True)

    for result in sortedResults:
        print(str(result[1]) + "/" + str(total), "trees said this recipe is", result[0])

    print()
    forestResult = sortedResults[0]
    print("The random forest declares this recipe to be", forestResult[0] + "!")