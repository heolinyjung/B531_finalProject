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

    options = ["1) bulgogi (korean) - yellow onion, green onion, soy sauce, white sugar, sesame seeds, garlic, sesame oil, red pepper flakes, ginger, pepper, sirloin steak, honey", "2) souffle (french) - butter, sugar, milk, flour, butter, vanilla extract, eggs, berries", "3) yorkshire pudding (british) - flour, salt, eggs, milk, beef juice", "4) chicken enchiladas (mexican) - vegetable oil, chicken breast, salt, pepper, cumin powder, garlic powder, red onion, garlic, corn, green chiles, tomatoes, flour, tortilla, chedder cheese", "5) shrimp jambalaya (creole) - garlic cloves, olive oil, onions, celery, bell pepper, cayenne pepper, tomatoes, Worcestershire sauce, hot sauce, bay leaves, black pepper, shrimp, green onions", "6) chicken fried rice (chinese) - sesame oil, canola oil, chicken breast, carrots, green onions, garlic cloves, eggs, rice, soy sauce, salt, pepper", "7) pasta (italian) - eggs, flour, olive oil, tomatoes, basil, water, salt"]

    print("Please enter a list of ingredients separated by commas and then press enter the run the model on those ingredients.")
    print("For example: flour, sugar, eggs, yeast, salt")
    print("Press S/s if you would like to see some sample recipes or press E/e to enter your own ingredients.")
    ingredientList = None

    showEachRecipe = "a"
    while showEachRecipe != 'S' and showEachRecipe != 's' and showEachRecipe != 'E' and showEachRecipe != 'e':
        showEachRecipe = input()
        if showEachRecipe != 'S' and showEachRecipe != 's' and showEachRecipe != 'E' and showEachRecipe != 'e':
            print("Please enter S or s to see an example list of recipes and E or e for enter your own.")

    if showEachRecipe == 's' or showEachRecipe == 'S':
        print()
        for recipe in options:
            print(recipe)
        print()
        print("Would you like to run the model on one of these recipes? Option: Y/y for yes and N/n for no.")
        runOnExample = "a"
        while runOnExample != 'Y' and runOnExample != 'y' and runOnExample != "N" and runOnExample != "n":
            runOnExample = input()
            if runOnExample != 'Y' and runOnExample != 'y' and runOnExample != "N" and runOnExample != "n":
                print("Please enter Y or y for yes and N or n for no.")

        if runOnExample == "Y" or runOnExample == "y":
            print("Which example would you like? Enter a number from 1-7.")
            recipeOption = 10
            while not (7 >= recipeOption and recipeOption >= 1):
                recipeOption = int(input())
                if not (7 >= recipeOption and recipeOption >= 1):
                    print("Please enter a number from 1-7.")

            optionIngredientList = ["yellow onion, green onion, soy sauce, white sugar, sesame seeds, garlic, sesame oil, red pepper flakes, ginger, pepper, sirloin steak, honey", "butter, sugar, milk, flour, butter, vanilla extract, eggs, berries", "flour, salt, eggs, milk, beef juice", "vegetable oil, chicken breast, salt, pepper, cumin powder, garlic powder, red onion, garlic, corn, green chiles, tomatoes, flour, tortilla, chedder cheese", "garlic cloves, olive oil, onions, celery, bell pepper, cayenne pepper, tomatoes, Worcestershire sauce, hot sauce, bay leaves, black pepper, shrimp, green onions", "sesame oil, canola oil, chicken breast, carrots, green onions, garlic cloves, eggs, rice, soy sauce, salt, pepper", "eggs, flour, olive oil, tomatoes, basil, water, salt"]
            ingredientList = optionIngredientList[recipeOption - 1]
        
    print()
    if ingredientList == None:
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