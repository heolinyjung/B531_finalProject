# reference: https://linuxconfig.org/how-to-parse-data-from-json-into-python
#            https://www.w3schools.com/python/python_json.asp
import json

# could probably put this all in functions and would be better but am lazy
if __name__ == '__main__':
    # opens training and testing dataset as json objects, train and test, basically just a list of dicts
    # only difference between train and test is that test does not have the cuisine types
    # each dict in the list is a recipe with keys id, cuisine, and ingredients (strings)
    # id is assigned to the recipe id (int)
    # cuisine is assigned to the cuisine type (string)
    # ingredients is assigned to the list of ingredients (list of strings)
    with open('train.json') as f:
        train = json.load(f)
    with open('test.json') as f2:
        test = json.load(f2)

    # dicts of ingredients and cusisine counts, ingredient/cuisine (string) assigned to ingredient/cuisine count (int)
    cuisineCounts = dict()
    ingredientCounts = dict()

    # fills a dictionary with all the ingredients and how many times they appear in the training dataset
    for recipe in train:
        ingredientList = recipe.get("ingredients")
        for ingredient in ingredientList:
            if ingredient in ingredientCounts:
                ingredientCounts[ingredient] += 1
            else:
                ingredientCounts[ingredient] = 1

    # fills a dictionary with all the cuisine type and how many times they appear in the training dataset
    for recipe in train:
        cuisine = recipe.get("cuisine")
        if cuisine in cuisineCounts:
            cuisineCounts[cuisine] += 1
        else:
            cuisineCounts[cuisine] = 1

    # counts the number of total ingredients in the training dataset
    totalNumOfIngredients = 0
    for ingredientCount in ingredientCounts.values():
        totalNumOfIngredients += ingredientCount

    # get the numbers of unique ingredients that appear more than a certain number of times in the training dataset
    filter = 5
    ingredientAmountsBiggerThanFilter = 0
    for ingredient in ingredientCounts:
        if ingredientCounts.get(ingredient) >= filter:
            ingredientAmountsBiggerThanFilter += 1

    # prints certain number of ingredients and how many times they appear in the training dataset
    # biased towards printing items with high number of appearances but not in a sorted type of way
    # change results to the number of ingredient and counts you want printed, change results to -1 for all
    print("Ingredients and the number of times they appear in the training dataset")
    results = 20
    counter = 0
    for ingredient, amount in ingredientCounts.items():
        print(ingredient, amount)
        counter += 1
        if counter > results and results != -1:
            break
    print()

    # prints all cuisine types and number of times they appear in total in the training dataset
    print("Types of cuisine and the number of times they appear in the training dataset")
    for cuisine, amount in cuisineCounts.items():
        print(cuisine, amount)

    print()
    print("# of total ingredients:", totalNumOfIngredients)
    print("# of unique ingredients:", len(ingredientCounts))
    print("# of unique ingredients with amounts more than or equal to", filter, ":", ingredientAmountsBiggerThanFilter)
    print("# of each ingredient on average:", int(totalNumOfIngredients / len(ingredientCounts)))
    print("# of cusines:", len(cuisineCounts))
    print("# of training items:", len(train))
    print("# of testing items:", len(test))