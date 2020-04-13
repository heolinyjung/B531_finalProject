import json
import jsonConverter

if __name__ == '__main__':
    with open('source/train.json') as f:
        train = json.load(f)
    with open('source/test.json') as f2:
        test = json.load(f2)

    cuisineCounts = dict()
    ingredientCounts = dict()

    for item in train:
        ingredientList = item.get("ingredients")
        for ingredient in ingredientList:
            if ingredient in ingredientCounts:
                ingredientCounts[ingredient] += 1
            else:
                ingredientCounts[ingredient] = 1

    # makes a dictionary of all the cuisine type and how many times they appear in the json object
    for item in train:
        cuisine = item.get("cuisine")
        if cuisine in cuisineCounts:
            cuisineCounts[cuisine] += 1
        else:
            cuisineCounts[cuisine] = 1

    ingredientTotal = 0
    for item in ingredientCounts.values():
        ingredientTotal += item

    a = 0
    for ingredient, amount in ingredientCounts.items():
        print(ingredient, amount)
        a += 1
        if a > 20:
            break
    print()
    for cuisine, amount in cuisineCounts.items():
        print(cuisine, amount)
    print()
    print("# of total ingredients:", ingredientTotal)
    print("# of unique ingredients:", len(ingredientCounts))
    print("# of each ingredient on average:", int(ingredientTotal / len(ingredientCounts)))
    print("# of cusines:", len(cuisineCounts))
    print("# of training items:", len(train))
    print("# of testing items:", len(test))