# Cleans the json data
# Removes all quantities and qualities from ingredients list
# Gathers statistics to help us find errors in data and add to cleaning list
# Will create our new file with only the data we need

import json
import jsonConverter

class json_clean(object):

    def __init__(self, json_convert=None):
        self.json_convert = json_convert
        self.ingredient_cleaning_list = ['small', 'large', 'medium', '1', '2', '3', '4', '5']
        self.ingredient_counts = {}
        self.category_counts = {}

    def add_ingredient_to_clean(self, ingredient):
        self.ingredient_cleaning_list.append(ingredient)

    # removes all words in cleaning list from recipes in the dictionary
    def clean_ingredients(self):
        for recipe in self.json_convert.recipe_dict:
            new_ingredients = []
            if 'ingredients' in recipe:
                for ingredient_phrase in recipe['ingredients']:
                    new_phrase = ""
                    for ingredient in ingredient_phrase.split():
                        if ingredient not in self.ingredient_cleaning_list:
                            new_phrase += " " + ingredient
                    new_ingredients.append(new_phrase[1:])

                recipe['ingredients'].clear()
                recipe['ingredients'].append(new_ingredients)

    # counts the quantity of each ingredient and updates the ingredient counts dictionary
    def count_ingredients(self):
        for recipe in self.json_convert.recipe_dict:
            if 'ingredients' in recipe:
                for ingredient_list in recipe['ingredients']:
                    for ingredient in ingredient_list:
                        if ingredient in self.ingredient_counts:
                            self.ingredient_counts[ingredient] += 1
                        else:
                            self.ingredient_counts[ingredient] = 1

    # gets the count of a given ingredient
    def get_ingredient_count(self, ingredient):
        if ingredient not in self.ingredient_counts:
            return None
        else:
            return self.ingredient_counts[ingredient]

    def count_categories(self):
        for recipe in self.json_convert.recipe_dict:
            if 'categories' in recipe:
                for category in recipe['categories']:
                    if category in self.category_counts:
                        self.category_counts[category] += 1
                    else:
                        self.category_counts[category] = 1

    def get_category_count(self, category):
        if category not in self.category_counts:
            return None
        else:
            return self.category_counts[category]

    def print_ingredient_counts(self):
        for recipe in self.ingredient_counts:
            print(recipe + ": " + str(self.ingredient_counts[recipe]))

if __name__ == '__main__':
    recipe_convert = jsonConverter.json_convert()
    recipe_convert.set_file("source/full_format_recipes.json")
    recipe_convert.parse_file()
    cleaner = json_clean(recipe_convert)
    cleaner.clean_ingredients()
    cleaner.count_ingredients()
    cleaner.print_ingredient_counts()
