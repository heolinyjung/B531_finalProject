# Cleans the json data
# Removes all quantities and qualities from ingredients list
# Gathers statistics to help us find errors in data and add to cleaning list
# Will create our new file with only the data we need

import json
from source import jsonConverter

class json_clean(object):

    def __init__(self, json_convert=None):
        self.json_convert = json_convert
        self.ingredient_cleaning_list = ['small', 'large', 'medium', '1','2','3','4','5']
        self.ingredient_counts = {}
        self.category_counts = {}

    def add_ingredient_to_clean(self, ingredient):
        self.ingredient_cleaning_list.append(ingredient)

    # removes all words in cleaning list from recipes in the dictionary
    def clean_ingredients(self):
        for recipe in self.json_convert.recipe_dict:
            new_ingredients = []
            for ingredient_phrase in recipe['ingredients']:
                new_phrase = ""
                for ingredient in ingredient_phrase.split():
                    if ingredient not in self.ingredient_cleaning_list:
                        new_phrase += " " + ingredient
                new_ingredients.append(new_phrase)

            recipe['ingredients'].clear()
            recipe['ingredients'].append(new_ingredients)
            print(recipe['ingredients'])


if __name__ == '__main__':
    recipe_convert = jsonConverter.json_convert()
    recipe_convert.set_file("full_format_recipes.json")
    recipe_convert.parse_file()
    cleaner = json_clean(recipe_convert)
    cleaner.clean_ingredients()
    recipe_convert.print_recipes()
