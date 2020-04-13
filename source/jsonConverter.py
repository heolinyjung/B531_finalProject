# reference: https://linuxconfig.org/how-to-parse-data-from-json-into-python
#            https://www.w3schools.com/python/python_json.asp

import json

class json_convert(object):

    def __init__(self, fileName = None):
        self.fileName = fileName
        self.recipe_dict = None

    # set file path
    def set_file(self, file):
        self.fileName = file

    # parse the file and load each recipe into a list of dictionaries
    def parse_file(self):
        with open(self.fileName, 'r') as f:
            self.recipe_dict = json.load(f)

    # utilize recipe_dict like so:
    def print_recipes(self):
        for recipe in self.recipe_dict:
            print(recipe['ingredients'])
            # prints the list of ingredients for each recipe

if __name__ == '__main__':
    recipe_convert = json_convert()
    recipe_convert.set_file("full_format_recipes.json")
    recipe_convert.parse_file()
