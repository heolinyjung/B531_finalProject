# Concepts:
# Bagging: given dataset with size N, randomly select N data points from the dataset for each tree.
#          If a data point is selected, it is not removed from the dataset, i.e. repeats are possible.
#          Ex. dataset = [1,2,3,4,5,6], tree 1 sample = [1,2,2,3,6,6], tree 2 sample = [2,3,3,3,4,4], ...
# Feature Randomness: Instead of looking through all features, splitting at nodes is done over a random set of features.
#                     Ex.:
#               Classic decision tree:             Forest:
#                            Node 1:                 Node 1:         Node 2:
#                           feature 1               feature 1       feature 1
#                           feature 2               --------        --------
#                  split -> feature 3      split -> feature 3       --------
#                           feature 4               --------        feature 4 <- split
#                           /       \               /       \       /       \
#                       left        right          L         R     L         R
#
# reference: https://towardsdatascience.com/understanding-random-forest-58381e0602d2
import random
from decisionTree import decisionTreeNode


class RandomForest:

    def __init__(self, tree_count=0, dataset=None):
        self.tree_count = tree_count
        self.dataset = dataset
        self.bags = []
        self.bag_data(self.dataset)
        self.forest = []
        self.create_forest()

    # Takes in a list (dataset) and returns a number of "bags" of values from that list of the same size but of random
    # selection w/ replacement. The number of bags is determined by the tree_count.
    # time complexity = O(dataset*tree_count)
    def bag_data(self, dataset):
        if dataset is None:
            return False
        for i in range(self.tree_count):
            self.bags.append(random.choices(dataset, k=len(dataset)))
        return True

    # Creates the forest using bagged data and feature randomness
    # O(tree_count*makeDecisionTree)
    def create_forest(self):
        for i in range(self.tree_count):
            new_tree = decisionTreeNode()
            new_tree.makeDecisionTree(self.bags[i])
            self.forest.append(new_tree)

    # Tests a recipe by running the test algorithm for each tree and returning the classification with the highest count
    # O(tree_count*O(test_point))
    def test_point(self, recipe):
        results = {}
        maxCuisine = ['', 0]
        for i in self.forest:
            cuisine = i.test_point(recipe)
            if cuisine in results.keys():
                results[cuisine] += 1
                if results[cuisine] > maxCuisine[1]:
                    maxCuisine[0] = cuisine
                    maxCuisine[1] += 1
            else:
                results[cuisine] = 1
                if maxCuisine[1] == 0:
                    maxCuisine[0] = cuisine
                    maxCuisine[1] = 1
        return maxCuisine[0]
