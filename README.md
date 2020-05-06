**Recipe Classifier**

By: Mary Ann Hazuga, Heoliny Jung, Joe Soellner

**Introduction:**

There are many different types of cuisines in the world today, and many of them use similar
ingredients for a large portion of their dishes. Using AI, this program attempts to build a system
that can classify any recipe to one of 20 cuisines using only its ingredients list.

Our problem set is a database of around 40,000 recipes, each with a cuisine type and ingredients
list as attributes. We have hand implemented and optimized a random forest algorithm for our 
primary model, and have used ML libraries to implement SVM, KNN, and MLP as competitors. We found 
that our random forest performed the best out of all the algorithms, reporting accuracies of around
59%-63% where the next best algorithm produced an accuracy of around 54%-58%.

**Running the demos:**

To see the algorithm in action, we have provided a few options for testing. The first is the 
main.py module, which will