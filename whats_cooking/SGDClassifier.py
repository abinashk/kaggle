'''
 The third approach that I used was Stochastic Gradient Descent model. It is a discriminative learning model that uses linear loss functions and different schemes of penalties for misclassification giving result to a model that is perfect fit for a classification problem that has large and sparse feature matrix. It is highly efficient and highly optimizable model with lots of optimizable parameters. Although it offered various parameters for tuning, I focused only on the loss function and the number of iteration over the train data. The parameters ‘loss’ and ‘n_iter’ controlled these. I tried three variations of this model with the following set of the two parameters:
	- Variation 1 : loss=”hinge” for linear SVM based loss function, penalty=”l2” and n_iter=5 which was the default number of iterations
	- Variation 2 : loss=”log” for logistic regression based loss function, penalty=”l2” and n_iter=5 which was the default number of iterations
	- Variation 3 : loss=”hinge” for linear SVM based loss function, penalty=”l2” and n_iter=10 to increase the number of iterations to 10
And the accuracy of classification for different variations were:
	- For the variation 1 the result was : 0.77082
	- For the variation 2 the result was : 0.72627
	- For the variation 3 the result was : 0.77343
'''

# imports basic packages like pandas and numpy
from pandas import DataFrame
import pandas as pd
import numpy as np

# imports nltk package and its lemmatizer class
import nltk
from nltk.stem import WordNetLemmatizer

# imports the text vectorizer from sklearn package
from sklearn.feature_extraction.text import TfidfVectorizer

# reads the train and test files into respective dataframes
traindf = pd.read_json("../input/train.json")
testdf = pd.read_json("../input/test.json")

# instantiates the WordNetLemmatizer class
wnl = WordNetLemmatizer()

# simple method to lemmatize a list of strings passed, returns the list of
# lemmatized strings


def lemmatize_each_row(x):
    y = []
    for each in x:
        y.append(wnl.lemmatize(each.lower()))
    return y

# Adds new column lemmatized_ingredients_list to the traindf using the
# apply() method of the dataframe
traindf['lemmatized_ingredients_list'] = traindf.apply(
    lambda row: lemmatize_each_row(row['ingredients']), axis=1)

# Empty list that is going to hold the list of all ingredients present in
# the whole traindf
all_ingredients_lemmatized = []

# Adds individual ingredients from each rows of traindf to the list
# initialized above by iterating through #each rows of traindf
for ingredients_lists in traindf.ingredients:
    for ingredient in ingredients_lists:
        # Applies lemmatization and conversion into lower case on the individual ingredients before
        # appending into the ingredients list
        all_ingredients_lemmatized.append(wnl.lemmatize(ingredient.lower()))

# Converts into set from list to remove the duplicate entries
all_ingredients_lemmatized = set(all_ingredients_lemmatized)

# Adds new column lemmatized_test_ingredients_list to the testdf using the
# apply() method of the #dataframe
testdf['lemmatized_test_ingredients_list'] = testdf.apply(
    lambda row: lemmatize_each_row(row['ingredients']), axis=1)

# Empty list that is going to hold the list of all ingredients present in
# the whole testdf
all_ingredients_lemmatized_test = []

# Adds individual ingredients from each rows of testdf to the list
# initialized above by iterating through #each rows of testdf
for ingredients_lists in testdf.ingredients:
    for ingredient in ingredients_lists:
        # Applies lemmatization and conversion into lower case on the individual ingredients before
        # appending into the ingredients list
        all_ingredients_lemmatized_test.append(
            wnl.lemmatize(ingredient.lower()))

# Converts into set from list to remove the duplicate entries
all_ingredients_lemmatized_test = set(all_ingredients_lemmatized_test)

# Union operation to get a set of distinct ingredients from traindf and testdf
all_ingredients_union = all_ingredients_lemmatized | all_ingredients_lemmatized_test

# Initializes a TfidVectorizer object using the distinct ingredients set
# created from traindf as the vocabulary, uses dummy lambda function to
# override the preprocessor and tokenizer
vect = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, sublinear_tf=True,
                       max_df=0.5, analyzer='word', stop_words='english', vocabulary=all_ingredients_union)

# Uses fit_transform() function to get sparse binary matrix fromt the
# traindf ingredients list
tfidf_matrix = vect.fit_transform(traindf['lemmatized_ingredients_list'])

# This matrix is the predictor matrix for traindf
predictor_matrix = tfidf_matrix

# The target class list is the cuisine column of traindf
target_classes = traindf['cuisine']

# Initializes a TfidVectorizer object using the distinct ingredients set
# created from testdf as the vocabulary, uses dummy lambda function to
# override the preprocessor and tokenizer
vect_test = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, sublinear_tf=True,
                            max_df=0.5, analyzer='word', stop_words='english', vocabulary=all_ingredients_union)

# Uses fit_transform() function to get sparse binary matrix fromt the
# traindf ingredients list
tfidf_matrix_test = vect_test.fit_transform(
    testdf['lemmatized_test_ingredients_list'])

# This matrix is the predictor matrix for testdf
predictor_matrix_test = tfidf_matrix_test

# Imports the SGDClassifier class from sklearn package
from sklearn.linear_model import SGDClassifier

# Instantiates the SGDClassifier class for variation 1 setting loss
# function as hinge, penalty #scheme as l2, and n_iter value set to 5,
# similar code with different parameter set for variation 2 #and 3
clf = SGDClassifier(loss="hinge", penalty="l2", n_iter=5)

# Fits the predictor_matrix and target_classes in the model
clf.fit(predictor_matrix, target_classes)

# Generates the prediction list using the predict() method of the model class
predicted_classes = clf.predict(predictor_matrix_test)

# Adds the predicted_class list as cuisine column in testdf
testdf['cuisine'] = predicted_classes

# Prepares submission dataframe from the id and cuisine columns of testdf
submission = testdf[['id',  'cuisine']]

# Writes the submission dataframe into a csv file that can be submitted to
# the competition
submission.to_csv("SGDSubmission.csv", index=False)
