# For the second approach I chose Random Forest Classifier. The motivation
# being its better performance against the problem of overfitting. I
# thought overfitting can be a major issue in this particular
# classification problem. And the sparse nature of the feature sets also
# meant the cost of overfitting was probably high in this case. In
# addition to that I thought I had to try a decision tree based model
# owing to its popularity in text classification. One parameter I focused
# in was the n_estimators which indicated the number of trees generated by
# the model, I kept its value 100.

# The prediction score(the categorization accuracy) of this model was 0.72546.

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

# Imports the RandomForestClassifier class from sklearn package
from sklearn.ensemble import RandomForestClassifier

# Instantiates the RandomForestClassifier class setting n_estimator value
# to 100
clf = RandomForestClassifier(n_estimators=100)

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
submission.to_csv("RFSubmission.csv", index=False)