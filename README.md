# Credit-Risk-Prediction

# Overview and Goals:
The objective of this assignment are the following:
Experiment with various classification models.
Think about dealing with data with different attribute types: categorical and numerical (ratio). 
Think about dealing with potentially sensitive or protected attributes like gender, race, age.
Think about dealing with imbalanced data i.e., class labels with varying distribution
F1 Scoring Metric

# Detailed Description:
Develop predictive models that can determine someone’s credit risk 0 - high risk, 1-low risk .
The goal is to allow you to develop predictive models that can determine given a particular individual whether their credit risk is high denoted by 0 or low denoted by1. As such, the goal would be to develop the best binary classification model.
Since the dataset is imbalanced the scoring function will be the F1-score instead of Accuracy.
Caveats:
+ Remember not all features will be good for predicting credit risk. Think of feature selection, engineering, reduction
+ The dataset has an imbalanced distribution i.e., within the training set there are 24720 (0) and 7841 (1). No information is provided for the test set regarding the distribution.

# Data Description:
The training dataset consists of 32561 records and the test dataset consists of 13305 records. We provide you with the training class labels and the test labels are held out.
In the training file there are 13th attributes with the 13-th attribute (or column) being the class label of interest. In the testing file there are 12 attributes.

train.csv Description
id - unique identifier
F1 - Continuous value describing number of years since last degree was completed F2 - Continuous value indicating hours worked per week
F3 - Categorical Value
F4 - Categorical Value indicating type of occupation
F5 - continuous value denoting gains
F6 - continuous value denoting loss
F7 - Categorical value denoting marital status
F8 - Categorical value denoting type of employment (e.g., Self)
F9 - Categorical Value denoting education type
F10 - Categorical Value denoting different race
F11 - Categorical - Female/Male
credit - 0: Bad, 1: Good

test.csv: Testing set


