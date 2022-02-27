# Charity Funding Predictor

## Background

The non-profit foundation Alphabet Soup wants to create an algorithm to predict whether or not applicants for funding will be successful. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

- **EIN** and **NAME**—Identification columns
- **APPLICATION_TYPE**—Alphabet Soup application type
- **AFFILIATION**—Affiliated sector of industry
- **CLASSIFICATION**—Government organization classification
- **USE_CASE**—Use case for funding
- **ORGANIZATION**—Organization type
- **STATUS**—Active status
- **INCOME_AMT**—Income classification
- **SPECIAL_CONSIDERATIONS**—Special consideration for application
- **ASK_AMT**—Funding amount requested
- **IS_SUCCESSFUL**—Was the money used effectively

## Instructions

### Step 1: Preprocess the data

Using your knowledge of Pandas and the Scikit-Learn’s `StandardScaler()`, you’ll need to preprocess the dataset in order to later compile, train, and evaluate the neural network model

### Step 2: Compile, Train, and Evaluate the Model

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

### Step 3: Optimize the Model

Using your knowledge of TensorFlow, optimize your model in order to achieve a target predictive accuracy higher than 75%. If you can't achieve an accuracy higher than 75%, you'll need to make at least three attempts to do so.

Optimize your model in a separate notebook in order to achieve a target predictive accuracy higher than 75%


## Summary

The purpose of this assignment was to create an algorithm to predict whether or not applicants for funding from a non-profit will be successful. The target for this model was IS_SUCCESSFUL. The features of this model are the Application Type, Affiliation, Classification, Use Case, Organization type, Status, Income Amount, Special Considerations, and Ask Amount. The EIN and Name fields were removed from the dataset as they did not contribute any useful information to the model.

I started with two layers (excluding the output layer) with ten nodes each and I ran it through 100 epochs. This resulted in a 72.96% accuracy score, a bit lower than the target. The first round of optimization, I decided to double the nodes to 20 and increase the epochs to 125. The accuracy score only increased marginally to 73.25%. The second round of optimization, I increased the nodes and epochs again, this time to 50 and 150, respectively. The accuracy score is still consistent with the previous attempts, coming in at 73.38%. 


I increased the number of nodes on each iteration, going from 10 nodes, to 20, then 50. I also increased the number of epochs to give the model a chance to increase its accuracy as much as possible. 


Overall, the model did not quite succeed, but it was close. I would likely continue to experiment with adding additional hidden layers as increasing nodes/epochs did not make much of a difference. If I were to choose a different model to use, I think a Random Forest would be a good option. It takes less training than a neural network and reduces data confusion. It would also allow you to tweak and refine your input fields more to see which have the greatest impact on the prediction results. 
