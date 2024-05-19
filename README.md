## Module 12 Report

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
  
    The analysis in this notebook aids in the prediction of loan default risk. 

* Explain what financial information the data was on, and what you needed to predict.
  
    Information available in this dataset includes loan size and interest rate, borrower information such as income, accounts and previous defaults.
    Status of a loan, or risk of default is the metric that this notebook seeks to build models to predict.

* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
  
    From instruction:
  
        "A value of 0 in the “loan_status” column means that the loan is healthy. A value of 1 means that the loan has a high risk of defaulting."
  
    Loan status contains boolean values, so given all other columns in the dataset, a predictive model for loan status of a value of "1" or "0" may provide indication
    of default risk for a potential new loan, based on loan and borrower information.

* Describe the stages of the machine learning process you went through as part of this analysis.
  
    * The data is first split into training and testing sets. Loan status is the value being tested, and loan and borrower information is used to
      indicate loan status. Loan status (y) is designated as the labels set, and the remaining columns (X) are designated as features.

    * The "train_test_split" function is used to split the data (documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). The genrated "y_train" and               "X_train"     is used to fit a logistic regression model that analyzes the data to find correlation between
      loan and borrower data, and loan status.

    * Predictions from this model are saved as "prediction" by fittting the model to the testing feature data "X_test".

    * A confusion matrix is generated using "y_test" and "prediction", as well as a classification report to deisplay results of the model. 

* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any other algorithms).
  
    "LogisticRegression" (documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) is a model
    that aids in classification of the binary metric, such as loan status for this data. Logistic regression uses a sigmoid funtion to classify data trends,
    due to bounds of the function being 1 and 0 such as in a binary case. If the predicted value is higher than 0.5, it is classified as 1. Values
    lower than 0.5 are classified as 0. Linear regression provides a "slope" value as a trend line that best predicts where a data point of two values
    is likely to occur, logistic regression classifies whether a data point is more likely to be a 1 or a 0.

    A confusion matrix is generated to evaluate the model's performance. The matrix presents the count of correct and incorrect predictions. The 
    "two by two" output of the matrix has the following format:

        [True Positives, False Positives,

        False Negatives, True Negatives]

    Finally a classification report displays the accuracy, precision, and recall of the model. Accuracy is the measure of how many correct predictions
    the model has out of the total number of predictions. Precision is the measure of how many correct positives were predicted out of total positive predictions,
    and recall measures how many correct negatives out of total negative predictions a model has.

## Results

Using bulleted lists, describe the accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
    * Accuracy: 
        TP + TN / TP + TN + FP + FN = 19226 / 19226 + 158 = 0.99

    * 0 - (Low Risk)
        * Precision: 1.00
            Precision is high, healthy loan status is calssified correctly.
        * Recall: 0.99
            Recall is high, 99% of healthy loans are predicted correctly.

    * 1 - (High Risk)
        * Precision: 0.85
            Precision is 85%, less default loans are classified correctly than healthy loans.

        * Recall: 0.91
            Recall is fairly high, 91% of default loans are predicted correctly.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

    I would recommend this model with a caveat. The model does a great job of classifying and predicting healthy loans, ensuring that borrowers 
    that deserve a loan are recommended. The model is less precise when it comes to loans with a higher default risk. Due to these metrics, the model
    ensures that healthy loans are issued and may recommend issuing loans that are high risk in error. Typically, banks would want to issue as many
    loans as possible. Although a defaulted loan is not desireable, alternative loan terms as well as collections methods are additional steps that can
    aid in the cases of high risk loans. 

    Caveat- should a bank have more stringent and conservative lending practices, this model may not be precise enough to produce the desired predictions.
