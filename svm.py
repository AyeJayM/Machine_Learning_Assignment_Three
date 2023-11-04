#-------------------------------------------------------------------------
# AUTHOR: Austin Martinez
# FILENAME: svm.py
# SPECIFICATION: First we define our hyperparameters. We then read in training data using Pandas.
# Then we initialize two variables - "highest_accuracy" and "best_hyperparameters" to keep track of 
# the highest accuracy and the corresponding hyperparameters respectively. We now iterate through
# the values of c, degree, kernel, and decision_function_shape where we will create our SVM 
# classifier with the current set of hyperparameters. We fit the SVM to the training data,
# make the SVM prediction for each test sample and start computing its accuracy. We only print
# if the current accuracy is the highest computed thus far.
# FOR: CS 4210- Assignment #3
# TIME SPENT: Total assignment - 6 hours.
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the training data by using Pandas library

X_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array


#  These variables will help us keep track of the highest recorded accuracy thus far as well as the corresponding hyperparameters.
highest_accuracy = 0.0
best_hyperparameters = ""

#Create 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
for c_value in c:
    for degree_value in degree:
        for kernel_value in kernel:
            for dfs_value in decision_function_shape:
                # Here we will create our SVM classifier with the current set of hyperparameters
                clf = svm.SVC(C=c_value, degree=degree_value, kernel=kernel_value, decision_function_shape=dfs_value)

                # Here we will fit our SVM to the training data.
                clf.fit(X_training, y_training)

                # Make the SVM prediction for each test sample and start computing its accuracy
                # Hint: to iterate over two collections simultaneously, use zip()
                # Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                correct_predictions = 0
                total_samples = len(y_test)

                for x_test_sample, y_test_sample in zip(X_test, y_test):
                    prediction = clf.predict([x_test_sample])
                    if prediction == y_test_sample:
                        correct_predictions += 1

                accuracy = correct_predictions / total_samples

                # Check if the calculated accuracy is higher than the previously one calculated. 
                # If so, update the highest accuracy and print it togetherwith the SVM hyperparameters. 
                # Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    best_hyperparameters = f"C={c_value}, degree={degree_value}, kernel={kernel_value}, decision_function_shape={dfs_value}"
                    #Print the highest accuracy and corresponding hyperparameters as of this point
                    print(f"Highest SVM accuracy so far: {highest_accuracy}, Parameters: {best_hyperparameters}")