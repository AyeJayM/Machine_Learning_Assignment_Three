#-------------------------------------------------------------------------
# AUTHOR: Austin Martinez
# FILENAME: svm.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #3
# TIME SPENT: how long it took you to complete the assignments
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

#Create 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
for c_value in c:
    for degree_value in degree:
        for kernel_value in kernel:
            for dfs_value in decision_function_shape:
                # Here we will create our SVM classifier with the current set of hyperparameters
                clf = svm.SVC(C=c_value, degree=degree_value, kernel=kernel_value, decision_function_shape=dfs_value)

                # Here we will fit our SVM to the training data.
                clf.fit(X_training, y_training)

                # Initialize variables to keep track of the highest accuracy and corresponding hyperparameters
                highest_accuracy = 0.0
                best_hyperparameters = ""

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

                # Print the highest accuracy and corresponding hyperparameters as of this point
                print(f"Highest SVM accuracy so far: {highest_accuracy:.2f}, Parameters: {best_hyperparameters}")