#-------------------------------------------------------------------------
# AUTHOR: Jeremiah Garcia
# FILENAME: perceptron.py
# SPECIFICATION: Runs the Perception and MLP models.
# FOR: CS 4210- Assignment #4
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

highest_percep_accuracy = 0
highest_MLP_accuracy = 0

for ns in n: #iterates over n

    for rs in r: #iterates over r

        #iterates over both algorithms
        #-->add your Pyhton code here
        perceps = [True, False]
        for percep in perceps: #iterates over the algorithms

            #Create a Neural Network classifier
            if percep:
                clf = Perceptron(eta0 = ns, shuffle = rs, max_iter=1000)    #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
            else:
                clf = MLPClassifier(activation='logistic', learning_rate_init = ns, hidden_layer_sizes = 25, shuffle = rs, max_iter=1000)
            #use those hyperparameters: activation='logistic', learning_rate_init = learning rate,
            #                          hidden_layer_sizes = number of neurons in the ith hidden layer - use 1 hidden layer with 25 neurons,
            #                          shuffle = shuffle the training data, max_iter=1000
            #-->add your Pyhton code here

            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            #for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #--> add your Python code here

            total_pred = 0
            correct_pred = 0

            for (x_testSample, y_testSample) in zip(X_test, y_test):
                pred = clf.predict([x_testSample])
                total_pred += 1
                if pred[0] == y_testSample:
                    correct_pred += 1

            acc = correct_pred / total_pred


            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
            #--> add your Python code here

            if percep:
                if acc > highest_percep_accuracy:
                    highest_percep_accuracy = acc
                    print(f"Highest Perceptron accuracy so far: {acc:.2f}, Parameters: learning rate = {ns}, shuffle = \"{rs}\"")
            else:
                if acc > highest_MLP_accuracy:
                    highest_MLP_accuracy = acc
                    print(f"Highest MLP accuracy so far: {acc:.2f}, Parameters: learning rate = {ns}, shuffle = \"{rs}\"")









