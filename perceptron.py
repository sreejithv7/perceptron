"""
Binary and Multi-class Perceptron
"""

import numpy as np
import pandas as pd

SEED = 6
# SEED = 3

MAX_ITER = 20

def get_user_choice():
    """Gets user choice for choosing binary and multi-class perceptron, 
    and reporting accuracies for different classes of test and train data
    """
    while True:
        try:
            print('\n\nPlease enter your choice from either 1 - 4: \n')
            choice = input('''    1. Classify class-1 and class-2, class-2 and class-3, class-1 and class-3
    2. Classify class-1 v/s class-2 v/s class-3
    3. Add l2 regularisation to the multi-class classifier
    4. Quit\n\n''')
            if choice in '1234' and len(choice) == 1:
                if choice == '4':
                    exit()
                return choice
            else:
                raise ValueError
        except ValueError:
            print(f"Your choice '{choice}' is invalid\n")


def clean_data(train_data, test_data, choice, positive_class=None, negative_class=None):
    """Clean train and test data, by selecting only classes needed for the classification 
    and changing 'positive' and 'negative' class labels to numeric values (+1 or -1)

    Args:
        train_data (numpy array): train data to be cleaned
        test_data (numpy array): test data to be cleaned
        choice (string): user choice for binary or multi-class classification
        positive_class (string, optional): Class chosen as positive class 
        for binary classification. Defaults to None.
        negative_class (string, optional):  Class chosen as negative class 
        for binary classification. Defaults to None.

    Returns:
        numpy array: Cleaned train and test data
    """
    
    if choice == '1':
        # clean_data = np.delete(train_data, np.where(train_data[:,-1]=='class-3'), axis=0)
        # select only classes needed for classification
        clean_train_data = train_data[np.where((train_data[:,-1] == positive_class) | (train_data[:,-1] == negative_class))]
        clean_test_data = test_data[np.where((test_data[:,-1] == positive_class) | (test_data[:,-1] == negative_class))]

        # converting class labels to numeric values [+1 or -1] for binary classification
        clean_train_data[clean_train_data==positive_class] = 1
        clean_train_data[clean_train_data==negative_class] = -1

        # converting class labels to numeric values [+1 or -1] for binary classification
        clean_test_data[clean_test_data==positive_class] = 1
        clean_test_data[clean_test_data==negative_class] = -1

    elif choice == '2' or choice == '3':

        # Taking a copy of train data to prevent modification to the original data
        train_data_1 = train_data.copy()
        train_data_2 = train_data.copy()
        train_data_3 = train_data.copy()

        # converting class labels to numeric values [+1 or -1] based on the 1 v/s rest approach
        train_data_1[train_data_1=='class-1'] = 1
        train_data_1[(train_data_1=='class-2') | (train_data_1=='class-3')] = -1
        train_data_2[train_data_2=='class-2'] = 1
        train_data_2[(train_data_2=='class-1') | (train_data_2=='class-3')] = -1
        train_data_3[train_data_3=='class-3'] = 1
        train_data_3[(train_data_3=='class-1') | (train_data_3=='class-2')] = -1

        return train_data_1, train_data_2, train_data_3

    return clean_train_data, clean_test_data


def label_numeric_converter(y_true):
    """Converting labels to numeric values for multiclass classification
    for calculating accuracies

    Args:
        y_true (numpy array): actual labels from test and train data

    Returns:
        numpy array: labels converted to numeric values
    """
    y_true[y_true=='class-1'] = 0
    y_true[y_true=='class-2'] = 1
    y_true[y_true=='class-3'] = 2
    return y_true


def accuracy_score(y_pred, y_true):
    """Calculates the accuracy score metric

    Args:
        y_pred (numpy array): _description_
        y_true (numpy array): _description_

    Returns:
        float: Accuracy score in percentage
    """
    # return np.sum(y_pred == y_true) / len(y_true)

    # Compare y_pred and y_true, and from the boolean numpy array, 
    # of matching values calculate the mean to get the accuracy score 
    return np.round((np.mean(y_pred == y_true))*100, 2)


def perceptron_train(cleaned_train_data, max_iter, l2_regularisation=False, l2_lambda=0):
    """Perceptron training function for calculating the model parameters

    Args:
        cleaned_train_data (numpy array): train data
        MaxIter (int): Number of training iterations (epochs)
        l2_regularisation (bool, optional): If L2 regularisation to be done. Defaults to False.
        l2_lambda (int, optional): L2 lambda coefficient. Defaults to 0 (if L2 not done).

    Returns:
        model parameters: bias and weights for the trained model
    """
    # initialise weights as a numpy array of length same as input features, initialised to values 0
    weights = np.zeros(cleaned_train_data.shape[1] - 1)
    # Initialies bias to 0
    bias = 0

    # Run 'MaxIter' training iterations 
    for _ in range(max_iter):
        # set random seed for reproducing the results later
        np.random.seed(SEED)
        # Shuffle the order of the data sets randomly for each epoch, for better train results
        np.random.shuffle(cleaned_train_data)

        for data in cleaned_train_data:
            # Separate the input data features for each data point
            input = data[:-1]
            # Get the true label for the data
            y = data[-1]

            # Calculate the activation score by taking the dot product of input and weights and adding the bias
            activation_score = np.dot(weights, input) + bias
            # For every misclassification update the weights and bias
            if y * activation_score <= 0:
                if l2_regularisation:
                    # Calcuate the weights following L2 regularisation for the L2 Lambda coefficient
                    weights = (1-l2_lambda)*weights + y*input
                else:
                    weights = weights + y * input
                bias = bias + y
    return bias, weights


def perceptron_test(model_params, cleaned_test_data, multi_class=False):
    """Predict the labels for each input, based on the activation score

    Args:
        model_params : bias and weights for the model
        cleaned_test_data (numpy array): test data
        multi_class (bool, optional): If multi-class classification or not. 
        If yes set to True. Defaults to False.

    Returns:
        numpy array: Return an array of predicted labels
    """
    y_pred = np.zeros(len(cleaned_test_data))
    bias, weights = model_params
    for data in cleaned_test_data:
        input = data[:-1]

        # Calculate the activation score by taking the dot product of input and weights and adding the bias
        activation_score = np.dot(weights, input) + bias

        # Check if multi-class, then get the activation score
        if multi_class:
            predicted_y = activation_score
        else:
            predicted_y = int(np.sign(activation_score))

        # match the input data with test data, to find the index position 
        # corresponding to test data, and store the predicted values
        y_pred[np.all(cleaned_test_data == data, axis=1)] = predicted_y

    return y_pred


if __name__ == '__main__':

    # load the training data and convert it to numpy array
    train_data = np.array(pd.read_csv('train.data', header=None))
    # load the testing data and convert it to numpy array
    test_data = np.array(pd.read_csv('test.data', header=None))

    # set random seed for reproducing the results later
    np.random.seed(SEED)

    # Shuffle the order of the data sets randomly, for better train results
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    train_data_original = train_data.copy()
    test_data_original = test_data.copy()

    while True:
        user_choice = get_user_choice()

        if user_choice == '1':
            # Get cleaned train and test data for each binary classification
            c1_c2_train_data, c1_c2_test_data = clean_data(train_data, test_data, user_choice, 'class-1', 'class-2')
            c2_c3_train_data, c2_c3_test_data = clean_data(train_data, test_data, user_choice, 'class-2', 'class-3')
            c1_c3_train_data, c1_c3_test_data = clean_data(train_data, test_data, user_choice, 'class-1', 'class-3')

            # Get predicted labels for train data for each binary classification 
            y_pred_c1_c2_train = perceptron_test(perceptron_train(c1_c2_train_data.copy(), MAX_ITER), c1_c2_train_data)
            y_pred_c2_c3_train = perceptron_test(perceptron_train(c2_c3_train_data.copy(), MAX_ITER), c2_c3_train_data)
            y_pred_c1_c3_train = perceptron_test(perceptron_train(c1_c3_train_data.copy(), MAX_ITER), c1_c3_train_data)

            # Get predicted labels for test data for each binary classification 
            y_pred_c1_c2_test = perceptron_test(perceptron_train(c1_c2_train_data.copy(), MAX_ITER), c1_c2_test_data)
            y_pred_c2_c3_test = perceptron_test(perceptron_train(c2_c3_train_data.copy(), MAX_ITER), c2_c3_test_data)
            y_pred_c1_c3_test = perceptron_test(perceptron_train(c1_c3_train_data.copy(), MAX_ITER), c1_c3_test_data)

            # Output accuracy scores for test and train data
            print('Accuracy Score for class-1 v/s class-2 for train data:', accuracy_score(y_pred_c1_c2_train, c1_c2_train_data[:,-1]))
            print('Accuracy Score for class-1 v/s class-2 for test  data:', accuracy_score(y_pred_c1_c2_test, c1_c2_test_data[:,-1]))
            print('Accuracy Score for class-2 v/s class-3 for train data:', accuracy_score(y_pred_c2_c3_train, c2_c3_train_data[:,-1]))
            print('Accuracy Score for class-2 v/s class-3 for test  data:', accuracy_score(y_pred_c2_c3_test, c2_c3_test_data[:,-1]))
            print('Accuracy Score for class-1 v/s class-3 for train data:', accuracy_score(y_pred_c1_c3_train, c1_c3_train_data[:,-1]))
            print('Accuracy Score for class-1 v/s class-3 for test  data:', accuracy_score(y_pred_c1_c3_test, c1_c3_test_data[:,-1]))

        elif user_choice == '2':
            # Get the list of model parameters for each 1 v/s rest model
            train_data_list = clean_data(train_data, test_data, user_choice)

            # Get predictions (activation scores) for train data from each 1 v/s rest model
            y_pred_1_train = perceptron_test(perceptron_train(train_data_list[0].copy(), MAX_ITER), train_data, True)
            y_pred_2_train = perceptron_test(perceptron_train(train_data_list[1].copy(), MAX_ITER), train_data, True)
            y_pred_3_train = perceptron_test(perceptron_train(train_data_list[2].copy(), MAX_ITER), train_data, True)
           
            # Get predictions (activation scores) for test data from each 1 v/s rest model
            y_pred_1_test = perceptron_test(perceptron_train(train_data_list[0].copy(), MAX_ITER), test_data, True)
            y_pred_2_test = perceptron_test(perceptron_train(train_data_list[1].copy(), MAX_ITER), test_data, True)
            y_pred_3_test = perceptron_test(perceptron_train(train_data_list[2].copy(), MAX_ITER), test_data, True)
                
            # Vertically stack each array of predictions from each model.copy()
            y_pred_train_stack = np.vstack((y_pred_1_train, y_pred_2_train, y_pred_3_train)).T
            y_pred_test_stack = np.vstack((y_pred_1_test, y_pred_2_test, y_pred_3_test)).T

            # From the numpy stack find the index of the largest activation score, 
            # to get the corresponding 1 vs rest model and thus the class label
            y_pred_train = np.argmax(y_pred_train_stack, axis=1)
            y_pred_test = np.argmax(y_pred_test_stack, axis=1)
                
            # convert the actual labels to match the argmax output above for each class
            y_true_train = label_numeric_converter(train_data[:,-1].copy())
            y_true_test = label_numeric_converter(test_data[:,-1].copy())

            # Output accuracy scores for test and train data
            print(f'Multiclass Accuracy score for train data:', accuracy_score(y_pred_train, y_true_train))
            print(f'Multiclass Accuracy score for test  data:', accuracy_score(y_pred_test, y_true_test))

        # For L2 regularisation in multiclass
        elif user_choice == '3':
            # Get the list of model parameters for each 1 v/s rest model
            train_data_list = clean_data(train_data, test_data, user_choice)

            # list of L2 Lambda coefficients
            l2_lambda_list = [0.01, 0.1, 1.0, 10.0, 100.0]

            # Find accuracy scores for different L2 Lambda coefficients
            for l2_lambda in l2_lambda_list:

                # Get predictions (activation scores) for train data for each L2 Lambda coefficient
                y_pred_1_train = perceptron_test(perceptron_train(train_data_list[0].copy(), MAX_ITER, True, l2_lambda), train_data, True)
                y_pred_2_train = perceptron_test(perceptron_train(train_data_list[1].copy(), MAX_ITER, True, l2_lambda), train_data, True)
                y_pred_3_train = perceptron_test(perceptron_train(train_data_list[2].copy(), MAX_ITER, True, l2_lambda), train_data, True)

                # Get predictions (activation scores) for test data for each L2 Lambda coefficient
                y_pred_1_test = perceptron_test(perceptron_train(train_data_list[0].copy(), MAX_ITER, True, l2_lambda), test_data, True)
                y_pred_2_test = perceptron_test(perceptron_train(train_data_list[1].copy(), MAX_ITER, True, l2_lambda), test_data, True)
                y_pred_3_test = perceptron_test(perceptron_train(train_data_list[2].copy(), MAX_ITER, True, l2_lambda), test_data, True)

                # Vertically stack each array of predictions from each model
                y_pred_train_stack = np.vstack((y_pred_1_train, y_pred_2_train, y_pred_3_train)).T
                y_pred_test_stack = np.vstack((y_pred_1_test, y_pred_2_test, y_pred_3_test)).T

                # From the numpy stack find the index of the largest activation score, 
                # to get the corresponding 1 vs rest model and thus the class label
                y_pred_train = np.argmax(y_pred_train_stack, axis=1)
                y_pred_test = np.argmax(y_pred_test_stack, axis=1)

                # convert the actual labels to match the argmax output above for each class
                y_true_train = label_numeric_converter(train_data[:,-1].copy())
                y_true_test = label_numeric_converter(test_data[:,-1].copy())

                # Output accuracy scores for test and train data
                print(f'Multiclass Accuracy score for train data for lambda {l2_lambda}:', accuracy_score(y_pred_train, y_true_train))
                print(f'Multiclass Accuracy score for test  data for lambda {l2_lambda}:', accuracy_score(y_pred_test, y_true_test))