from distutils.command.clean import clean
import numpy as np
import pandas as pd
import random

SEED = 27

def get_user_choice():
    while True:
        try:
            print('Please enter your choice from either 1 - 6: \n')
            choice = input('''    1. Classify class-1 and class-2
    2. Classify class-2 and class-3
    3. Classify class-1 and class-3
    4. Classify class-1 v/s class-2 v/s class-3
    5. Add l2 regularisation to the multi-class classifier
    6. Quit\n\n''')
            if choice in '123456':
                if choice == '6':
                    exit()
                return choice
            else:
                raise ValueError
        except ValueError:
            print(f"Your choice '{choice}' is invalid\n")


def clean_data(train_data, test_data, choice):
    
    if choice == '1':
        # clean_data = np.delete(train_data, np.where(train_data[:,4]=='class-3'), axis=0)
        clean_train_data = train_data[np.where(train_data[:,4] != 'class-3')]
        clean_test_data = test_data[np.where(test_data[:,4] != 'class-3')]

        clean_train_data[clean_train_data=='class-1'] = 1
        clean_train_data[clean_train_data=='class-2'] = -1

        clean_test_data[clean_test_data=='class-1'] = 1
        clean_test_data[clean_test_data=='class-2'] = -1

        # print(clean_train_data)

    elif choice == '2':
        clean_train_data = train_data[np.where(train_data[:,4] != 'class-1')]
        clean_test_data = test_data[np.where(test_data[:,4] != 'class-1')]

        clean_train_data[clean_train_data=='class-2'] = 1
        clean_train_data[clean_train_data=='class-3'] = -1

        clean_test_data[clean_test_data=='class-2'] = 1
        clean_test_data[clean_test_data=='class-3'] = -1

        # print(clean_train_data)

    elif choice == '3':
        clean_train_data = train_data[np.where(train_data[:,4] != 'class-2')]
        clean_test_data = test_data[np.where(test_data[:,4] != 'class-2')]

        clean_train_data[clean_train_data=='class-1'] = 1
        clean_train_data[clean_train_data=='class-3'] = -1

        clean_test_data[clean_test_data=='class-1'] = 1
        clean_test_data[clean_test_data=='class-3'] = -1

        # print(clean_train_data)

    elif choice == '4':

        train_data_1 = train_data.copy()
        train_data_2 = train_data.copy()
        train_data_3 = train_data.copy()

        train_data_1[train_data_1=='class-1'] = 1
        train_data_1[(train_data_1=='class-2') | (train_data_1=='class-3')] = -1
        train_data_2[train_data_2=='class-2'] = 1
        train_data_2[(train_data_2=='class-1') | (train_data_2=='class-3')] = -1
        train_data_3[train_data_3=='class-3'] = 1
        train_data_3[(train_data_3=='class-1') | (train_data_3=='class-2')] = -1

        return train_data_1, train_data_2, train_data_3

    return clean_train_data, clean_test_data


def classification_type(label_list):
    pass


def label_num_converter(label):


    # if label == 'class-1':
    #     return 1
    # elif label == 'class-2':
    #     return -1
    # else:
    #     return 0

    if label == 'class-2':
        return 1
    elif label == 'class-3':
        return -1
    else:
        return 0

    if label == 'class-1':
        return 1
    elif label == 'class-3':
        return -1
    else:
        return 0


def accuracy_score(y_pred, y_true):
    # return np.sum(y_pred == y_true) / len(y_pred)
    return np.mean(y_pred == y_true)


def perceptron_train(tr_data, MaxIter):
    np.random.seed(SEED)
    # dataset = new_dataset[:80, :]
    weights = np.zeros(tr_data.shape[1] - 1)
    bias = 0

    for _ in range(MaxIter):
        for data in tr_data:
            input = data[:-1]
            label = data[-1]
            # print(label)
            # print('input:', type(input))
            # print('output:', type(label))

            a = np.dot(weights, input) + bias
            # print('a:', a)
            y = label
            # print('y:', y)
            # print('y * a :', y * a)
            if y * a <= 0:
                # print('weights: ', weights)
                # print('input:', input)
                weights = weights + y * input
                # print('updated weights:', weights)
                bias = bias + y
                # print('bias:', bias)
    return bias, weights


def perceptron_test(model_params, te_data, multi_class=False):
    
    y_pred = np.zeros(len(te_data))
    bias, weights = model_params
    np.random.seed(SEED)
    for data in te_data:
        # print('data:',data)
        input = data[:-1]
        label = data[-1]
        # print('weights:',weights)
        # print('bias:',bias)
        a = np.dot(weights, input) + bias
        # print('a:', a)
        if multi_class:
            predicted_y = a
        else:
            predicted_y = int(np.sign(a))
        print('Predicted label:', predicted_y, 'Actual label:', label)
        y_pred[np.all(te_data == data, axis=1)] = predicted_y
        # print(np.all(te_data == data, axis=1))
        # print(te_data == data)

    print(accuracy_score(y_pred, te_data[:, -1]))
    return y_pred


if __name__ == '__main__':

    np.random.seed(SEED)

    train_data = np.array(pd.read_csv('/Users/sreejith/Dev/projects/uni_assignments/perceptron/CA1data/train.data', header=None))
    # print(train_data)

    test_data = np.array(pd.read_csv('/Users/sreejith/Dev/projects/uni_assignments/perceptron/CA1data/test.data', header=None))
    # print(test_data)

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    user_choice = get_user_choice()
    print('user choice', user_choice)

    if user_choice == '1' or user_choice == '2' or user_choice == '3':
        new_train_data, new_test_data = clean_data(train_data, test_data, user_choice)
        bias, weights = perceptron_train(new_train_data, 20)
        print(bias , weights)
        perceptron_test((bias, weights), new_test_data)

    elif user_choice == '4':
        train_data_tuple = clean_data(train_data, test_data, user_choice)
        bias_1, weights_1 = perceptron_train(train_data_tuple[0], 20)
        bias_2, weights_2 = perceptron_train(train_data_tuple[1], 20)        
        bias_3, weights_3 = perceptron_train(train_data_tuple[2], 20)        

        model_parameters = {'model_1': (bias_1, weights_1),
                            'model_2': (bias_2, weights_2),
                            'model_3': (bias_3, weights_3)}

        y_pred_1 = perceptron_test(model_parameters['model_1'], test_data)
        y_pred_2 = perceptron_test(model_parameters['model_2'], test_data)
        y_pred_3 = perceptron_test(model_parameters['model_3'], test_data)

        y_pred_stack = np.vstack((y_pred_1, y_pred_2, y_pred_3)).T
        print(y_pred_stack)
        print(y_pred_stack.shape)


    # print('iteration 1:', perceptron_train(train_data, 1))
    # print('iteration 4:', perceptron_train(train_data, 4))
    # print('iteration 100:', perceptron_train(train_data, 20))

  

    

