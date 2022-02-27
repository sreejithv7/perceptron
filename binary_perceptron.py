
import numpy as np
import pandas as pd
import random

SEED = 2

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

    # if label == 'class-1':
    #     return 1
    # elif label == 'class-3':
    #     return -1
    # else:
    #     return 0


def perceptron_train(train_data, MaxIter, multi_class=False):
    np.random.seed(SEED)
    # dataset = new_dataset[:80, :]
    weights = np.zeros(train_data.shape[1] - 1)
    bias = 0

    for _ in range(MaxIter):
        for data in train_data:
            input = data[:-1]
            label = data[-1]
            # print(label)
            # print('input:', type(input))
            # print('output:', type(label))

            a = np.dot(weights, input) + bias
            # print('a:', a)
            y = label_num_converter(label)
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


def perceptron_test(bias, weights, test_data):
  
    np.random.seed(SEED)
    for data in test_data:
        # print('data:',data)
        input = data[:-1]
        label = data[-1]
        # print('weights:',weights)
        # print('bias:',bias)
        a = np.dot(weights, input) + bias
        # print('a:', a)
        predicted_y = np.sign(a)
        print('Predicted label:', predicted_y, 'Actual label:', label)


if __name__ == '__main__':


    np.random.seed(SEED)

    train_data = np.array(pd.read_csv('/Users/sreejith/Dev/projects/uni_assignments/perceptron/CA1data/train.data', header=None))
    # print(train_data)

    test_data = np.array(pd.read_csv('/Users/sreejith/Dev/projects/uni_assignments/perceptron/CA1data/test.data', header=None))
    # print(test_data)

    # clean_train_data = train_data[np.where(train_data[:,4] != 'class-3')]
    # clean_test_data = test_data[np.where(test_data[:,4] != 'class-3')]

    clean_train_data = train_data[np.where(train_data[:,4] != 'class-1')]
    clean_test_data = test_data[np.where(test_data[:,4] != 'class-1')]

    # clean_train_data = train_data[np.where(train_data[:,4] != 'class-2')]
    # clean_test_data = test_data[np.where(test_data[:,4] != 'class-2')]

    np.random.shuffle(clean_train_data)
    np.random.shuffle(clean_test_data)

    # print('iteration 1:', perceptron_train(train_data, 1))
    # print('iteration 2:', perceptron_train(train_data, 2))
    # print('iteration 3:', perceptron_train(train_data, 3))
    # print('iteration 4:', perceptron_train(train_data, 4))
    # print('iteration 20:', perceptron_train(train_data, 20))

    bias, weights = perceptron_train(clean_train_data, 20)

    print(bias, weights)

    perceptron_test(bias, weights, clean_test_data)