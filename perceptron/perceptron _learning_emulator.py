from tabulate import tabulate
import numpy as np
from numpy import array


def main():
    lines = open('data.csv', mode='r')
    class_index = 0
    data_set = []
    desired = []

    # read data from file
    for line in lines:
        d = list(map(float, line.split(',')))
        desired.append(d[class_index])
        data_set.append(array(d[0:class_index] + d[class_index + 1:len(d)]))

    # initialize parameters
    w = np.random.rand(len(data_set[0]))            # weights vector
    eta = .01                                       # learning rate
    threshold = 0                                   # threshold of the activation function
    
    table_header = create_table_header(data_set[0])

    while True:
        i = 0
        output_table = []
        for x in data_set:
            # perform learning equation
            y = activation_func(w, x, threshold)
            error = desired[i] - y
            w_new = w + eta * error * x

            # populate output table at the last itteration
            output_table.append(flatten([x, desired[i], y, error, w, w_new]))
            # next dataset example
            i += 1
            if not np.array_equal(w, w_new):
                w = w_new
        print(tabulate(output_table,
                       headers=table_header))
        input("Press Enter to continue...")


def activation_func(w, x, thresh):
    h = calc_hypothesis(w, x)
    return 1 if h > thresh else -1


def calc_hypothesis(w, x):
    return np.inner(w, x)


def flatten(lis):
    for item in lis:
        try:
            yield from flatten(item)
        except TypeError:
            yield item


def create_table_header(record):
    features = len(record)
    table_header = []
    for i in range(features):
        table_header.append('x' + str(i))

    table_header.append('desired')
    table_header.append('actual')
    table_header.append('error')

    for i in range(features):
        table_header.append('old w' + str(i))

    for i in range(features):
        table_header.append('new w' + str(i))

    return table_header



if __name__ == '__main__': main()
