from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
import time
import numpy
from numpy.random import shuffle
import random

def split_into_portions(arr, large_percent):
    large_sample_size = int(len(arr) * large_percent)
    shuffle(arr)
    return (arr[0:large_sample_size], arr[large_sample_size:])

def get_data_file(filename):
    return genfromtxt(open(filename,'r'), delimiter=';', dtype='float64')[1:]

def split_into_target_and_data(dataset):
    return ([x[0] for x in dataset], [x[1:] for x in dataset])

def get_train_and_test_data(filenames, train_percent):
    training = []
    testing = []
    for filename in filenames:
        train, test = split_into_portions(get_data_file(filename), train_percent)
        training.append(train)
        testing.append(test)
    return (numpy.concatenate(training), numpy.concatenate(testing))

def get_classification_proportion(predicted):
    dic = {}
    for prediction in predicted:
        if prediction in dic:
            dic[prediction] += 1
        else:
            dic[prediction] = 1
    for classification in dic:
        print float(dic[classification]) / len(predicted)

def main():
    #create the training & test sets, skipping the header row with [1:]
    training, testing = get_train_and_test_data(['orientation/sawyer_or_cross.csv',
        'orientation/sawyer_or_other.csv', 'orientation/alanna_cross_or.csv',
        'orientation/alanna_straight_or.csv'], 0.9)
    new_testing = get_data_file('orientation/alanna_other_or.csv')
    testing = numpy.concatenate([testing, new_testing])
    testing_target, testing_data = split_into_target_and_data(testing)
    #create and train the random forest
    train_target, train_data = split_into_target_and_data(training)
    #new_target, new_testing = split_into_target_and_data(get_data_file('alanna.csv'))
    rf = RandomForestClassifier(n_estimators=300, n_jobs=2)
    rf.fit(train_data, train_target)
    res = rf.predict(testing_data)
    total = 0
    for predicted, actual in zip(res, testing_target):
        if predicted == actual:
            total += 1
    print float(total) / len(res)

if __name__=="__main__":
    main()
