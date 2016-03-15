from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
import time
import numpy
import random

def split_into_portions(arr, large_percent):
    large_sample = random.sample(arr, int(len(arr) * large_percent))
    small_sample = [x for x in arr if not numpy.any(
        [y is x for y in large_sample]
    )]
    return (large_sample, small_sample)

def main():
    #create the training & test sets, skipping the header row with [1:]
    cross_dataset = genfromtxt(open('cross.csv','r'), delimiter=';', dtype='float64')[1:]
    other_dataset = genfromtxt(open('other.csv','r'), delimiter=';', dtype='float64')[1:]
    other_train, other_test = split_into_portions(other_dataset, 0.9)
    cross_train, cross_test = split_into_portions(cross_dataset, 0.9)
    train = other_train
    test = other_test
    train.extend(cross_train)
    test.extend(cross_test)
    train_target = [x[0] for x in train]
    test_target = [x[0] for x in test]
    train_data = [x[1:] for x in train]
    test_data = [x[1:] for x in test]
    #create and train the random forest
    rf = RandomForestClassifier(n_estimators=300, n_jobs=2)
    rf.fit(train_data, train_target)
    res = rf.predict(test_data)
    total = 0
    for predicted, actual in zip(res, test_target):
        print actual
        if predicted == actual:
            total += 1
    print total / len(res)

if __name__=="__main__":
    main()
