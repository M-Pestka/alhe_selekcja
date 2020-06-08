#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:41:45 2020

@author: kristofer
"""

from data_reader import Data_reader, get_labels
from feature_provider import Feature_provider
from simanneal import Annealer

import numpy as np
from sklearn.linear_model import LogisticRegression

def file_reader(file):
    config = {}
    config['file_path'] = '../dorothea/' + file
    return Data_reader(config)

def __change_labels__(labels):
    for i, label in enumerate(labels):
        if(label == -1):
            labels[i] = 0
            
data = file_reader('dorothea_train.data').get_matrix()
labels = get_labels('../dorothea/dorothea_train.labels')

train_data = data[:600, :]
train_labels = labels[:600]
__change_labels__(train_labels)

valid_data = data[600:, :]
valid_labels = labels[600:]
__change_labels__(valid_labels)

test_data = file_reader('dorothea_valid.data').get_matrix()
test_labels = get_labels('../dorothea/dorothea_valid.labels')
__change_labels__(test_labels)

#train_data = file_reader('dorothea_train.data').get_matrix()
#train_labels = get_labels('../dorothea/dorothea_train.labels')
#__change_labels__(train_labels)

#valid_data = file_reader('dorothea_valid.data').get_matrix()
#valid_labels = get_labels('../dorothea/dorothea_valid.labels')
#__change_labels__(valid_labels)

feature_train_provider = Feature_provider(train_data)
feature_valid_provider = Feature_provider(valid_data)
feature_test_provider = Feature_provider(test_data)


def fitting_score(feature_array):
    
    model = LogisticRegression(solver = 'lbfgs')
    
    x_train = feature_train_provider.get_slice(feature_array)
    y_train = train_labels
    model.fit(x_train,y_train)
    
    x_valid = feature_valid_provider.get_slice(feature_array)
    y_valid = valid_labels
    score = model.score(x_valid, y_valid)
    
    return score

def test_score(feature_array):
    
    model = LogisticRegression(solver = 'lbfgs')
    
    x_train = feature_train_provider.get_slice(feature_array)
    y_train = train_labels
    model.fit(x_train,y_train)
    
    x_test = feature_test_provider.get_slice(feature_array)
    y_test = test_labels
    score = model.score(x_test, y_test)
    
    return score

if __name__ == '__main__':
    scores = []
    for feature in range(100000):
        scores.append(fitting_score(feature))
        if(feature % 10000 == 0):
            print("Finished: " + str(feature))
    
    np_scores = np.array(scores)
    sorted_indicies = np.argsort(np_scores)[::-1]
    
    for number_of_features in np.array(range(20))+1:
        score = test_score(sorted_indicies[:number_of_features])
        print("Liczba cech: " + str(number_of_features) + " , wynik: " + str(score))
