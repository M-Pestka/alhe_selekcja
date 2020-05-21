#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from data_reader import Data_reader, get_labels
from feature_provider import Feature_provider
from simanneal import Annealer

import numpy as np
from sklearn.linear_model import LogisticRegression

import random

def file_reader(file):
    config = {}
    config['file_path'] = '../dorothea/' + file
    return Data_reader(config)

train_data = file_reader('dorothea_train.data').get_matrix()
train_labels = get_labels('../dorothea/dorothea_train.labels')

valid_data = file_reader('dorothea_valid.data').get_matrix()
valid_labels = get_labels('../dorothea/dorothea_valid.labels')

feature_train_provider = Feature_provider(train_data)
feature_valid_provider = Feature_provider(valid_data)

def __change_labels__(labels):
    for i, label in enumerate(labels):
        if(label == -1):
            labels[i] = 0


def fitting_score(feature_index_array):
    
    model = LogisticRegression(solver = 'lbfgs')
    
    x_train = feature_train_provider.get_slice(feature_index_array)
    y_train = train_labels
    __change_labels__(y_train)
    model.fit(x_train,y_train)
    
    x_valid = feature_valid_provider.get_slice(feature_index_array)
    y_valid = valid_labels
    __change_labels__(y_valid)
    score = model.score(x_valid, y_valid)
    
    return score

#zmienia tablicę [0,1,1,0,...] na [2,3,...]
def __convert_array_to_features__(x):
    feature_index_array = []
    for i, feature in enumerate(x):
        if(feature > 0):
            feature_index_array.append(i)
    return feature_index_array

#liczy funkcję którą trzeba zminimalizować
def objective_function(x):
    feature_index_array = __convert_array_to_features__(x)
    
    if not feature_index_array:
        return 0
    
    result = fitting_score(feature_index_array)
    
    #negate result so it can be minimized
    result = -result
    
    return result


class DorotheaProblem(Annealer):
    def __init__(self, state, p):
        super(DorotheaProblem, self).__init__(state)  # important!
        self.p = p
    
    #adaptacja stanu
    def move(self):
        for i, feature in enumerate(self.state):
            if(random.uniform(0,1) < self.p):
                self.state[i] = 1 - self.state[i]

    def energy(self):
        return objective_function(self.state)

def run(initial_state):
    DP = DorotheaProblem(initial_state, 0.1)
    DP.Tmax = 0.01
    DP.Tmin = 0.0001
    DP.updates = 10
    DP.steps = 1000
    array, result = DP.anneal()
    features = __convert_array_to_features__(array)
    return -result, features

if __name__ == '__main__':
    initial_state = [1 for x in range(100000)]
    result, features = run(initial_state)