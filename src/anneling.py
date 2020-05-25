#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

def change_features(x):
    g = np.array(x)

    zero_flip_mask = ((g==0)*np.random.uniform(size = g.shape) > (1-1e-5)).astype(int)
    g = g*(1-zero_flip_mask) + (1-g)*zero_flip_mask

    ones_flip_mask = ((g==1)*np.random.uniform(size = g.shape) > (1-0.01)).astype(int)
    g = g*(1-ones_flip_mask) + (1-g)*ones_flip_mask

    return g.tolist()


class DorotheaProblem(Annealer):
    def __init__(self, state, p):
        super(DorotheaProblem, self).__init__(state)  # important!
        self.p = p
    
    #adaptacja stanu
    def move(self):
        self.state = change_features(self.state)

    def energy(self):
        return objective_function(self.state)

def run(initial_state):
    DP = DorotheaProblem(initial_state, 0.1)
    DP.Tmax = 0.01
    DP.Tmin = 0.0001
    DP.updates = 10
    DP.steps = 1000
    array, r = DP.anneal()
    features = __convert_array_to_features__(array)
    
    #run model on a test data
    result = test_score(features)
    
    return result, features

if __name__ == '__main__':
    initial_state = [1 for x in range(100000)]
    result, features = run(initial_state)