import numpy as np


class Feature_provider:
    def __init__(self, data):
        self.data = data
        
    def get_slice(self, features):
        
        if(isinstance(features, int)):
            return self._get_col(features)
        
        arr = []
        for f in features:
            arr.append(self.data.getcol(f).toarray())
            
        return np.concatenate(arr, axis = -1)
    
    def _get_col(self, f):
        return self.data.getcol(f).toarray()
        
