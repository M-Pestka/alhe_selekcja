import numpy as np


class Feature_provider:
    def __init__(self, data):
        self.data = data
        
    def get_slice(self, features, sparse_output = True):
        if(sparse_output):
            return self.data[:, features]
        
        else:
            return self.data[:, features].toarray()
        
    
    def _get_col(self, f):
        if(sparse):
            return self.data.getcol(f)

        return self.data.getcol(f).toarray()
        
