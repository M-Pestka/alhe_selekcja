import numpy as np
import scipy 

class Data_reader:
    def __init__(self, config):
        self._path = config['file_path']
        self._data = None
        self._read()

    def _read(self):
        '''
        method for loading the dataset into memory
        '''
        rows = []
        columns = []
        data = []

        with open(self._path, 'r') as file:
            for i, line in enumerate(file):
                for feature_number in line.strip().split(' '):
                        rows.append(i)
                        columns.append(int(feature_number))
                        data.append(True)

            num_rows = len(rows)
            num_features = max(columns)
            data = scipy.sparse.csr_matrix((data, (rows, columns)), dtype = np.bool)

            self._data = data


    def get_matrix(self):
        return self._data

