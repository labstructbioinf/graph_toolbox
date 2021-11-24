'''storage for quick function'''
import json
import os
import pickle
import gzip


def get_json(file):
    '''
    read json to object
    '''
    assert isinstance(file, str)
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def save_json(file, data):
    '''
    save json to file
    '''
    assert isinstance(file, str)
    assert isinstance(data, dict)
    with open(file, 'w') as f:
        json.dump(data, f)
    
    
def load_gpickle(file):
    '''
    params:
        file (str) path to pickled object
    return content of gziped (optional requires .gz extension) pickle file
    '''  
    assert isinstance(file, str), f'file must be a valid string'
    assert os.path.isfile(file), f'no such file: {file}'
    if file.endswith('.gz'):
        with gzip.open(file, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(file, 'rb') as f:
            data = pickle.load(f)
    return data

def save_gpickle(obj, file):
    '''
    pickles `obj` if file endswith .gz then zip pickle
    '''
    assert isinstance(file, str)
    assert os.path.isdir(os.path.dirname(file)), f'no such directory: {file}'
    
    if file.endswith('.gz'):
        with gzip.open(file, 'wb') as f:
            pickle.dump(obj, f)
    else:
        with open(file, 'rb') as f:
            pickle.dump(obj, f)
            