'''storage for quick function file save & write operations'''
import json
import os
import pickle
import gzip
import tempfile
from typing import List, Union

import atomium


def get_json(file: str):
    '''
    read json to object
    '''
    assert isinstance(file, str)
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def save_json(data: Union[dict, list], file: str):
    '''
    save json to file
    '''
    assert isinstance(file, str)
    assert isinstance(data, dict)
    with open(file, 'w') as f:
        json.dump(data, f)
    
    
def load_gpickle(file: str):
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

def save_gpickle(obj, file: str):
    '''
    pickles `obj` if file endswith .gz then zip pickle
    '''
    assert isinstance(file, str)
    dirname = os.path.dirname(file)
    if dirname:
        assert os.path.isdir(dirname), f'no such directory: {dirname}'
    
    if file.endswith('.gz'):
        with gzip.open(file, 'wb') as f:
            pickle.dump(obj, f)
    else:
        with open(file, 'wb') as f:
            pickle.dump(obj, f)
            
def unpack_gzip_to_pdb_atomium_obj(gzip_file_path: str):
    '''
    transform gzip pdb file to atomium.Model object.
    return:
        handle
    '''
    if not isinstance(gzip_file_path, str):
        raise TypeError(f'invalid arg type expected: {type(gzip_file_path)}')
    if not os.path.isfile(gzip_file_path):
        raise FileNotFoundError(f'file : {gzip_file_path} doesn\'t exists')
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file_name = os.path.join(tmp_dir, 'tmp.pdb')
        # load gzip file context
        with gzip.open(gzip_file_path, 'rt') as f:
            tmp = f.read()
        # save file context as formatted pdb string
        with open(tmp_file_name, 'wt') as f:
            f.write(tmp)
        # load as atomium object
        handle = atomium.open(tmp_file_name).model
    return handle

        
    
    
        
            