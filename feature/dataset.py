import os
import time
from typing import List, Any, Union

import h5py 
import numpy as np
import torch

from base import GraphData

class HDF5Handle:
    dataset_attrs = {
        'compression' : 'lzf',
        'dtype': np.float32
        }
    groupname: str = 'embeddings'
    preffix: str = 'emb_'
    direct_read: bool = True
    filename: str
    wait_time: float = 0.1
    def __init__(self, filename : Union[str, os.PathLike]):
        self.filename = filename
# https://github.com/harvardnlp/botnet-detection/blob/master/graph_data_storage.md
    def write_graph(self, g: GraphData, groupname: str):

        with h5py.File(self.filename, "a") as hf:
            emb_group = hf.require_group(self.groupname)
            for index, emb in batch_iter:
                if isinstance(emb, torch.FloatTensor):
                    emb = emb.numpy()
                emb_group.create_dataset(
                    name=f'{self.preffix}{index}',
                      shape=emb.shape, data=emb,
                        **self.dataset_attrs)

    def read_batch(self, start, size = None) -> List[np.ndarray]:
        '''
        if size is none read all record from start to the end
        '''
        emb_list = list()
        with h5py.File(self.filename, 'r') as hf:
            if not 'embeddings' in hf.keys():
                raise KeyError('missing embedding group, probably the file is empty')
            else:
                emb_group = hf['embeddings']
            if size is None:
                size = len(emb_group.keys())
            if start > size:
                raise ValueError(f'start >= then dataset size {start} >= {size}')
            for index in range(start, start+size):
                dataset_name = f'{self.preffix}{index}'
                emb_list.append(emb_group[dataset_name][:])
        return emb_list