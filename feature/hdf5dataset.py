import os
import time
from typing import List, Any, Union

import h5py 
import numpy as np
import torch

from .base import GraphData

class HDF5Handle:
    filename: str
    def __init__(self, filename : Union[str, os.PathLike]):
        self.filename = filename

# https://github.com/harvardnlp/botnet-detection/blob/master/graph_data_storage.md
    def write_graph(self, g: GraphData):

        pdb, chain, hnum = g.code.split("_")
        with h5py.File(self.filename, "a") as hf:
            group = hf.require_group(pdb)
            subgroup = group.require_group(g.code)
            for dname, ddata in g.to_hdf5():
                subgroup.add_dataset(dname, data=ddata)

            
    def read_graph(self, code: str) -> dict:
        '''
        if size is none read all record from start to the end
        '''
        pdb, chain, hnum = code.split("_")
        with h5py.File(self.filename, 'r') as hf:
            group = hf[pdb]
            subgroup = group[code]
            graph_dict = {key: subgroup[key][:] for key in GraphData.__savekeys__}

        return graph_dict
             