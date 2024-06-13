import os
import time
from typing import List, Any, Union

import h5py 
import numpy as np
import torch

import dgl
from .base import GraphData


class H5Handle:
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
    def write_graph(self, g: GraphData):

        pdb, chain, hnum = g.code.split("_")

        with h5py.File(self.filename, "a") as hf:
            pdbgr = hf.require_group(pdb)
            if g.code not in pdbgr:
                pdbsubgr = pdbgr.require_group(g.code)
                for key, val in g.to_h5().items():
                    pdbsubgr.create_dataset(name=key, data=val)

            
    def read_graph(self, code):
        '''
        if size is none read all record from start to the end
        '''

        pdb, chain, hnum = code.split("_")
        with h5py.File(self.filename, 'r') as hf:
            
            pdbsubgr = hf[pdb][code]

            u, v = torch.from_numpy(pdbsubgr['u'][:]), torch.from_numpy(pdbsubgr['v'][:])
            sequence = torch.from_numpy(pdbsubgr['sequence'][:])
            nfeats, efeats = torch.from_numpy(pdbsubgr['nfeats'][:]), torch.from_numpy(pdbsubgr['efeats'][:])
            distancemx = torch.from_numpy(pdbsubgr['distancemx'][:])

            g = dgl.graph((u, v))
            g.ndata['seq'] = sequence
            g.ndata['angles'] = nfeats
            g.edata['f'] = efeats
            return g, distancemx

    @property
    def pdbs(self) -> list:
        with h5py.File(self.filename, 'r') as hf:
            pdbs = list(hf.keys())
        return pdbs
    
    @property
    def codes(self) -> list:
        with h5py.File(self.filename, 'r') as hf:
            codes = list()
            for group in hf:
                codes.extend(list(hf[group].keys()))
        return codes
