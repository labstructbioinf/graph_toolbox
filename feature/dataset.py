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
    reqkeys = {'u', 'v', 'nfeats', 'efeats', 'sequence', 'distancemx'}
    numkeys = len(reqkeys)
    error_group: str = "errors"
    direct_read: bool = True
    filename: str
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
        
    def write_corrupted(self, code):
        with h5py.File(self.filename, 'a') as hf:
            pdbgr = hf.require_group(self.error_group)
            pdbgr.create_dataset(str(code), data=h5py.Empty("f"))

    @property
    def pdbs(self) -> list:
        with h5py.File(self.filename, 'r') as hf:
            pdbs = list(hf.keys())
        return pdbs
    
    @property
    def codes(self) -> list:
        with h5py.File(self.filename, 'r') as hf:
            valid_codes = list()
            pdbs = hf.keys()
            
            for pdb in pdbs:
                hfpdb = hf[pdb]
                codes = hfpdb.keys()
                for code in codes:
                    hfcode = hfpdb[code]
                    if len(self.reqkeys & hfcode.keys()) == self.reqkeys:
                        valid_codes.append(code)
        return valid_codes
    @property
    def invalid(self) -> list:
        with h5py.File(self.filename, 'a') as hf:
            error_grp = hf.require_group(self.error_group)
            return list(error_grp.keys())
