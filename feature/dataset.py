import os
import time
from typing import List, Any, Union

import h5py 
import numpy as np
import torch

import dgl
from .base import GraphData


class H5Handle:
    '''
    for graphs
    '''
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
        if not os.path.isfile(filename):
            with h5py.File(filename, "w") as hf:
                pass
        self.filename = filename

# https://github.com/harvardnlp/botnet-detection/blob/master/graph_data_storage.md
    def write_graph(self, g: GraphData):


        group = self.group_from_code(g.code)
        with h5py.File(self.filename, "a") as hf:
            pdbgr = hf.require_group(group)
            pdbgr.attrs['is_valid'] = False
            for key, val in g.to_h5().items():
                pdbgr.create_dataset(name=key, data=val)
            pdbgr.attrs['is_valid'] = True
            
    def read_graph(self, code):
        '''
        if size is none read all record from start to the end
        '''
        with h5py.File(self.filename, 'r') as hf:
            group = self.group_from_code(code)
            pdbsubgr = hf[group]
            try:
                u, v = torch.from_numpy(pdbsubgr['u'][:]), torch.from_numpy(pdbsubgr['v'][:])
                sequence = torch.from_numpy(pdbsubgr['sequence'][:])
                nfeats, efeats = torch.from_numpy(pdbsubgr['nfeats'][:]), torch.from_numpy(pdbsubgr['efeats'][:])
                distancemx = torch.from_numpy(pdbsubgr['distancemx'][:])

                g = dgl.graph((u, v))
                g.ndata['seq'] = sequence
                g.ndata['angles'] = nfeats
                g.edata['f'] = efeats
            except KeyError as e:
                raise KeyError(f'missing {e} for gr {group}')
            return g, distancemx
        
    def read_key(self, code: str, key: str):

        with h5py.File(self.filename, 'r') as hf:
            group = self.group_from_code(code)
            return hf[group][key][:]

    def write_corrupted(self, code):
        with h5py.File(self.filename, 'a') as hf:
            pdbgr = hf.require_group(self.error_group)
            pdbgr.create_dataset(str(code), data=h5py.Empty("f"))

    def group_from_code(self, code):
        """
        locate h5 group based on code
        """
        pdb, _, _ = code.split("_")
        preffix = pdb[:2]
        #group = f"{preffix}/{pdb}/{code}"
        group = f"{preffix}/{pdb}/{code}"
        return group
    
    @property
    def pdbs(self) -> list:
        with h5py.File(self.filename, 'r') as hf:
            pdbs = list(hf.keys())
        return pdbs
    
    @property
    def codes(self) -> list:
        with h5py.File(self.filename, 'r') as hf:
            valid_codes = list()
            pdbs = set(hf.keys())
            for pdb in pdbs:
                hfpdb = hf[pdb]
                codes = hfpdb.keys()
                for code in codes:
                    if hfpdb[code].attrs['is_valid']:
                        valid_codes.append(code)
        return valid_codes
    
    @property
    def invalid(self) -> list:
        with h5py.File(self.filename, 'a') as hf:
            error_grp = hf.require_group(self.error_group)
            return list(error_grp.keys())



class EmbH5Handle:
    reqkeys = {'emb'}
    numkeys = len(reqkeys)
    error_group: str = "errors"
    direct_read: bool = True
    filename: str
    def __init__(self, filename : Union[str, os.PathLike]):
        self.filename = filename
        if not os.path.isfile(self.filename):
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            with h5py.File(self.filename, "w") as hf:
                pass
            
# https://github.com/harvardnlp/botnet-detection/blob/master/graph_data_storage.md
    def write(self, emb, code):
        pdb, chain, hnum = code.split("_")
        with h5py.File(self.filename, "a") as hf:
            pdbgr = hf.require_group(pdb)
            if code not in pdbgr:
                pdbsubgr = pdbgr.require_group(code)
                pdbsubgr.create_dataset('emb', data=emb.numpy())
                
    def read(self, code) -> torch.Tensor:
        pdb, chain, hnum = code.split("_")
        with h5py.File(self.filename, "a") as hf:
            pdbgr = hf.require_group(pdb)
            if code in pdbgr:
                pdbsubgr = pdbgr.require_group(code)
                return torch.from_numpy(pdbsubgr['emb'][:])
            else:
                raise FileNotFoundError(f"missing {code}")
            
    @property
    def codes(self) -> list:
        with h5py.File(self.filename, 'r') as hf:
            valid_codes = list()
            pdbs = set(hf.keys())
            for pdb in pdbs:
                hfpdb = hf[pdb]
                valid_codes += list(hfpdb.keys())
            return valid_codes