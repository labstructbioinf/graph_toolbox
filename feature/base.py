import os
from typing import List
from dataclasses import dataclass
import pickle

import torch
import dgl

from .calc import read_struct
from .calc import FEATNAME
from .params import ACIDS_MAP_DEF

_PDBCHAIN_COL = 'pdb_chain'
_SEQUENCE_COL = 'dssp_sequence'

class GraphData:
    metadata: dict = dict()
    feats: torch.Tensor
    featname: List[str] = FEATNAME
    sequence: List[str]
    sequence_int: torch.LongTensor
    ca_threshold: float = 7

    def __init__(self, path, metadata):
        self.metadata = metadata
        # check pdb chain
        if _PDBCHAIN_COL in metadata:
            pdbchain = metadata[_PDBCHAIN_COL]
        else:
            pdbchain = None
        self.u, self.v, self.feats, sequence = read_struct(path, chain=pdbchain, t = self.ca_threshold)
        if _SEQUENCE_COL in metadata:
            sequence = metadata[_SEQUENCE_COL]
        assert self.feats.shape[1] == len(self.featname)
        seqasint = [ACIDS_MAP_DEF[res] for res in sequence]
        self.sequence_int = torch.LongTensor(seqasint)


    @classmethod
    def from_pdb(cls, path: str, metadata: dict = dict(), **kwargs) -> "GraphData":
        """
        metadata columns used
        `pdb_chain`
        """
        assert os.path.isfile(path)
        for key, val in kwargs.items():
            metadata[key] = val
        metadata['path'] = path
        obj = cls(path, metadata)
        return obj
    
    def to_dgl(self) -> dgl.graph:
        """
        create graph from data
        """
        g = dgl.graph((self.u, self.v))
        g.ndata['f'] = self.sequence_int
        g.edata['f'] = self.feats
        return g

    @staticmethod
    def from_pickle(path: str) -> "GraphData":
        assert os.path.isfile(path)
        with open(path, 'rb') as fp:
            return pickle.load(fp)

        
