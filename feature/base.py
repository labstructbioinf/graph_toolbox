import os
from typing import List
from dataclasses import dataclass
import pickle

import torch
import dgl

from .calc import read_struct
from .calc import FEATNAME
from .params import ACIDS_MAP_DEF, ACIDS_MAP_DEF3

_PDBCHAIN_COL = 'pdb_chain'
_SEQUENCE_COL = 'dssp_sequence'


class GraphData:
    __version__ = "0.1"
    metadata: dict = dict()
    feats: torch.Tensor
    featname: List[str] = FEATNAME
    sequence: List[str]
    ca_threshold: float = 7

    def __init__(self, path, metadata, u, v, feats, sequence, **kwargs):
        self.metadata = metadata
        self.path = path
        self.sequence = sequence
        self.u = u
        self.v = v
        self.feats = feats
        self.featname
        self.kwargs = kwargs


    @classmethod
    def from_pdb(cls, path: str, metadata: dict = dict(), ca_threshold: float = 7, **kwargs) -> "GraphData":
        """
        metadata columns used
        `pdb_chain`
        """
        assert os.path.isfile(path)
        for key, val in kwargs.items():
            metadata[key] = val
        metadata['ca_threshold'] = float(ca_threshold)
        metadata['path'] = path
        if _PDBCHAIN_COL in metadata:
            pdbchain = metadata[_PDBCHAIN_COL]
        else:
            pdbchain = None
        u, v, feats, struct_sequence = read_struct(path, chain=pdbchain, t = ca_threshold)
        if _SEQUENCE_COL in metadata:
            sequence = metadata[_SEQUENCE_COL]
        else:
            sequence = struct_sequence
        return cls(path, metadata, u, v, feats, sequence)
    
    def to_dgl(self) -> dgl.graph:
        """
        create graph from data
        """
        if len(self.sequence[0]) == 3:
            seqasint = [ACIDS_MAP_DEF3[res] for res in self.sequence]
        elif len(self.sequence[0] == 1):
            seqasint = [ACIDS_MAP_DEF[res] for res in self.sequence]
        else:
            raise TypeError('invalid aa sequence letter: {self.sequence[0]} dictionary should be in one ore three letter code')    
        self.sequence_int = torch.LongTensor(seqasint)
        g = dgl.graph((self.u, self.v))
        g.ndata['f'] = self.sequence_int
        g.edata['f'] = self.feats
        return g

    @classmethod
    def load(cls, path: str) -> "GraphData":
        assert os.path.isfile(path)
        data = torch.load(path)
        return cls(**data)

    def save(self, path: str):

        torch.save(
            {
            '__version__': self.__version__,
            'sequence': self.sequence,
            'metadata': self.metadata,
            'path': self.path,
            'u': self.u,
            'v': self.v,
            'feats': self.feats
            }, path)