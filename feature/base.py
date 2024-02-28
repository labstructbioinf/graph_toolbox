import os
from typing import List
from dataclasses import dataclass
import pickle

import pandas as pd
import torch
import dgl

from .calc import read_struct
from .calc import FEATNAME
from .params import ACIDS_MAP_DEF, ACIDS_MAP_DEF3

_PDBCHAIN_COL = 'pdb_chain'
_SEQUENCE_COL = 'dssp_sequence'


class GraphData:
    __version__ = "0.11"
    metadata: dict = dict()
    feats: torch.Tensor
    nfeats: torch.Tensor
    featname: List[str] = FEATNAME
    nfeatname: List[str] = ['phi', 'psi', 'chi1', 'chi2']
    sequence: List[str]
    ca_threshold: float = 7

    def __init__(self, path, metadata, u, v, feats, nfeats, sequence, **kwargs):
        self.metadata = metadata
        self.path = path
        self.sequence = sequence
        self.u = u
        self.v = v
        self.nfeats = nfeats
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
        u, v, feats, nfeats, struct_sequence = read_struct(path, chain=pdbchain, t = ca_threshold)
        if _SEQUENCE_COL in metadata:
            sequence = metadata[_SEQUENCE_COL]
        else:
            sequence = struct_sequence
        return cls(path=path,
         metadata=metadata,
        u=u,
        v=v,
        feats=feats,
        nfeats=nfeats,
        sequence=sequence)
    
    def to_dgl(self) -> dgl.graph:
        """
        create graph from data
        """
        if len(self.sequence[0]) == 3:
            seqasint = [ACIDS_MAP_DEF3[res] for res in self.sequence]
        elif len(self.sequence[0]) == 1:
            seqasint = [ACIDS_MAP_DEF[res] for res in self.sequence]
        else:
            raise TypeError(f'invalid aa sequence letter: {self.sequence[0]} dictionary should be in one ore three letter code')    
        self.sequence_int = torch.LongTensor(seqasint)
        g = dgl.graph((self.u, self.v))
        g.ndata['f'] = self.sequence_int
        g.edata['f'] = self.feats
        return g

    def to_edgedf(self) -> pd.DataFrame:

        data = pd.DataFrame(self.feats.numpy(), columns=self.featname)
        # source aa residue
        u = [self.sequence[ui] for ui in self.u.tolist()]
        v = [self.sequence[vi] for vi in self.v.tolist()]
        data['u'] = u
        data['v'] = v
        return data
    
    def to_nodedf(self) -> pd.DataFrame:

        feats = self.nfeats.numpy()
        assert feats.ndim == 2, f'node feats invalid shape: {feats.shape}'
        data = pd.DataFrame(feats, columns=self.nfeatname)
        data['residue'] = self.sequence
        return data

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
            'feats': self.feats,
            'nfeats': self.nfeats
            }, path)