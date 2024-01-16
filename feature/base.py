import os
from typing import List
from dataclasses import dataclass

import torch
import dgl

from .calc import read_struct
from .calc import FEATNAME
from .params import ACIDS_MAP_DEF


class GraphData:
    metadata: dict = dict()
    feats: torch.Tensor
    featname: List[str] = FEATNAME
    sequence: List[str]
    sequence_int: torch.LongTensor
    ca_threshold: float = 7

    def __init__(self, path, metadata):
        self.metadata = metadata
        self.u, self.v, self.feats, sequence = read_struct(path, chain=None, t = self.ca_threshold)
        assert self.feats.shape[1] == len(self.featname)
        seqasint = [ACIDS_MAP_DEF[res] for res in sequence]
        self.sequence_int = torch.LongTensor(seqasint)


    @classmethod
    def from_pdb(cls, path: str, metadata: dict = dict(), **kwargs) -> "GraphData":
        """
        do all
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


        
