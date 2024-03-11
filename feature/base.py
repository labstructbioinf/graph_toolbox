import os
from typing import List
from dataclasses import dataclass
import pickle

import pandas as pd
import torch
import dgl

from .calc import read_struct
from .params import (ACIDS_MAP_DEF,
                     ACIDS_MAP_DEF3,
                     SS_MAP_EXT,
                     FEATNAME,
                     NFEATNAME)

_PDBCHAIN_COL = 'pdb_chain'
_SEQUENCE_COL = 'sequence'
_DSSP_COL = 'dssp'


class GraphObjectEerror(Exception):
    pass

class GraphData:
    __version__ = "0.13"
    metadata: dict = dict()
    feats: torch.Tensor
    nfeats: torch.Tensor
    featname: List[str] = FEATNAME
    nfeatname: List[str] = NFEATNAME
    distancemx: torch.Tensor
    sequence: List[str]
    dssp: List[str]
    ca_threshold: float = 7
    __savekeys__ = ['metadata', 
                    'u',
                    'v',
                    'feats', 
                    'nfeats', 
                    'featname', 
                    'nfeatname',
                    'sequence', 
                    'dssp',
                    'distancemx']

    def __init__(self, metadata, u, v, feats, nfeats, sequence, dssp, distancemx, **kwargs):
        self.metadata = metadata
        self.sequence = sequence
        self.u = u
        self.v = v
        self.nfeats = nfeats
        self.feats = feats
        self.featname
        self.kwargs = kwargs
        self.distancemx = distancemx
        self.dssp = dssp


    @classmethod
    def from_pdb(cls, path: str, metadata: dict = dict(), ca_threshold: float = 7, **kwargs) -> "GraphData":
        """
        metadata columns used
        `pdb_chain`, `sequence`, `dssp`
        """
        assert os.path.isfile(path)
        for key, val in kwargs.items():
            metadata[key] = val
        metadata['ca_threshold'] = float(ca_threshold)
        metadata['path'] = path
        pdbchain = metadata.get(_PDBCHAIN_COL, None)
        sequence = metadata.get(_SEQUENCE_COL, None)
        dssp = metadata.get(_DSSP_COL, None)
        
        u, v, feats, nfeats, struct_sequence, distancemx = read_struct(path, chain=pdbchain, t = ca_threshold)
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
        sequence=sequence,
        dssp=dssp,
        distancemx=distancemx)
    
    def to_dgl(self, validate: bool = False, with_dist: bool = False) -> dgl.graph:
        """
        create graph from data
        node schema: `seq` (1, ) long, `dssp` (1, long), `angles` (4, ) float
        edge schema `f` (11, ) bool
        Args:
            validate: (bool) if True validate against ca-ca discon
        Returns:
            dgl.graph
        """
        if validate:
            self.validate_ca_gaps()
        if len(self.sequence[0]) == 3:
            seqasint = [ACIDS_MAP_DEF3[res] for res in self.sequence]
        elif len(self.sequence[0]) == 1:
            seqasint = [ACIDS_MAP_DEF[res] for res in self.sequence]
        else:
            raise TypeError(f'invalid aa sequence letter: {self.sequence[0]} dictionary should be in one ore three letter code')
        # dssp 
        dsspasint = [SS_MAP_EXT[letter] for letter in self.dssp]    
        seqasint = torch.LongTensor(seqasint)
        dsspasint = torch.LongTensor(dsspasint)

        g = dgl.graph((self.u, self.v))
        g.ndata['seq'] = seqasint
        g.ndata['dssp'] = dsspasint
        g.ndata['angles'] = self.nfeats
        g.edata['f'] = self.feats
        if with_dist:
            return g, self.distancemx
        else:
            return g

    def validate_ca_gaps(self):
        """
        find gaps in Ca-Ca sequential connections
        """
        featid = self.featname.index('self')
        breakpoint()
        feat = self.feats[:, featid].sum()
        if feat <= self.feats.shape[0]:
            print(f"feat: {feat} is below threshold {self.feats.shape[0]}, path: {self.path}") 
            raise GraphObjectEerror("some CA-CA sequence connections are above given threshold")

    def to_edgedf(self) -> pd.DataFrame:

        feats = self.feats.numpy()
        if feats.shape[1] != len(self.featname):
            raise GraphObjectEerror(f'number of edge features is diffrent then featnames {feats.shape} and {self.featname}')
        data = pd.DataFrame(feats, columns=self.featname)
        # source aa residue
        u = [self.sequence[ui] for ui in self.u.tolist()]
        v = [self.sequence[vi] for vi in self.v.tolist()]
        data['u'] = u
        data['v'] = v
        data['u_resid'] = self.u.tolist()
        data['v_resid'] = self.v.tolist()
        return data
    
    def to_nodedf(self) -> pd.DataFrame:

        feats = self.nfeats.numpy()
        if feats.shape[1] != len(self.nfeatname):
            raise GraphObjectEerror(f'number of node features is different then featnames {feats.shape} and {self.nfeatname}')
        data = pd.DataFrame(feats, columns=self.nfeatname)
        data['residue'] = self.sequence
        return data

    @classmethod
    def load(cls, path: str) -> "GraphData":
        assert os.path.isfile(path)
        data = torch.load(path)
        return cls(**{k: data[k] for k in cls.__savekeys__})

    def save(self, path: str):

        torch.save(
            {
            '__version__': self.__version__,
            **{key: getattr(self, key) for key in self.__savekeys__}
            }, path)