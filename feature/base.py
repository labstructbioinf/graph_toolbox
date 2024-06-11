import os
from typing import List, Optional
from dataclasses import dataclass
import pickle

import pandas as pd
import torch
import dgl

from .calc import read_struct, strucfeats

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
    __version__ = "0.14hdf"
    metadata: dict = dict()
    efeats: torch.Tensor
    nfeats: torch.Tensor
    efeatname: List[str] = FEATNAME
    nfeatname: List[str] = NFEATNAME
    distancemx: torch.Tensor
    sequence: List[str]
    dssp: List[str]
    ca_threshold: float = 7
    __savekeys__ = ['code', 
                    'u',
                    'v',
                    'efeats', 
                    'nfeats', 
                    'sequence', 
                    'distancemx']

    def __init__(self, metadata, u, v, efeats, nfeats, sequence, dssp, distancemx, **kwargs):
        self.metadata = metadata
        self.sequence = sequence
        self.u = u
        self.v = v
        self.nfeats = nfeats
        self.efeats = efeats
        self.kwargs = kwargs
        self.distancemx = distancemx
        self.dssp = dssp


    @classmethod
    def from_pdb(cls, path: str, 
                 pdbchain: Optional[str] = None,
                 metadata: dict = dict(),
                 ca_threshold: float = 7,
                 **kwargs) -> "GraphData":
        """
        metadata columns used
        `pdb_chain`, `sequence`, `dssp`
        """
        assert os.path.isfile(path)
        for key, val in kwargs.items():
            metadata[key] = val
        metadata['ca_threshold'] = float(ca_threshold)
        metadata['path'] = path
        metadata['pdbchain'] = pdbchain
        
        structdata = read_struct(path, chain=pdbchain, t = ca_threshold)
        return cls(path=path, metadata=metadata, **structdata._asdict(), dssp="")
    
    def to_dgl(self, validate: bool = False,
                with_dist: bool = False,
                with_dssp: bool = False) -> dgl.graph:
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
        # dssp to one letter code
        if with_dssp:
            dsspasint = [letter[:1] for letter in self.dssp]
            dsspasint = [SS_MAP_EXT[letter] for letter in self.dssp]    
            dsspasint = torch.LongTensor(dsspasint)
        seqasint = torch.LongTensor(seqasint)
        

        g = dgl.graph((self.u, self.v))
        g.ndata['seq'] = seqasint
        if with_dssp:
            g.ndata['dssp'] = dsspasint
        g.ndata['angles'] = self.nfeats
        g.edata['f'] = self.efeats
        if with_dist:
            return g, self.distancemx
        else:
            return g
        
    def to_dgl_angles(self, validate: bool = False,
                with_dist: bool = False) -> dgl.graph:
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

        seqasint = torch.LongTensor(seqasint)

        g = dgl.graph((self.u, self.v))
        g.ndata['seq'] = seqasint
        g.ndata['angles'] = self.nfeats
        g.edata['f'] = self.efeats
        if with_dist:
            return g, self.distancemx
        else:
            return g

    def validate_ca_gaps(self):
        """
        find gaps in Ca-Ca sequential connections
        """
        featid = self.efeatname.index('self')
        breakpoint()
        feat = self.efeats[:, featid].sum()
        if feat <= self.efeats.shape[0]:
            print(f"feat: {feat} is below threshold {self.efeats.shape[0]}, path: {self.path}") 
            raise GraphObjectEerror("some CA-CA sequence connections are above given threshold")

    def to_edgedf(self) -> pd.DataFrame:

        feats = self.efeats.numpy()
        if feats.shape[1] != len(self.efeatname):
            raise GraphObjectEerror(f'number of edge features is diffrent then featnames {feats.shape} and {self.efeatname}')
        data = pd.DataFrame(feats, columns=self.efeatname)
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

    def to_hdf5(self) -> dict:
        
        if len(self.sequence[0]) == 3:
            seqasint = [ACIDS_MAP_DEF3[res] for res in self.sequence]
        elif len(self.sequence[0]) == 1:
            seqasint = [ACIDS_MAP_DEF[res] for res in self.sequence]
        else:
            raise TypeError(f'invalid aa sequence letter: {self.sequence[0]} dictionary should be in one ore three letter code')
        seqasint = torch.LongTensor(seqasint)

        return {'u': self.u.numpy(),
                'v': self.v.numpy(),
                'efeats': self.efeats.numpy(),
                'nfeats': self.nfeats.numpy(),
                'distancemx': self.distancemx.numpy(),
                'sequence': seqasint.numpy()}


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