import os
from typing import List, Optional, Union
from dataclasses import dataclass
import pickle
from biopandas.pdb import PandasPdb

import pandas as pd
import torch
import dgl

from .calc import read_struct, StructFeats

from .params import (ACIDS_MAP_DEF,
                     ACIDS_MAP_DEF3,
                     SS_MAP_EXT,
                     FEATNAME,
                     NFEATNAME)


_PDBCHAIN_COL = 'pdb_chain'
_SEQUENCE_COL = 'sequence'
_DSSP_COL = 'dssp'
_INVALID_AA: set = {'UNK', '?', 'SEC'}

class GraphObjectError(Exception):
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
                    'distancemx',
                    'residueid']

    def __init__(
            self, 
            code, 
            u, v, 
            efeats, 
            nfeats, 
            sequence, 
            distancemx, 
            residueid=None, 
            chainids=None, 
            **kwargs):
        self.code = code
        self.sequence = sequence
        self.u = u
        self.v = v
        self.nfeats = nfeats
        self.num_nodes = nfeats.shape[0]
        self.efeats = efeats
        self.kwargs = kwargs
        self.distancemx = distancemx
        self.residueid = residueid
        self.chainids = chainids

    @classmethod
    def from_pdb(
        cls, 
        path: Union[str, pd.DataFrame], 
        code: str,
        ca_threshold: float = 7,
        **kwargs) -> "GraphData":
        """
        from pdb file
        """
        if (not isinstance(path, pd.DataFrame)) and (not os.path.isfile(path)):
            raise FileNotFoundError(f'missing .pdb file for: {path}')
        try:
            structdata = read_struct(path, t = ca_threshold)
        except Exception as e:
            raise GraphObjectError(e)
        _seqlen = len(structdata.sequence)
        _nodes = max(structdata.u.max().item(), structdata.v.max().item()) + 1
        if _seqlen != _nodes:
            raise GraphObjectError(f"sequence is not matching Ca-Ca nodes {_seqlen} vs {_nodes}")
        if _INVALID_AA & set(structdata.sequence):
            raise GraphObjectError(f"invalid aa in sequence {set(structdata) - _INVALID_AA}")
        return cls(path=path, code=code, **structdata.asdict())
    
    @classmethod
    def from_h5(cls, path: str, key: str, ca_threshold = 7):
        
        atoms = pd.read_hdf(path, key=key, mode='r')
        structdata = read_struct(atoms, t=ca_threshold)
        return cls(path=path, code=key, **structdata.asdict())

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
            raise GraphObjectError("some CA-CA sequence connections are above given threshold")

    def to_edgedf(self) -> pd.DataFrame:

        feats = self.efeats.numpy()
        if feats.shape[1] != len(self.efeatname):
            raise GraphObjectError(f'number of edge features is diffrent then featnames {feats.shape} and {self.efeatname}')
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
            raise GraphObjectError(f'number of node features is different then featnames {feats.shape} and {self.nfeatname}')
        data = pd.DataFrame(feats, columns=self.nfeatname)
        data['residue'] = self.sequence
        data['resid'] = self.residueid.numpy()
        data['chain_id'] = self.chainids
        return data

    def to_h5(self) -> dict:
        
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