import os
from typing import List, Optional, Union
from dataclasses import dataclass
import pickle
from biopandas.pdb import PandasPdb

import numpy as np
import pandas as pd
import torch
import dgl

from graph_toolbox.feature.cc_specific import helix_indices_from_series

from .calc import read_struct, StructFeats
from .cc_calc import read_struct_cc, StructFeatsCC
from .params import ACIDS_MAP_DEF, ACIDS_MAP_DEF3, SS_MAP_EXT, FEATNAME, NFEATNAME


_PDBCHAIN_COL = "pdb_chain"
_SEQUENCE_COL = "sequence"
_DSSP_COL = "dssp"
_INVALID_AA: set = {"UNK", "?", "SEC"}


class GraphObjectError(Exception):
    pass


class GraphData:
    __version__ = "0.17_spectral_edges"
    metadata: dict = {}
    efeats: torch.Tensor
    nfeats: torch.Tensor
    efeatname: List[str]
    nfeatname: List[str] = NFEATNAME
    distancemx: torch.Tensor
    sequence: List[str]
    dssp: List[str]
    ca_threshold: float = 7
    __savekeys__ = [
        "code",
        "u",
        "v",
        "efeats",
        "nfeats",
        "sequence",
        "distancemx",
        "backbone_relrot",
        "sidechain_relrot",
        "residueid",
        "segment_id",
    ]

    def __init__(
        self,
        code,
        u,
        v,
        efeats,
        nfeats,
        sequence,
        distancemx,
        backbone_relrot=None,
        sidechain_relrot=None,
        residueid=None,
        chainids=None,
        with_interactions=True,
        segment_id=None,
        **kwargs,
    ):
        self.code = code
        self.sequence = sequence
        self.u = u
        self.v = v
        self.nfeats = nfeats
        self.efeatname = StructFeats.edge_features(with_interactions=with_interactions)
        self.num_nodes = nfeats.shape[0]
        self.backbone_relrot = backbone_relrot
        self.sidechain_relrot = sidechain_relrot
        self.efeats = efeats
        self.kwargs = kwargs
        self.distancemx = distancemx
        self.residueid = residueid
        self.chainids = chainids
        self.segment_id = segment_id

    @classmethod
    def from_pdb(
        cls,
        path: Union[str, pd.DataFrame],
        code: str,
        ca_threshold: float = 7,
        **kwargs,
    ) -> "GraphData":
        """
        from pdb file
        """
        if (not isinstance(path, pd.DataFrame)) and (not os.path.isfile(path)):
            raise FileNotFoundError(f"missing .pdb file for: {path}")
        try:
            structdata = read_struct(path, t=ca_threshold)
        except Exception as e:
            raise GraphObjectError(e)
        _seqlen = len(structdata.sequence)
        _nodes = max(structdata.u.max().item(), structdata.v.max().item()) + 1
        if _seqlen != _nodes:
            raise GraphObjectError(
                f"sequence is not matching Ca-Ca nodes {_seqlen} vs {_nodes}"
            )
        if _INVALID_AA & set(structdata.sequence):
            raise GraphObjectError(
                f"invalid aa in sequence {set(structdata) - _INVALID_AA}"
            )
        return cls(path=path, code=code, **structdata.asdict())

    @classmethod
    def from_h5(
        cls, path: str, key: str, ca_threshold=7, dssp: list[str] = None, with_interactions: bool = True, with_relative_rotations: bool = False
    ) -> "GraphData":
        """
        Args:
            path (str): path to .h5 file
            key (str): key within .h5 file pointing to desired protein
            ca_threshold (float): ca-ca A threshold
            with_interactions (bool): If true attaches residue-residue interactions to edge featrues
        """
        atoms = pd.read_hdf(path, key=key, mode="r")
        structdata = read_struct(
            atoms, t=ca_threshold, with_interactions=with_interactions, with_relative_rotations=with_relative_rotations
        )
        return cls(code=key, **structdata.asdict())

    @classmethod
    def from_h5_contrastive(cls, path: str, key: str, env_metadata, ca_threshold=7) -> "GraphData":
        """
        Args:
            path (str): path to .h5 file with coordinates
            key (str): key within .h5 file pointing to desired protein
            ca_threshold (float): ca-ca A threshold
            env_metadata: pd.Series with values `env_index`, `chain_id` and `pdb_index` used to determine which helices are in the sample for contrastive learning
        """
        atoms = pd.read_hdf(path, key=key, mode="r")
        helix_indices = helix_indices_from_series(env_metadata)
        helix_indices = torch.LongTensor(helix_indices)
        structdata = read_struct_cc(
            atoms, t=ca_threshold, with_interactions=False, helix_indices=helix_indices
        )
        return cls(code=key, **structdata.asdict())

    def to_dgl(
        self, validate: bool = False, with_dist: bool = False, with_dssp: bool = False, with_relative_rotations: bool = False
    ) -> dgl.graph:
        """
        create graph from data
        node schema: `seq` (1, ) long, `dssp` (1, long), `angles` (4, ) float
        edge schema `f` (11, ) bool
        Args:
            validate: (bool) if True validate against ca-ca discon
        Returns:
            dgl.graph
        """
        if with_relative_rotations and not with_dist:
            raise ValueError("with_dist and with_relrot are mutually exclusive")
        if validate:
            self.validate_ca_gaps()
        if len(self.sequence[0]) == 3:
            seqasint = [ACIDS_MAP_DEF3[res] for res in self.sequence]
        elif len(self.sequence[0]) == 1:
            seqasint = [ACIDS_MAP_DEF[res] for res in self.sequence]
        else:
            raise TypeError(
                f"invalid aa sequence letter: {self.sequence[0]} dictionary should be in one ore three letter code"
            )
        # dssp
        # dssp to one letter code
        if with_dssp:
            dsspasint = [letter[:1] for letter in self.dssp]
            dsspasint = [SS_MAP_EXT[letter] for letter in self.dssp]
            dsspasint = torch.LongTensor(dsspasint)
        seqasint = torch.LongTensor(seqasint)

        g = dgl.graph((self.u, self.v))
        g.ndata["seq"] = seqasint
        if with_dssp:
            g.ndata["dssp"] = dsspasint
        g.ndata["angles"] = self.nfeats
        g.edata["f"] = self.efeats
        if with_dist and with_relative_rotations:
            return g, self.distancemx, self.backbone_relrot, self.sidechain_relrot
        elif with_dist:
            return g, self.distancemx
        else:
            return g


    def to_edgedf(self) -> pd.DataFrame:

        feats = self.efeats.numpy()
        if feats.shape[1] != len(self.efeatname):
            raise GraphObjectError(
                f"number of edge features is diffrent then featnames {feats.shape} and {self.efeatname}"
            )
        data = pd.DataFrame(feats, columns=self.efeatname)
        # source aa residue
        u = [self.sequence[ui] for ui in self.u.tolist()]
        v = [self.sequence[vi] for vi in self.v.tolist()]
        data["u"] = u
        data["v"] = v
        data["u_resid"] = self.u.tolist()
        data["v_resid"] = self.v.tolist()
        return data

    def to_h5(self) -> dict:

        if len(self.sequence[0]) == 3:
            seqasint = [ACIDS_MAP_DEF3[res] for res in self.sequence]
        elif len(self.sequence[0]) == 1:
            seqasint = [ACIDS_MAP_DEF[res] for res in self.sequence]
        else:
            raise TypeError(
                f"invalid aa sequence letter: {self.sequence[0]} dictionary should be in one ore three letter code"
            )
        seqasint = torch.LongTensor(seqasint)

        N = self.distancemx.shape[0]
        d  = self.distancemx.numpy()                           # [N, N, 1 or 2]
        br = self.backbone_relrot.numpy().reshape(N, N, -1)    # [N, N, 9]
        sr = self.sidechain_relrot.numpy().reshape(N, N, -1)   # [N, N, 9]
        target = np.concatenate([d, br, sr], axis=-1).astype(np.float32)  # [N, N, 19]

        return {
            "u":           self.u.numpy(),
            "v":           self.v.numpy(),
            "efeats_flat": self.efeats.numpy().astype(np.float32),  # [E, D]
            "nfeats":      self.nfeats.numpy(),
            "sequence":    seqasint.numpy(),
            "segment_id":  self.segment_id.numpy(),
            "target":      target,
        }

    @classmethod
    def load(cls, path: str) -> "GraphData":
        assert os.path.isfile(path)
        data = torch.load(path)
        return cls(**{k: data[k] for k in cls.__savekeys__})

    def save(self, path: str):

        torch.save(
            {
                "__version__": self.__version__,
                **{key: getattr(self, key) for key in self.__savekeys__},
            },
            path,
        )
