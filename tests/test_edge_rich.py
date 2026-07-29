"""
Tests for the biologically-scaled spectral edge features
(`compute_edge_features_sparse_bio`) wired into the coiled-coil pipeline
(`read_struct_cc` -> `GraphData`).

Every test is parametrized over all real structures in ``tests/data/*.pdb``.
"""

import glob
import os

import numpy as np
import pytest
import torch as th
from biopandas.pdb import PandasPdb

from graph_toolbox.feature.cc_calc import read_struct_cc
from graph_toolbox.feature.base import GraphData
from graph_toolbox.feature.numeric_edge import compute_edge_features_sparse_bio
from graph_toolbox.feature.params import rich_edge_feature_names

test_dir = "tests/data"

NUM_RPE_FREQS = 16
EDGE_DIM = 5 + NUM_RPE_FREQS  # 5 boolean flags + RPE sin/cos spectrum
# indices inside the rich feature vector
IDX_CROSS_SEG, IDX_CROSS_CHAIN, IDX_SELF, IDX_FWD, IDX_BWD = range(5)
RPE_SLICE = slice(5, EDGE_DIM)


def get_test_pdbs() -> list[str]:
    """all .pdb structures under tests/data/"""
    return sorted(glob.glob(os.path.join(test_dir, "*.pdb")))


def _chain_id_tensor(chainids) -> th.Tensor:
    """factorize chain labels the same way ``read_struct_cc`` does"""
    return th.as_tensor(
        np.unique(np.array(chainids), return_inverse=True)[1], dtype=th.long
    )


def _load_cc(pdb: str, t: float = 9):
    """parse a structure through the CC pipeline with a helix covering the first residues"""
    atoms = PandasPdb().read_pdb(pdb).df["ATOM"]
    num_res = atoms.groupby(["residue_number", "chain_id"]).ngroups
    # mark the first (up to) 10 residues as the helix segment (segment_id == 0)
    helix_indices = th.arange(min(10, num_res // 2))
    return read_struct_cc(
        atoms,
        t=t,
        with_interactions=False,
        with_relative_rotations=True,
        helix_indices=helix_indices,
    )


@pytest.mark.parametrize("pdb", get_test_pdbs())
@pytest.mark.parametrize("t", [7, 9])
def test_rich_edge_shape(pdb, t):
    """rich edge features have one row per edge and width 5 + num_rpe_freqs"""
    sd = _load_cc(pdb, t=t)
    assert sd.u.shape == sd.v.shape
    assert sd.efeats.ndim == 2
    assert sd.efeats.shape[0] == sd.u.shape[0]
    assert sd.efeats.shape[1] == EDGE_DIM


@pytest.mark.parametrize("pdb", get_test_pdbs())
def test_rich_edge_matches_direct_call(pdb):
    """the pipeline output is exactly what compute_edge_features_sparse_bio produces"""
    sd = _load_cc(pdb)
    chain_id = _chain_id_tensor(sd.chainids)
    expected = compute_edge_features_sparse_bio(
        res_index=sd.residueid,
        segment_id=sd.segment_id,
        chain_id=chain_id,
        u=sd.u,
        v=sd.v,
        num_rpe_freqs=NUM_RPE_FREQS,
    )
    assert th.allclose(sd.efeats, expected)


@pytest.mark.parametrize("pdb", get_test_pdbs())
def test_rich_edge_semantics(pdb):
    """flag columns are binary and encode the documented topology"""
    sd = _load_cc(pdb)
    f = sd.efeats
    u, v = sd.u, sd.v
    chain_id = _chain_id_tensor(sd.chainids)

    # the five leading flags are strictly boolean
    flags = f[:, :5]
    assert th.all((flags == 0) | (flags == 1))

    # is_self is true node identity (u == v), not a sequence coincidence
    assert th.equal(f[:, IDX_SELF].bool(), (u == v))

    # cross-segment / cross-chain flags match the node labels
    assert th.equal(f[:, IDX_CROSS_SEG].bool(), (sd.segment_id[u] != sd.segment_id[v]))
    assert th.equal(f[:, IDX_CROSS_CHAIN].bool(), (chain_id[u] != chain_id[v]))

    # forward / backward are mutually exclusive and never set on self-loops
    assert th.all(f[:, IDX_FWD] + f[:, IDX_BWD] <= 1)
    assert th.all(f[(u == v), IDX_FWD] == 0)
    assert th.all(f[(u == v), IDX_BWD] == 0)


@pytest.mark.parametrize("pdb", get_test_pdbs())
def test_bio_edge_cross_segment_nullified(pdb):
    """cross-segment (interface) edges rely purely on 3D: RPE and direction are zeroed.

    Note: unlike the fully-nullifying variant, `compute_edge_features_sparse_bio`
    keeps the cross-chain sequence difference as an axial register shift, so only
    cross-segment edges are nullified.
    """
    sd = _load_cc(pdb)
    f = sd.efeats
    cross_seg = f[:, IDX_CROSS_SEG].bool()

    # RPE spectrum fully nullified on cross-segment edges
    assert th.all(f[cross_seg][:, RPE_SLICE] == 0)
    # direction flags nullified on cross-segment edges
    assert th.all(f[cross_seg, IDX_FWD] == 0)
    assert th.all(f[cross_seg, IDX_BWD] == 0)


@pytest.mark.parametrize("pdb", get_test_pdbs())
def test_graphdata_cc_roundtrip(pdb):
    """GraphData names, dataframe export and h5 export stay consistent for CC graphs"""
    sd = _load_cc(pdb)
    g = GraphData(code=f"code_{pdb}", **sd.asdict())

    # efeatname is the rich naming and matches the tensor width
    assert g.efeatname == rich_edge_feature_names(NUM_RPE_FREQS)
    assert len(g.efeatname) == g.efeats.shape[1] == EDGE_DIM

    # edge dataframe builds without the shape-mismatch assertion firing
    edf = g.to_edgedf()
    assert list(edf.columns)[: EDGE_DIM] == g.efeatname
    assert len(edf) == g.efeats.shape[0]

    # h5 export preserves the 21-dim edge features
    h5 = g.to_h5()
    assert h5["efeats_flat"].shape == (g.efeats.shape[0], EDGE_DIM)
