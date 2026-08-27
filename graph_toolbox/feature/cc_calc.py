"""
script for calculating residue - residue interactions
input structure is assumed to be cleaned, i.e. no altLoc, continous sequence numbering, etc.
"""

import sys
from typing import List, Optional

sys.path.append("..")
import pandas as pd
import torch as th
import numpy as np

#from biopandas.pdb import PandasPdb

from .models import StructFeats, StructFeatsCC
from .numeric import compute_edge_features, distance, backbone_dihedral, sidechain_dihedral, virtual_cb
from .numeric_edge import compute_edge_features_sparse_bio
from .rotary_matrix import calculate_relative_rotation_matrix, calculate_relative_sidechain_rotation
from .params import (
    C_DELTA,
    C_GAMMA,
    gamma_choice,
    delta_choice
)


# def map_aa_name(resname):
#     """
#     map residue residue names into standrad ones
#     """
#     if resname in aa_trans:
#         return aa_trans[resname]
#     else:
#         return aa_trans


nan_type = float("nan")
nan_xyz = (nan_type, nan_type, nan_type)

def read_struct_cc(
    pdbloc: pd.DataFrame, t: Optional[float] = None, with_interactions=True, with_relative_rotations=True, order: Optional[list] = None,
    helix_indices: Optional[th.Tensor] = None
) -> StructFeatsCC:
    """
    Parses structural features from a PDB dataframe.
    * adds self loops for further graph processing
    * adds pseudo cb for glycine
    Parameters
    ----------
    pdbloc : pd.DataFrame
        DataFrame representing the parsed PDB structure, with at least 'chain_id',
        'residue_number', and 'atom_name' columns.
    t : float, optional
        Distance threshold (in Å) for defining CA-CA contacts. Default is 8.0.
    helix_indices : bool, optional
        If True, include additional residue-residue interaction features.
    order : list of tuple, optional
        Optional list specifying the desired order of residues as (residue_number, chain_id).
        If empty, residues are processed in the order they appear in the DataFrame (groupby).

    Returns
    -------
    StructFeatsCC
        An object containing structural features (e.g., node and edge information).
    """

    if with_interactions:
        raise ValueError('with_interactions option not yet implemented')

    data = pdbloc.sort_values(by=["chain_id", "residue_number"]).copy()

    if not isinstance(t, (int, float)):
        raise ValueError(f"CA distance cut-off must be a number, given: {type(t)}")

    # per residue data containers
    chainids = list()
    residues = list()
    res_at_num = list()  # number of atoms
    res_per_res = list()  # three-letter residue name

    # per atom data containers
    ca_xyz, cb_xyz = list(), list()
    cg_xyz, cd_xyz = list(), list()
    c_xyz, n_xyz = list(), list()
    residues_name = list()
    #is_side_chain = list()
    atoms, name = list(), list()

    # Group residues
    groups = data.groupby(["residue_number", "chain_id"])
    
    # Prepare the iterator based on whether 'order' is provided
    if order:
        order = list(map(tuple, order))  # ensure hashable and consumable multiple times
    
        assert set(order) == set(groups.groups.keys()), \
            "'order' must contain exactly the same (residue_number, chain_id) pairs as in 'data'."
    
        # Create iterator with ordered access using precomputed groups
        residue_iterator = (
            (i, ((res_id, chain_id), groups.get_group((res_id, chain_id))))
            for i, (res_id, chain_id) in enumerate(order)
        )
    else:
        # Default iterator (natural groupby order)
        residue_iterator = (
            (i, ((res_id, chain_id), residue))
            for i, ((res_id, chain_id), residue) in enumerate(groups)
        )
    
    ## iterate over residues
    #for res_enum, ((chainid, resid), residue) in enumerate(
    #    data.groupby(["chain_id", "residue_number"])
    #):

    #print(list(residue_iterator))

    # use adaptative iterator
    for res_enum, ((resid, chainid), residue) in residue_iterator:
    
        missing_ca = True
        missing_cb = True
        missing_c = True
        missing_n = True
        missing_cg = True
        missing_cd = True

        # fill per residue data containers
        chainids.append(chainid)
        residues.append(resid)
        res_at_num.append(residue.shape[0])
        res_name = residue.iloc[0].residue_name
        res_per_res.append(res_name)
        
        # iterate over atoms of a residue
        for _, atom in residue.iterrows():

            n = atom.atom_name

            if n == "CA":
                assert missing_ca
                ca_xyz.append((atom.x_coord, atom.y_coord, atom.z_coord))
                missing_ca = False
            elif n == "CB":
                assert missing_cb, residue
                cb_xyz.append((atom.x_coord, atom.y_coord, atom.z_coord))
                missing_cb = False
            elif n == "C":
                assert missing_c
                c_xyz.append((atom.x_coord, atom.y_coord, atom.z_coord))
                missing_c = False
            elif n == "N":
                assert missing_n
                n_xyz.append((atom.x_coord, atom.y_coord, atom.z_coord))
                missing_n = False
                
            elif n in C_GAMMA:
                if n == gamma_choice[res_name]:  # pick the best CG atom for a given residue
                    assert missing_cg
                    cg_xyz.append((atom.x_coord, atom.y_coord, atom.z_coord))
                    missing_cg = False
            elif n in C_DELTA:
                if n == delta_choice[res_name]:  # pick the best CD atom for a given residue
                    assert missing_cd
                    cd_xyz.append((atom.x_coord, atom.y_coord, atom.z_coord))
                    missing_cd = False

            # FIXME: why?
            elif len(n) == 3:
                n = n[:2]

            # fill per atom data containers
            atoms.append((atom.x_coord, atom.y_coord, atom.z_coord))
            residues_name.append(atom.residue_name)
            name.append(n)
            #is_side_chain.append(True if n in side_chain_atom_names else False)

        # residue description for error messages
        tmp_res_text = f"{chainid}:{res_name}-{resid}"

        # check backbone atoms
        assert not missing_ca, f"missing Ca atom in {tmp_res_text}!"
        assert not missing_c, f"missing C atom in {tmp_res_text}!"
        assert not missing_n, f"missing N atom in {tmp_res_text}!"

        # chek Cb atom
        if missing_cb:
            # gly does not have CB by definition
            if res_name != "GLY":
                raise ValueError(f"missing Cb atom in {tmp_res_text}!")
            else:
                cb_xyz.append(nan_xyz)

        # fill missing gamma and delta carbons
        if missing_cg:
            cg_xyz.append(nan_xyz)
        if missing_cd:
            cd_xyz.append(nan_xyz)

    # check per atom containers
    is_side_chain = residues_name # quick fix for assert. is_side_chain is not used.
    assert len(set(map(len, [residues_name, is_side_chain, atoms, name]))) == 1, (
        f"All per-atom lists must have equal length, got lengths: "
        f"res_name={len(residues_name)}, side_chain={len(is_side_chain)}, "
        f"atoms={len(atoms)}, name={len(name)}"
    )

    # check per residue containers
    assert (
        len(chainids)
        == len(residues)
        == len(res_at_num)
        == len(res_per_res)
        == len(ca_xyz)
        == len(cb_xyz)
        == len(cg_xyz)
        == len(cd_xyz)
        == len(c_xyz)
        == len(n_xyz)
    ), (
        f"All per-residue and backbone/sidechain atom lists must have equal length, got lengths: "
        f"chainids={len(chainids)}, residues={len(residues)}, "
        f"res_at_num={len(res_at_num)}, res_per_res={len(res_per_res)}, "
        f"ca={len(ca_xyz)}, cb={len(cb_xyz)}, cg={len(cg_xyz)}, cd={len(cd_xyz)}, "
        f"c={len(c_xyz)}, n={len(n_xyz)}"
    )

    num_atoms = len(name)  # number of atoms in the structure
    num_residues = len(res_per_res)  # number of residues in the structure
    
    # identyfikatory reszt i łańcuchów (zeby mialy potrzebna potem funkcje roll)
    res_number   = th.LongTensor(residues)        # numery reszt (torch)
    res_chain_np = np.array(chainids)             # łańcuchy jako numpy array

    # convert residue level atoms coordinates to tensors
    res_ca = th.FloatTensor(ca_xyz)
    res_cb = th.FloatTensor(cb_xyz)
    res_c = th.FloatTensor(c_xyz)
    res_n = th.FloatTensor(n_xyz)
    res_cg = th.FloatTensor(cg_xyz)
    res_cd = th.FloatTensor(cd_xyz)

    # glycine lacks a CB atom (nan placeholder above); fill an idealized virtual
    # CB from the backbone N, CA, C so cb_dist / distancemx are NaN-free.
    # Vectorized single cross-product -> negligible cost; real CBs untouched.
    gly_mask = th.isnan(res_cb).any(dim=-1, keepdim=True)
    res_cb = th.where(gly_mask, virtual_cb(res_n, res_ca, res_c), res_cb)

    # binary mask of helix residues
    segment_ids = th.ones(num_residues, dtype=th.long)
    segment_ids[helix_indices] = 0

    # integer chain id per residue (rich edge features need a numeric chain tensor)
    chain_id = th.as_tensor(np.unique(res_chain_np, return_inverse=True)[1], dtype=th.long)

    ca_dist = distance(res_ca)
    cb_dist = distance(res_cb)
    
    # fill self CA distance to 0
    ca_dist = ca_dist.fill_diagonal_(0)

    # fill self CB distance to 0
    cb_dist = cb_dist.fill_diagonal_(0)

    # define CA contacts (row and column indexes) below the cut-off parameter `t`.
    # Force the diagonal on so every residue always keeps exactly one self-loop,
    # independent of the cut-off `t` or the CA-CA geometry. DGL expects self-loops
    # and otherwise warns about 0-in-degree nodes. `th.where` emits each (i, i)
    # once, so this stays idempotent (no duplicated self-loops).
    contact_mask = ca_dist < t
    contact_mask.fill_diagonal_(True)
    u, v = th.where(contact_mask)

    # is_seq[i, j] stores the sequence distance |i - j| between residue indices; 
    # shape: (num_residues, num_residues)
    res_id_short = th.arange(0, num_residues)
    if helix_indices is not None and helix_indices.numel() > 0:
        efeats = compute_edge_features_sparse_bio(
            res_index=res_number, segment_id=segment_ids, chain_id=chain_id, u=u, v=v
        )
    else:
        raise NotImplementedError("edge features without helix_indices not implemented")
    # dihedral angles
    backbone_dih = backbone_dihedral(res_n, res_ca, res_c)
    sidechain_dih = sidechain_dihedral(res_n, res_ca, res_cb, res_cg, res_cd)
    
    # backbone dihedral angles do not make sense when sequence is discontinuous

    # --- maska „dziur w numeracji” (torch)
    is_prev_num_gap = (res_number - res_number.roll(1)).abs() != 1
    is_next_num_gap = (res_number - res_number.roll(-1)).abs() != 1
    
    # --- maska „zmiany łańcucha” (numpy)
    is_prev_chain_gap = res_chain_np != np.roll(res_chain_np, 1)
    is_next_chain_gap = res_chain_np != np.roll(res_chain_np, -1)
    is_prev_chain_gap[0]  = True   # brak poprzednika
    is_next_chain_gap[-1] = True   # brak następcy
    
    # --- łączymy obie maski
    is_prev_break = is_prev_num_gap | th.from_numpy(is_prev_chain_gap)
    is_next_break = is_next_num_gap | th.from_numpy(is_next_chain_gap)
    
    # --- zamieniamy niesensowne kąty na NaN
    backbone_dih[is_prev_break, 0] = float("nan")   # φ
    backbone_dih[is_next_break, 1] = float("nan")   # ψ
    
    # residue-residue interactions on atomic level
    # this part needs to be checked
    if with_interactions:
        raise NotImplementedError("with_interactions - feature ignored")
    else:
        # without full interaction descriptiors
        # edge features are only binary labels about structural or sequential contacts
        pass
    if with_relative_rotations:
        backbone_rotations = calculate_relative_rotation_matrix(n=res_n, ca=res_ca, c=res_c)
        sidechain_rotations = calculate_relative_sidechain_rotation(ca=res_ca, cb=res_cb, cg=res_cg)
    else:
        backbone_rotations = None
        sidechain_rotations = None
    return StructFeatsCC(
        u,
        v,
        efeats=efeats,
        nfeats=th.cat((backbone_dih, sidechain_dih), dim=-1),
        sequence=res_per_res,
        distancemx=th.stack((ca_dist, cb_dist), dim=2),
        backbone_relrot=backbone_rotations,
        sidechain_relrot=sidechain_rotations,
        residueid=res_number,
        chainids=chainids,
        with_interactions=with_interactions,
        segment_id=segment_ids
    )