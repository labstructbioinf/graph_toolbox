"""script for calculating residue - residue interactions"""

from typing import List, Optional

import sys

sys.path.append("..")
import pandas as pd
import torch as th

from biopandas.pdb import PandasPdb

from .models import StructFeats
from .numeric import distance, backbone_dihedral, sidechain_dihedral

from .params import (
    BACKBONE,
    HYDROPHOBIC,
    AROMATIC,
    CATION_PI,
    SALT_BRIDGE_C1,
    SALT_BRIDGE_C2,
    CHARGE,
    SIGMA,
    EPSILON,
    HYDROGEN_ACCEPTOR,
    HYDROGEN_DONOR,
    VDW_RADIUS,
    VDW_ATOMS,
    C_DELTA,
    C_GAMMA,
    aa_trans,
)


def map_aa_name(resname):
    """
    map residue residue names into standrad ones
    """
    if resname in aa_trans:
        return aa_trans[resname]
    else:
        return aa_trans


nan_type = float("nan")
nan_xyz = (nan_type, nan_type, nan_type)
atom_id = {ch: i for i, ch in enumerate(CHARGE.keys())}
EPS = th.Tensor([78 * 1e-2])  # unit Farad / angsterm
side_chain_atom_names = ["CA", "C", "N", "O"]
invalid_location = {"B", "C", "D"}


def is_atom_in_group(atoms: List[str], group: set, size: int) -> th.BoolTensor:
    arr = th.zeros(size, dtype=th.bool)
    for i, at in enumerate(atoms):
        if at in group:
            arr[i] = True
    return arr


def residue_atoms_criteria(iterator, criteria_dict: dict, storage: list):
    """
    iterates over structure and look for match in residue - atom level
    """
    for i, (res, at) in iterator:
        # residue level criteria
        if res in criteria_dict:
            # atomic level criteria
            if at in VDW_ATOMS[res]:
                storage[i] = True
    return storage


def read_struct(
    pdbloc: pd.DataFrame, t: Optional[float] = None, with_interactions=True
) -> StructFeats:
    """

    Args:
        pdb_loc (pd.DataFrame): dataframe of parsed PDB structure
        t (float): contact distance threshold
        with_interactions (bool): whether to use add reside-residue interactions features
    return u, v for feats
    """

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
    is_side_chain = list()
    atoms, name = list(), list()

    # iterate over residues
    for res_enum, ((chainid, resid), residue) in enumerate(
        data.groupby(["chain_id", "residue_number"])
    ):
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
                assert missing_cg
                cg_xyz.append((atom.x_coord, atom.y_coord, atom.z_coord))
                missing_cg = False
            elif n in C_DELTA:
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
            is_side_chain.append(True if n in side_chain_atom_names else False)

        # residue description for error messages
        tmp_res_text = f"{chainid}:{res_name}-{resid}"

        # check backbone atoms
        assert not missing_ca, f"missing Ca atom in {tmp_res_text}!"
        assert not missing_c, f"missing C atom in {tmp_res_text}!"
        assert not missing_n, f"missing N atom in {tmp_res_text}!"

        # chek Cb atom
        if missing_cb:
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
    res_number = th.LongTensor(residues)  # list of residue ids

    # convert residue level atoms coordinates to tensors
    res_ca = th.FloatTensor(ca_xyz)
    res_cb = th.FloatTensor(cb_xyz)
    res_c = th.FloatTensor(c_xyz)
    res_n = th.FloatTensor(n_xyz)
    res_cg = th.FloatTensor(cg_xyz)
    res_cd = th.FloatTensor(cd_xyz)

    ca_dist = distance(res_ca)
    cb_dist = distance(res_cb)
    # fill self distance to 0
    ca_dist = ca_dist.fill_diagonal_(0)
    u, v = th.where(ca_dist < t)

    res_id_short = th.arange(0, num_residues)
    is_seq = th.abs(res_id_short.unsqueeze(0) - res_id_short.unsqueeze(1))
    is_self = is_seq == 0
    is_seq_0 = is_seq == 1
    is_seq_1 = is_seq > 5
    # turns usually consider 3 to 5
    is_struct_0 = is_seq > 1

    feats_res = th.stack((is_self, is_seq_0, is_seq_1, is_struct_0), dim=2)
    feats_res = feats_res[u, v]
    # dihedral angles
    backbone_dih = backbone_dihedral(res_n, res_ca, res_c)
    sidechain_dih = sidechain_dihedral(res_n, res_ca, res_cb, res_cg, res_cd)
    # backbone diheral angles does not make sens when sequence is discontinious
    is_prev_res_seq = (res_number - res_number.roll(1)).abs() != 1
    is_next_res_seq = (res_number - res_number.roll(-1)).abs() != 1

    backbone_dih[is_prev_res_seq, 0] = float("nan")
    backbone_dih[is_next_res_seq, 1] = float("nan")
    # residue-residue interactions on atomic level
    if with_interactions:
        # indices of edges ad 2d tensor
        uv = th.where(ca_dist < t)[0]
        # hydrophobic residues mask
        is_res_hf = [True if r in HYDROPHOBIC else False for r in residues_name]

        # hydrogen bonds acceptors and donors atom mask
        is_at_hb_a = [False] * num_atoms
        is_at_hb_d = [False] * num_atoms
        is_at_hb_ac = [True if at.startswith("C") else False for at in name]
        is_at_hb_ad = [True if at.startswith("NH") else False for at in name]

        is_res_ar = is_atom_in_group(residues_name, AROMATIC, num_atoms)
        is_res_cpi = is_atom_in_group(name, CATION_PI, num_atoms)
        is_res_arg = is_atom_in_group(residues_name, "ARG", num_atoms)

        # base atom names (CA,CB,C->C etc.)
        name_base = [n[0] for n in name]

        # van der Waals
        is_at_vdw_other = [False] * num_atoms
        for i, (res, at) in enumerate(zip(residues_name, name)):

            # print(i, res, at)

            # residue level criteria
            if res in VDW_ATOMS:  # przeciez tutaj sa tylko dwie reszty?
                # atomic level criteria
                if at in VDW_ATOMS[res]:
                    is_at_vdw_other[i] = True
            # hbonds
            if res in HYDROGEN_ACCEPTOR:
                if at in HYDROGEN_ACCEPTOR[res]:
                    is_at_hb_a[i] = True
            if res in HYDROGEN_DONOR:
                if at in HYDROGEN_DONOR[res]:
                    is_at_hb_a[i] = True  # ????????? przeciez tutaj musi byc donor?
        is_at_vdw = [True if at in {"C", "S"} else False for at in name_base]

        # sys.exit(-1)

        # get Lennarda-Jones parameters
        at_vdw = [SIGMA[n] for n in name_base]
        sigma = th.FloatTensor(at_vdw)

        at_eps = [EPSILON[n] for n in name_base]
        at_id = th.LongTensor(at_eps)  # ?????? why id?

        at_xyz = th.FloatTensor(atoms)
        at_dist = distance(at_xyz)

        at_is_side = th.BoolTensor(is_side_chain)

        # hbonds acceptor donors
        at_is_hba = th.BoolTensor(is_at_hb_a) | th.BoolTensor(is_at_hb_ac)
        at_is_hbd = th.BoolTensor(is_at_hb_d) | th.BoolTensor(is_at_hb_ad)

        # vdw
        at_is_vdw = th.BoolTensor(is_at_vdw) | th.BoolTensor(is_at_vdw_other)

        # set inverse of the atom self distance to zero to avoid nan/inf when summing
        sigma_radii = sigma.view(-1, 1) + sigma.view(1, -1)

        # === DEFINE EDGE FEATURES

        # binary residue-residue interactions
        disulfde = (at_id == 4) & (at_dist < 2.2)  ### will be never 4!!!
        at_dist_5a = at_dist < 5.0
        hydrophobic = at_dist_5a & (at_is_side == False) & th.BoolTensor(is_res_hf)
        cation_pi = (at_dist < 6) & is_res_cpi
        arg_arg = at_dist_5a & is_res_arg

        # salt bridges

        # SALT_BRIDGE_PAIRS = {
        #     ("ARG", "ASP"),
        #     ("ARG", "GLU"),
        #     ("LYS", "ASP"),
        #     ("LYS", "GLU"),
        # }

        is_at_sb_c1 = is_atom_in_group(name, SALT_BRIDGE_C1, num_atoms)
        is_res_sb_c1 = is_atom_in_group(residues_name, {"ARG", "LYS"}, num_atoms)
        is_at_sb_c2 = is_atom_in_group(
            name, {"ARG", "GLU"}, num_atoms
        )  ## ARG a nie ASP????
        is_res_sb_c2 = is_atom_in_group(residues_name, SALT_BRIDGE_C2, num_atoms)

        sb_tmp1 = th.BoolTensor(is_at_sb_c1).view(-1, 1) & th.BoolTensor(
            is_at_sb_c2
        ).view(1, -1)
        sb_tmp2 = th.BoolTensor(is_res_sb_c1).view(-1, 1) & th.BoolTensor(
            is_res_sb_c2
        ).view(1, -1)

        salt_bridge = sb_tmp1 & (at_dist < 5.0) & sb_tmp2

        # hbonds
        hbond = at_is_hba.view(-1, 1) & at_is_hbd.view(1, -1) & (at_dist < 3.5)

        # vdw interactions
        vdw = ((at_dist - sigma_radii) < 0.5) & at_is_vdw

        # define edge features
        feats = th.cat(
            (
                disulfde.unsqueeze(2),
                hydrophobic.unsqueeze(2),
                cation_pi.unsqueeze(2),
                arg_arg.unsqueeze(2),
                salt_bridge.unsqueeze(2),
                hbond.unsqueeze(2),
                vdw.unsqueeze(2),
            ),
            dim=2,
        )

        # residue level features
        efeat_list = list()

        first_dim_split = feats.split(res_at_num, 0)
        for i in range(len(res_at_num)):
            efeat_list.extend(list(first_dim_split[i].split(res_at_num, 1)))
        # sum only for residues within threshold
        feats_at = th.cat([efeat_list[e].sum((0, 1), keepdim=True) for e in uv], dim=0)
        th.clamp(feats_at, min=0, max=1, out=feats_at)
        # gather residue level feature, such as edge criteria
        efeats = th.cat((feats_at.float().squeeze(), feats_res), dim=-1)
    else:
        # without interactions edge features are only binary labels about structural or sequential contacts
        efeats = feats_res
    return StructFeats(
        u,
        v,
        efeats=efeats,
        nfeats=th.cat((backbone_dih, sidechain_dih), dim=-1),
        sequence=res_per_res,
        distancemx=th.stack((ca_dist, cb_dist), dim=2),
        residueid=res_number,
        chainids=chainids,
        with_interactions=with_interactions,
    )


def edge_embedding_to_3d_tensor(u, v, emb):
    num_nodes = u.max() + 1
    contacts = th.zeros((num_nodes, num_nodes, emb.shape[-1]))
    emb_normed = emb / emb.norm(2, dim=1, keepdim=True)
    contacts[u, v] = emb_normed
    return contacts


if __name__ == "__main__":
    file = "../tests/data/3sxw.pdb.gz"
    _, _, _, _ = read_struct(file, "A", t=7)
