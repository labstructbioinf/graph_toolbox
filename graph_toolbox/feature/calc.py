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
    pdbloc: str | pd.DataFrame,
    chain: Optional[str] = None,
    t: Optional[float] = None,
    indices_to_read: Optional[list] = None,
) -> StructFeats:
    """
    params:
        pdb_loc (str, set, atomium.Model): path to structure, atomium selection
        chain (str): one letter pdb code
        t (float): contact distance threshold
    return u, v for feats
    """
    if isinstance(pdbloc, str):
        data = PandasPdb().read_pdb(pdbloc)
        atoms = data.df["ATOM"]
        hetatm = data.df["HETATM"]
        hetatm = hetatm[hetatm.residue_name.isin(aa_trans)]
        hetatm["residue_name"] = hetatm["residue_name"].apply(map_aa_name)
        data = pd.concat([atoms, hetatm], axis=0)
    elif isinstance(pdbloc, pd.DataFrame):
        data = pdbloc
    if not isinstance(t, (int, float, type(None))):
        raise ValueError(f"threshold must be number or None, given: {type(t)}")
    else:
        if isinstance(t, (int, float)) and t < 5:
            print("dumb threshold")
    if chain is not None:
        data = data.loc[data["chain_id"] == chain].copy()
        if data.shape[0] == 0:
            raise ValueError(f"no data in chain {chain}")
    if indices_to_read is not None:
        data = data.loc[data["residue_number"].isin(indices_to_read)].copy()
    atoms, name = list(), list()
    ca_xyz, cb_xyz = list(), list()
    cg_xyz, cd_xyz = list(), list()
    c_xyz, n_xyz = list(), list()
    residues, residues_name = list(), list()
    res_per_res = list()
    chainids = list()
    is_side_chain = list()
    res_at_num = list()
    # remove ligands like DD1, dd2
    mask = ~data.atom_name.str.startswith("D")
    # remove alt located atoms
    mask &= ~data.alt_loc.isin(invalid_location)
    # remove insertions - this column may not exist
    if "insertion" in data.columns:
        mask &= data.insertion == ""
    data = data[mask].sort_values(by=["chain_id", "residue_number"]).copy()
    for res_enum, ((chainid, resid), residue) in enumerate(
        data.groupby(["chain_id", "residue_number"])
    ):
        missing_cg = True
        missing_cb = True
        missing_ca = True
        missing_cg = True
        missing_cd = True
        missing_c = True
        chainids.append(chainid)
        num_atoms = residue.shape[0]
        res_at_num.append(num_atoms)
        residues.append(resid)
        res_per_res.append(residue.iloc[0].residue_name)
        for _, atom in residue.iterrows():
            n = atom.atom_name
            if atom.atom_name == "CA":
                ca_xyz.append((atom.x_coord, atom.y_coord, atom.z_coord))
                missing_ca = False
            elif atom.atom_name == "CB":
                cb_xyz.append((atom.x_coord, atom.y_coord, atom.z_coord))
                missing_cb = False
            elif atom.atom_name == "C":
                c_xyz.append((atom.x_coord, atom.y_coord, atom.z_coord))
                missing_c = False
            elif atom.atom_name == "N":
                n_xyz.append((atom.x_coord, atom.y_coord, atom.z_coord))
            elif atom.atom_name in C_GAMMA:
                missing_cg = False
                cg_xyz.append((atom.x_coord, atom.y_coord, atom.z_coord))
            elif atom.atom_name in C_DELTA:
                missing_cd = False
                cd_xyz.append((atom.x_coord, atom.y_coord, atom.z_coord))
            elif len(n) == 3:
                n = n[:2]
            atoms.append((atom.x_coord, atom.y_coord, atom.z_coord))
            residues_name.append(atom.residue_name)
            name.append(n)
            is_side_chain.append(True if n in side_chain_atom_names else False)
        if missing_cb:
            cb_xyz.append(nan_xyz)
        if missing_ca:
            ca_xyz.append(nan_xyz)
        if missing_cg:
            cg_xyz.append(nan_xyz)
        if missing_cd:
            cd_xyz.append(nan_xyz)
        if missing_c:
            c_xyz.append(nan_xyz)

    # assign parameters to atoms
    num_atoms = len(name)
    num_residues = len(res_per_res)
    res_number = th.LongTensor(residues)
    # add vitual Carbon Beta to glycine
    # ca_cb_len = 1.53363
    # for rid, res_name in enumerate(residues_name):
    #     if res_name.lower() == "gly":
    #         n = n_xyz[i]
    #         ca = ca_xyz[i]
    #         c = ca_xyz[i]

    name_base = [n[0] for n in name]
    # at_charge = [CHARGE[n] for n in name_base]
    at_vdw = [SIGMA[n] for n in name_base]
    # atom_arr = [atom_id[n] for n in name_base]
    at_eps = [EPSILON[n] for n in name_base]
    # convert to tensors
    res_ca = th.FloatTensor(ca_xyz)
    res_dist = distance(res_ca)
    res_cb = th.FloatTensor(cb_xyz)
    res_cg = th.FloatTensor(cg_xyz)
    res_cd = th.FloatTensor(cd_xyz)
    # res_cg = th.FloatTensor(cg_xyz)
    res_c = th.FloatTensor(c_xyz)
    res_n = th.FloatTensor(n_xyz)
    # print(minlength, chainlength)
    if num_residues != res_ca.shape[0]:
        raise ValueError(
            f"number of residues is different then CA atoms {num_residues} and {res_ca.shape[0]}"
        )
    # check variuos atom/residue types
    # hydrophobic
    is_res_hf = [True if r in HYDROPHOBIC else False for r in residues_name]
    # hydrogen bonds
    is_at_hb_a = [False] * num_atoms
    is_at_hb_d = [False] * num_atoms
    is_at_hb_ac = [True if at.startswith("C") else False for at in name]
    is_at_hb_ad = [True if at.startswith("NH") else False for at in name]
    is_res_ar = is_atom_in_group(residues_name, AROMATIC, num_atoms)
    is_res_cpi = is_atom_in_group(name, CATION_PI, num_atoms)
    is_res_arg = is_atom_in_group(residues_name, "ARG", num_atoms)
    # salt bridge
    is_at_sb_c1 = is_atom_in_group(name, SALT_BRIDGE_C1, num_atoms)
    is_res_sb_c1 = is_atom_in_group(residues_name, {"ARG", "LYS"}, num_atoms)
    is_at_sb_c2 = is_atom_in_group(name, {"ARG", "GLU"}, num_atoms)
    is_res_sb_c2 = is_atom_in_group(residues_name, SALT_BRIDGE_C2, num_atoms)
    # van der Waals
    is_at_vdw_other = [False] * num_atoms
    for i, (res, at) in enumerate(zip(residues_name, name)):
        # residue level criteria
        if res in VDW_ATOMS:
            # atomic level criteria
            if at in VDW_ATOMS[res]:
                is_at_vdw_other[i] = True
        # hbonds
        if res in HYDROGEN_ACCEPTOR:
            if at in HYDROGEN_ACCEPTOR[res]:
                is_at_hb_a[i] = True
        if res in HYDROGEN_DONOR:
            if at in HYDROGEN_DONOR[res]:
                is_at_hb_a[i] = True
    is_at_vdw = [True if at in {"C", "S"} else False for at in name_base]
    at_xyz = th.FloatTensor(atoms)
    at_dist = distance(at_xyz)
    at_id = th.LongTensor(at_eps)
    sigma = th.FloatTensor(at_vdw)
    at_is_side = th.BoolTensor(is_side_chain)
    # hbonds acceptor donors
    at_is_hba = th.BoolTensor(is_at_hb_a) | th.BoolTensor(is_at_hb_ac)
    at_is_hbd = th.BoolTensor(is_at_hb_d) | th.BoolTensor(is_at_hb_ad)
    # vdw
    at_is_vdw = th.BoolTensor(is_at_vdw) | th.BoolTensor(is_at_vdw_other)
    # set inverse of the atom self distance to zero to avoid nan/inf when summing
    sigma_radii = sigma.view(-1, 1) + sigma.view(1, -1)
    # binary interactions
    disulfde = (at_id == 4) & (at_dist < 2.2)
    at_dist_5a = at_dist < 5.0
    hydrophobic = at_dist_5a & (at_is_side == False) & th.BoolTensor(is_res_hf)
    cation_pi = (at_dist < 6) & is_res_cpi
    arg_arg = at_dist_5a & is_res_arg
    vdw = ((at_dist - sigma_radii) < 0.5) & at_is_vdw
    # hbonds
    hbond = at_is_hba.view(-1, 1) & at_is_hbd.view(1, -1) & (at_dist < 3.5)
    # salt bridge
    sb_tmp1 = th.BoolTensor(is_at_sb_c1).view(-1, 1) & th.BoolTensor(is_at_sb_c2).view(
        1, -1
    )
    sb_tmp2 = th.BoolTensor(is_res_sb_c1).view(-1, 1) & th.BoolTensor(
        is_res_sb_c2
    ).view(1, -1)
    salt_bridge = sb_tmp1 & (at_dist < 5.0) & sb_tmp2

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
    # change feature resolution
    # residue level features
    # breakpoint()
    efeat_list = list()

    first_dim_split = feats.split(res_at_num, 0)
    for i in range(len(res_at_num)):
        efeat_list.extend(list(first_dim_split[i].split(res_at_num, 1)))
    res_dist = res_dist.fill_diagonal_(0)
    u, v = th.where(res_dist < t)
    uv = th.where(res_dist < t)[0]
    # sum only for residues within threshold
    feats_at = th.cat([efeat_list[e].sum((0, 1), keepdim=True) for e in uv], dim=0)
    th.clamp(feats_at, min=0, max=1, out=feats_at)
    # gather residue level feature, such as edge criteria
    cb_dist = distance(res_cb)
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
    # breakpoint()
    # is_prev_res_seq[0] = False
    # is_next_res_seq[0] = False
    backbone_dih[is_prev_res_seq, 0] = float("nan")
    backbone_dih[is_next_res_seq, 1] = float("nan")

    return StructFeats(
        u,
        v,
        efeats=th.cat((feats_at.float().squeeze(), feats_res), dim=-1),
        nfeats=th.cat((backbone_dih, sidechain_dih), dim=-1),
        sequence=res_per_res,
        distancemx=th.stack((res_dist, cb_dist), dim=2),
        residueid=res_number,
        chainids=chainids,
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
