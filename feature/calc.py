'''script for calculating residue - residue interactions'''
import atomium
import torch as th
from .params import (HYDROPHOBIC,
                     AROMATIC,
                     CATION_PI,
                     SALT_BRIDGE_C1,
                     SALT_BRIDGE_C2,
                     CHARGE)


nan_type = float('nan')
atom_id = {ch : i for i, ch in enumerate(CHARGE.keys())}


def read_struct(pdb_loc, chain, t):
    '''
    params:
        pdb_loc (str or atomium.model)
    return distances, edge binary feats, u, v for feats
    '''
    if isinstance(pdb_loc, str):
        data = atomium.open(pdb_loc).model
    elif isinstance(pdb_loc, atomium.structures.Model):
        data = pdb_loc
    else:
        raise KeyError('wrong pdb_loc')

    atoms = []
    name = []
    ca_xyz, cb_xyz = [], []
    residues = []
    residues_name = []
    is_side_chain = []
    res_at_num = []
    for chain in data.chains():
        for i, res in enumerate(chain.residues()):
            r_at_name = [r.name for r in res.atoms()]
            res_at_num.append(len(r_at_name))
            for atom in res.atoms():
                n = atom.name
                if n == 'CA':
                    ca_xyz.append(atom.location)
                elif n == 'CB':
                    cb_xyz.append(atom.location)
                elif len(n) == 3:
                    n = n[:2]
                name.append(n)
                is_side_chain.append(atom.is_side_chain)
                atoms.append(atom.location)
                residues.append(i)
                residues_name.append(res.name)
            if 'CB' not in r_at_name:
                cb_xyz.append((nan_type, nan_type, nan_type))

    name_base = [n[0] for n in name]
    at_charge = [CHARGE[n] for n in name_base]
    atom_arr = [atom_id[n] for n in name_base]
    at_xyz = th.FloatTensor(atoms)
    at_dist = th.cdist(at_xyz, at_xyz)
    at_id = th.LongTensor(atom_arr)
    at_is_side = th.BoolTensor(is_side_chain)

    res_id = th.LongTensor(residues)
    res_xyz = th.FloatTensor(ca_xyz)
    res_dist = th.cdist(res_xyz, res_xyz)
    res_cb = th.FloatTensor(cb_xyz)
    
    is_res_hf = [True if r in HYDROPHOBIC else False for r in residues_name]
    #is_at_rg = [True if at is False else False for at in is_side_chain ]

    is_res_ar = [True if r in AROMATIC else False for r in residues_name]

    is_res_cpi = [True if at in CATION_PI else False for at in name]

    is_res_arg = [True if r in {'ARG'} else False for r in residues_name]

    is_at_sb_c1 = [True if at in SALT_BRIDGE_C1 else False for at in name]
    is_res_sb_c1 = [True if at in {'ARG', 'LYS'} else False for at in residues_name]
    is_at_sb_c2 = [True if at in {'ARG', 'GLU'} else False for at in name]
    is_res_sb_c2 = [True if at in SALT_BRIDGE_C2 else False for at in residues_name]

    at_xyz = th.FloatTensor(atoms)
    at_dist = th.cdist(at_xyz, at_xyz)
    at_id = th.LongTensor(atom_arr)
    at_is_side = th.BoolTensor(is_side_chain)

    res_id = th.LongTensor(residues)
    res_xyz = th.FloatTensor(ca_xyz)
    res_dist = th.cdist(res_xyz, res_xyz)
    res_cb = th.FloatTensor(cb_xyz)
    
    atat_charge = th.FloatTensor(at_charge).view(-1, 1)
    atat_charge = atat_charge * th.FloatTensor(at_charge).view(1, -1)

    at_dist_inv = 1e-6/th.pow(at_dist + 1e-2, 2)
    
    disulfde = (at_id == 4) & (at_dist < 2.2)
    hydrophobic = (at_dist < 5.0) & (at_is_side == False) & th.BoolTensor(is_res_hf)
    cation_pi = (at_dist < 6) & th.BoolTensor(is_res_cpi)
    arg_arg = (at_dist < 5.0) & th.BoolTensor(is_res_arg)

    sb_tmp1 = th.BoolTensor(is_at_sb_c1).view(-1, 1) & th.BoolTensor(is_at_sb_c2).view(1, -1)
    sb_tmp2 = th.BoolTensor(is_res_sb_c1).view(-1, 1) & th.BoolTensor(is_res_sb_c2).view(1, -1)

    salt_bridge = sb_tmp1 & (at_dist < 5.0) & sb_tmp2

    feats = th.cat((disulfde.unsqueeze(2),
                   hydrophobic.unsqueeze(2),
                   cation_pi.unsqueeze(2),
                   arg_arg.unsqueeze(2),
                   salt_bridge.unsqueeze(2)), dim=2)
    feats = feats.float()
    coulomb_force = at_dist_inv * atat_charge
    feats = th.cat((feats, 
                    coulomb_force.unsqueeze(2)), dim=2)
    
    efeat_list = list()
    first_dim_split = feats.split(res_at_num, 0)
    for i in range(len(res_at_num)):
        efeat_list.extend(list(first_dim_split[i].split(res_at_num, 1)))
        
    u, v = th.where(res_dist < t)
    uv = th.where(res_dist.ravel() < t)[0]
    feats_at = th.cat([efeat_list[e].sum((0,1), keepdim=True) for e in uv], dim=0)
    efeats = th.zeros_like(res_dist)
    
    cb1 = th.linalg.norm(res_cb - res_xyz, dim=1, keepdim=True)
    cb2 = cb1.clone().swapdims(0, 1)
    tn_cb12 = cb1 / (cb2 + 1e-2)
    tn_cb12[th.isnan(tn_cb12)] = -1
    inv_ca12 = 1/(res_dist - 1e-3)

    res_id_short = th.arange(0, res_id.max()+1, 1)
    is_seq = th.abs(res_id_short.unsqueeze(0) - res_id_short.unsqueeze(1))
    is_self = is_seq == 0
    is_seq_0 = is_seq == 1
    is_seq_1 = is_seq == 2
    is_struct_0 = ~is_seq_0
    feats_res = th.cat((tn_cb12.unsqueeze(2),
                       inv_ca12.unsqueeze(2),
                        is_self.unsqueeze(2),
                       is_seq_0.unsqueeze(2),
                       is_seq_1.unsqueeze(2),
                       is_struct_0.unsqueeze(2)), dim=2)
    feats_res = feats_res[u,v]
    feats_all = th.cat((feats_at.squeeze(), feats_res), dim=-1)
    
    return u, v, feats_all