'''script for calculating residue - residue interactions'''
from typing import Union

import atomium
import torch as th
from .params import (HYDROPHOBIC,
                     AROMATIC,
                     CATION_PI,
                     SALT_BRIDGE_C1,
                     SALT_BRIDGE_C2,
                     CHARGE,
                    SIGMA,
                     EPSILON,
                    HYDROGEN_ACCEPTOR,
                    HYDROGEN_DONOR)


nan_type = float('nan')
atom_id = {ch : i for i, ch in enumerate(CHARGE.keys())}
EPS = th.Tensor([78*1e-2]) # unit Farad / angsterm
CLIP_MAX = th.FloatTensor([1,1,1,1,1,1,10,10,10,10,1,1,1,1,1,1])
CLIP_MIN = th.FloatTensor([0,0,0,0,0,0,-10,-10,-10,1e-20,0,0,0,0,0,0])


def read_struct(pdb_loc: Union[str,atomium.structures.Model], chain: str, t: int):
    '''
    params:
        pdb_loc (str or atomium.model)
        t (float) residue-residue distance criteria
    return distances, edge binary feats, u, v for feats
    '''
    if isinstance(pdb_loc, str):
        data = atomium.open(pdb_loc).model
    elif isinstance(pdb_loc, atomium.structures.Model):
        data = pdb_loc
    else:
        raise KeyError('wrong pdb_loc')
    if not isinstance(t, (int, float)):
        raise ValueError(f'threshold must be number, given: {type(t)}')
    else: 
        if t < 5:
            print('dumb threshold')

    atoms, name = [], []
    ca_xyz, cb_xyz = [], []
    residues, residues_name = [], []
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
            if 'CA' not in r_at_name:
                raise KeyError('missing CA atom')
    
    name_base = [n[0] for n in name]
    at_charge = [CHARGE[n] for n in name_base]
    at_vdw = [SIGMA[n] for n in name_base]
    atom_arr = [atom_id[n] for n in name_base]
    at_eps = [EPSILON[n] for n in name_base]
    
    res_id = th.LongTensor(residues)
    res_xyz = th.FloatTensor(ca_xyz)
    res_dist = th.cdist(res_xyz, res_xyz)
    res_cb = th.FloatTensor(cb_xyz)

    is_res_hf = [True if r in HYDROPHOBIC else False for r in residues_name]
    is_at_hb_a = [True if r in HYDROGEN_ACCEPTOR else False for r in name]
    is_at_hb_d = [True if r in HYDROGEN_DONOR else False for r in name]
    is_res_ar = [True if r in AROMATIC else False for r in residues_name]
    is_res_cpi = [True if at in CATION_PI else False for at in name]
    is_res_arg = [True if r in {'ARG'} else False for r in residues_name]
    is_at_sb_c1 = [True if at in SALT_BRIDGE_C1 else False for at in name]
    is_res_sb_c1 = [True if at in {'ARG', 'LYS'} else False for at in residues_name]
    is_at_sb_c2 = [True if at in {'ARG', 'GLU'} else False for at in name]
    is_res_sb_c2 = [True if at in SALT_BRIDGE_C2 else False for at in residues_name]

    at_xyz = th.FloatTensor(atoms)
    at_dist = th.cdist(at_xyz, at_xyz)
    at_id = th.LongTensor(at_eps)
    sigma = th.FloatTensor(at_vdw)
    epsilon = th.FloatTensor(at_eps)
    at_is_side = th.BoolTensor(is_side_chain)
    at_is_hba = th.BoolTensor(is_at_hb_a)
    at_is_hbd = th.BoolTensor(is_at_hb_d)
    
    at_dist_inv = 1/(at_dist + 1e-6)
    # set inverse of the atom self distance to zero to avoid nan/inf when summing
    at_dist_inv.fill_diagonal_(0) 
    atat_charge = th.FloatTensor(at_charge).view(-1, 1)
    atat_charge = atat_charge * atat_charge.view(1, -1)
    sigma = (sigma.view(-1, 1) + sigma.view(1, -1))/2
    epsilon = th.sqrt(epsilon.view(-1, 1) * epsilon.view(1, -1))
    
    lj_r = sigma*at_dist_inv * (at_dist < 10)
    lj6 = th.pow(lj_r, 6) 
    lj12 = th.pow(lj_r, 12)
    disulfde = (at_id == 4) & (at_dist < 2.2)
    hydrophobic = (at_dist < 5.0) & (at_is_side == False) & th.BoolTensor(is_res_hf)
    cation_pi = (at_dist < 6) & th.BoolTensor(is_res_cpi)
    arg_arg = (at_dist < 5.0) & th.BoolTensor(is_res_arg)
    hbond = at_is_hba.view(-1, 1) & at_is_hba.view(1, -1)
    
    sb_tmp1 = th.BoolTensor(is_at_sb_c1).view(-1, 1) & th.BoolTensor(is_at_sb_c2).view(1, -1)
    sb_tmp2 = th.BoolTensor(is_res_sb_c1).view(-1, 1) & th.BoolTensor(is_res_sb_c2).view(1, -1)

    salt_bridge = sb_tmp1 & (at_dist < 5.0) & sb_tmp2
    
    feats = th.cat((disulfde.unsqueeze(2),
                   hydrophobic.unsqueeze(2),
                   cation_pi.unsqueeze(2),
                   arg_arg.unsqueeze(2),
                   salt_bridge.unsqueeze(2),
                   hbond.unsqueeze(2)), dim=2)
    feats = feats.float()
    coulomb_energy =  (1.0/3.14*EPS) * atat_charge * at_dist_inv
    lenard_jones_energy = epsilon* (lj12 - lj6) 
    energy_sum = coulomb_energy + lenard_jones_energy
    feats = th.cat((feats, 
                    coulomb_energy.unsqueeze(2),
                   lenard_jones_energy.unsqueeze(2),
                   energy_sum.unsqueeze(2)),
                   dim=2)
    # change feature resolution
    # from atomic level to residue level
    efeat_list = list()
    first_dim_split = feats.split(res_at_num, 0)
    for i in range(len(res_at_num)):
        efeat_list.extend(list(first_dim_split[i].split(res_at_num, 1)))
    u, v = th.where(res_dist < t)
    uv = th.where(res_dist.ravel() < t)[0]
    feats_at = th.cat([efeat_list[e].sum((0,1), keepdim=True) for e in uv], dim=0)
    efeats = th.zeros_like(res_dist)
    # gather residue level feature, such as edge criteria
    cb1 = th.linalg.norm(res_cb - res_xyz, dim=1, keepdim=True)
    cb_dist = th.cdist(res_cb, res_cb)
    cb2 = cb1.clone().swapdims(0, 1)
    tn_cb12 = cb1 / (cb2 + 1e-5)
    tn_cb12[th.isnan(tn_cb12)] = 0
    inv_ca12 = 1/(res_dist + 1e-5)
    inv_ca12.fill_diagonal_(0)
    res_id_short = th.arange(0, res_id.max()+1, 1)
    is_seq = th.abs(res_id_short.unsqueeze(0) - res_id_short.unsqueeze(1))
    is_self = is_seq == 0
    is_seq_0 = is_seq == 1
    is_seq_1 = is_seq > 5
    is_struct_0 = is_seq > 1
    is_caca_cbcb = cb_dist < res_dist
    feats_res = th.cat((tn_cb12.unsqueeze(2),
                       inv_ca12.unsqueeze(2),
                        is_caca_cbcb.unsqueeze(2),
                        is_self.unsqueeze(2),
                       is_seq_0.unsqueeze(2),
                       is_seq_1.unsqueeze(2),
                       is_struct_0.unsqueeze(2)), dim=2)
    feats_res = feats_res[u,v]
    feats_all = th.cat((feats_at.squeeze(), feats_res), dim=-1)
    feats_all = th.where(feats_all < CLIP_MIN, CLIP_MIN, feats_all)
    feats_all = th.where(feats_all > CLIP_MAX, CLIP_MAX, feats_all)
    
    return u, v, feats_all


def calc_named(pdb_loc, chain, t):
    name_i = ['disulfide', 'hydrophobic', 'cation_pi', 'arg_arg', 'salt_bridge', 'hbond']
    name_i += ['c', 'lj', 'e']
    name_i += ['cbcb', 'dist', 'ca_vs_cb', 'self', 'is_seq', 'is_seq_not', 'is_struct']
    _, _, feats = read_struct(pdb_loc, chain, t)
    return feats, name_i

