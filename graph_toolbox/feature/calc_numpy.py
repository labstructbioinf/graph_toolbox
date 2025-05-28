'''script for calculating residue - residue interactions'''
from typing import Union, Tuple, List

import sys
sys.path.append('..')
import atomium
import pandas as pd
import numpy as np
import torch as th
from .params import (BACKBONE,
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
                    VDW_ATOMS)
try:
    from parse import atomium_select, atomium_chain_pdb_list
except:
    pass


nan_type = float('nan')
atom_id = {ch : i for i, ch in enumerate(CHARGE.keys())}
EPS = th.Tensor([78*1e-2]) # unit Farad / angsterm
CLIP_MAX = th.FloatTensor([1,1,1,1,1,1,1,10,10,10,10,1,1,1,1,1,1])
CLIP_MIN = th.FloatTensor([0,0,0,0,0,0,0,-10,-10,-10,1e-20,0,0,0,0,0,0])

def cdist(arr1: np.ndarray, arr2: np.ndarray):
    dist = arr1[:, np.newaxis] - arr2[np.newaxis,:]
    return np.sqrt((dist**2).sum(-1))

def is_atom_in_group(atoms, group):
    arr = np.zeros(len(atoms), dtype=np.bool)
    for i, at in enumerate(atoms):
        if at in group:
            arr[i] = True
    return arr

def residue_atoms_criteria(iterator, criteria_dict : dict, storage: list):
    '''
    iterates over structure and look for match in residue - atom level
    '''
    for i, (res, at) in iterator:
        # residue level criteria
        if res in criteria_dict:
            # atomic level criteria
            if at in VDW_ATOMS[res]:
                storage[i] = True
    return storage

def read_struct(pdb_loc: Union[str, list, atomium.structures.Model],
                chain: Union[str, None],
                t: float) -> Tuple[th.Tensor]:
    '''
    params:
        pdb_loc (str, set, atomium.Model): path to structure, atomium selection 
        chain (str): one letter pdb code 
        t (float): contact distance threshold
    return u, v for feats
    '''
    if isinstance(pdb_loc, str):
        data = atomium.open(pdb_loc).model.residues()
    elif isinstance(pdb_loc, atomium.structures.Model):
        data = pdb_loc.residues()
    elif isinstance(pdb_loc, list):
        data = pdb_loc
    else:
        raise KeyError(f'wrong pdb_loc type {type(pdb_loc)}')
    if not isinstance(t, (int, float)):
        raise ValueError(f'threshold must be number, given: {type(t)}')
    else: 
        if t < 5:
            print('dumb threshold')
    if chain is not None:
        data = data.chain(chain)
    atoms, name = [], []
    ca_xyz, cb_xyz = [], []
    residues, residues_name = [], []
    is_side_chain = []
    res_at_num = []
    for i, res in enumerate(data):
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
    # assign parameters to atoms
    num_atoms = len(name)
    name_base = [n[0] for n in name]
    at_charge = [CHARGE[n] for n in name_base]
    at_vdw = [SIGMA[n] for n in name_base]
    atom_arr = [atom_id[n] for n in name_base]
    at_eps = [EPSILON[n] for n in name_base]
    # convert to tensors
    res_id = np.array(residues, dtype=np.int32)
    res_xyz = np.array(ca_xyz, dtype=np.float32)
    res_dist = cdist(res_xyz, res_xyz)
    res_dist = res_xyz
    res_cb = np.array(cb_xyz, dtype=np.float32)
    # check variuos atom/residue types
    # hydrophobic
    is_res_hf = [True if r in HYDROPHOBIC else False for r in residues_name]
    # hydrogen bonds
    is_at_hb_a = [False]*num_atoms
    is_at_hb_d = [False]*num_atoms
    is_at_hb_ac = [True if at.startswith("C") else False for at in name]
    is_at_hb_ad = [True if at.startswith('NH') else False for at in name]
    is_res_ar = is_atom_in_group(residues_name, AROMATIC)
    is_res_cpi = is_atom_in_group(name, CATION_PI)
    is_res_arg = is_atom_in_group(residues_name, {'ARG'})
    # salt bridge
    is_at_sb_c1 = is_atom_in_group(name, SALT_BRIDGE_C1)
    is_res_sb_c1 = is_atom_in_group(residues_name, {'ARG', 'LYS'})
    is_at_sb_c2 = [True if at in {'ARG', 'GLU'} else False for at in name]
    is_res_sb_c2 = [True if at in SALT_BRIDGE_C2 else False for at in residues_name]
    # van der Waals
    is_at_vdw_other = [False]*num_atoms
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
    is_at_vdw  = [True if at in {'C', 'S'} else False for at in name_base]
    at_xyz = th.FloatTensor(atoms)
    at_dist = th.cdist(at_xyz, at_xyz)
    at_id = th.LongTensor(at_eps)
    sigma = th.FloatTensor(at_vdw)
    epsilon = th.FloatTensor(at_eps)
    at_is_side = th.BoolTensor(is_side_chain)
    # hbonds acceptor donors
    at_is_hba = th.BoolTensor(is_at_hb_a) | th.BoolTensor(is_at_hb_ac)
    at_is_hbd = th.BoolTensor(is_at_hb_d) | th.BoolTensor(is_at_hb_ad)
    #vdw
    at_is_vdw = th.BoolTensor(is_at_vdw) | th.BoolTensor(is_at_vdw_other)
    at_dist_inv = 1/(at_dist + 1e-6)
    # set inverse of the atom self distance to zero to avoid nan/inf when summing
    at_dist_inv.fill_diagonal_(0) 
    atat_charge = th.FloatTensor(at_charge).view(-1, 1)
    atat_charge = atat_charge * atat_charge.view(1, -1)
    sigma_coeff = (sigma.view(-1, 1) + sigma.view(1, -1))/2
    sigma_radii = (sigma.view(-1, 1) + sigma.view(1, -1))
    epsilon = th.sqrt(epsilon.view(-1, 1) * epsilon.view(1, -1))
    
    lj_r = sigma_coeff*at_dist_inv * (at_dist < 10)
    lj6 = th.pow(lj_r, 6) 
    lj12 = th.pow(lj_r, 12)
    # binary interactions
    disulfde = (at_id == 4) & (at_dist < 2.2)
    hydrophobic = (at_dist < 5.0) & (at_is_side == False) & th.BoolTensor(is_res_hf)
    cation_pi = (at_dist < 6) & th.BoolTensor(is_res_cpi)
    arg_arg = (at_dist < 5.0) & th.BoolTensor(is_res_arg)
    vdw = ((at_dist - sigma_radii) < 0.5) & at_is_vdw
    # hbonds
    hbond = at_is_hba.view(-1, 1) & at_is_hba.view(1, -1) & (at_dist < 3.5)
    # salt bridge
    sb_tmp1 = th.BoolTensor(is_at_sb_c1).view(-1, 1) & th.BoolTensor(is_at_sb_c2).view(1, -1)
    sb_tmp2 = th.BoolTensor(is_res_sb_c1).view(-1, 1) & th.BoolTensor(is_res_sb_c2).view(1, -1)
    salt_bridge = sb_tmp1 & (at_dist < 5.0) & sb_tmp2
    
    feats = th.cat((disulfde.unsqueeze(2),
                   hydrophobic.unsqueeze(2),
                   cation_pi.unsqueeze(2),
                   arg_arg.unsqueeze(2),
                   salt_bridge.unsqueeze(2),
                   hbond.unsqueeze(2),
                   vdw.unsqueeze(2)), dim=2)
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
    uv = th.where(res_dist < t)[0]
    feats_at = th.cat([efeat_list[e].sum((0,1), keepdim=True) for e in uv], dim=0)
    efeats = th.zeros_like(res_dist)
    # gather residue level feature, such as edge criteria
    if hasattr(th, 'linalg'):
        cb1 = th.linalg.norm(res_cb - res_xyz, dim=1, keepdim=True)
    else:
        cb1 = th.norm(res_cb - res_xyz, dim=1, keepdim=True)
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


def calc_struct_properties(resname: List[str],
                            atomname: List[str],
                            resatnum: List[int],
                            atomxyz: th.Tensor,
                            caxyz: th.Tensor,
                            cbxyz: th.Tensor,
                            t: float = 7):

    num_atoms = len(resname)
    num_residues = caxyz.shape[0]
    atombase = [n[0] for n in atomname]
    at_charge = [CHARGE[n] for n in atombase]
    at_vdw = [SIGMA[n] for n in atombase]
    atom_arr = [atom_id[n] for n in atombase]
    at_eps = [EPSILON[n] for n in atombase]
    # convert to tensors
    # Residues level
    res_id = np.arange(0, num_residues, dtype=np.float32)
    res_xyz = np.array(caxyz, dtype=np.float32)
    res_dist = cdist(res_xyz, res_xyz)
    res_cb =  np.array(cbxyz, dtype=np.float32)
    # Atomic level
    at_xyz = atomxyz
    at_dist = cdist(at_xyz, at_xyz)
    at_id = np.array(at_eps, dtype=np.int32)
    sigma = np.array(at_vdw, dtype=np.float32)
    epsilon = np.array(at_eps, dtype=np.float32)
    # check variuos atom/residue types
    #is atom from side chain
    is_side_chain = is_atom_in_group(atomname, BACKBONE)
    # hydrophobic
    is_res_hf = is_atom_in_group(resname, HYDROPHOBIC)
    # hydrogen bonds
    is_res_hf = is_atom_in_group(resname, HYDROPHOBIC)
    # hydrogen bonds
    is_at_hb_a = np.zeros(num_atoms, dtype=np.bool)
    is_at_hb_d = np.zeros(num_atoms, dtype=np.bool)
    is_at_hb_ac = [True if at.startswith("C") else False for at in atomname]
    is_at_hb_ad = [True if at.startswith('NH') else False for at in atomname]
    is_res_ar = is_atom_in_group(resname, AROMATIC)
    is_res_cpi = is_atom_in_group(atomname, CATION_PI)
    is_res_arg = is_atom_in_group(resname, {'ARG'})
    # salt bridge
    is_at_sb_c1 = is_atom_in_group(atomname, SALT_BRIDGE_C1)
    is_res_sb_c1 = is_atom_in_group(resname, {'ARG', 'LYS'})
    is_at_sb_c2 = is_atom_in_group(resname, {'ARG', 'GLU'})
    is_res_sb_c2 = is_atom_in_group(resname, SALT_BRIDGE_C2)
    # van der Waals
    is_at_vdw_other = np.zeros(num_atoms, dtype=np.bool)

    for i, (res, at) in enumerate(zip(resname, atomname)):
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
    is_at_vdw  = is_atom_in_group(atombase, {'C', 'S'})

    ### conditions 
    is_at_hb_ac = np.asarray(is_at_hb_ac, dtype=np.bool)
    is_at_hb_ad = np.asarray(is_at_hb_ad, dtype=np.bool)
    at_is_side = is_side_chain
    # hbonds acceptor donors
    at_is_hba = is_at_hb_a | is_at_hb_ac
    at_is_hbd = is_at_hb_d | is_at_hb_ad
    #vdw
    at_is_vdw = is_at_vdw | is_at_vdw_other
    at_dist_inv = 1/(at_dist + 1e-6)
    # set inverse of the atom self distance to zero to avoid nan/inf when summing
    np.fill_diagonal(at_dist_inv, 0) 
    atat_charge = np.array(at_charge, dtype=np.float).reshape(-1, 1)
    atat_charge = atat_charge * np.swapaxes(atat_charge, 0, 1)
    sigma_coeff = (sigma.reshape(-1, 1) + sigma.reshape(1, -1))/2
    sigma_radii = (sigma.reshape(-1, 1) + sigma.reshape(1, -1))
    epsilon = np.sqrt(epsilon.reshape(-1, 1) * epsilon.reshape(1, -1))
    
    lj_r = sigma_coeff*at_dist_inv * (at_dist < 10)
    lj6 = lj_r**6
    lj12 = lj_r**12
    # binary interactions
    disulfde = (at_id == 4) & (at_dist < 2.2)
    hydrophobic = (at_dist < 5.0) & (at_is_side == False) & is_res_hf
    cation_pi = (at_dist < 6) & is_res_cpi
    arg_arg = (at_dist < 5.0) & is_res_arg
    vdw = ((at_dist - sigma_radii) < 0.5) & at_is_vdw
    # hbonds
    hbond = at_is_hba.reshape(-1, 1) & at_is_hba.reshape(1, -1) & (at_dist < 3.5)
    # salt bridge
    sb_tmp1 = is_at_sb_c1.reshape(-1, 1) & is_at_sb_c2.reshape(1, -1)
    sb_tmp2 = is_res_sb_c1.reshape(-1, 1) & is_res_sb_c2.reshape(1, -1)
    salt_bridge = sb_tmp1 & (at_dist < 5.0) & sb_tmp2
    
    feats = np.concatenate((disulfde[:,:,np.newaxis],
                   hydrophobic[:,:,np.newaxis],
                   cation_pi[:,:,np.newaxis],
                   arg_arg[:,:,np.newaxis],
                   salt_bridge[:,:,np.newaxis],
                   hbond[:,:,np.newaxis],
                   vdw[:,:,np.newaxis]), axis=2)
    #feats = feats.float()
    coulomb_energy =  (1.0/3.14*EPS) * atat_charge * at_dist_inv
    lenard_jones_energy = epsilon* (lj12 - lj6) 
    energy_sum = coulomb_energy + lenard_jones_energy
    feats = np.concatenate((feats, 
                    coulomb_energy[:,:,np.newaxis],
                   lenard_jones_energy[:,:,np.newaxis],
                   energy_sum[:,:,np.newaxis]),axis=2)
    # change feature resolution
    # from atomic level to residue level
    efeat_list = list()
    first_dim_split = np.array_split(feats, resatnum, axis=0)
    for i in range(len(resatnum)):
        efeat_list.extend(list(np.array_split(first_dim_split[i], resatnum, axis=1)))
    u, v = np.where(res_dist < t)
    uv = np.where(res_dist < t)[0]
    feats_at = np.concatenate([efeat_list[e].sum((0,1), keepdims=True) for e in uv], axis=0)
    efeats = np.zeros_like(res_dist)
    # gather residue level feature, such as edge criteria
    cb1 = np.linalg.norm(res_cb - res_xyz, axis=1, keepdims=True)
    cb_dist = cdist(res_cb, res_cb)
    cb2 = np.swapaxes(cb1.copy(), 0, 1)
    tn_cb12 = cb1 / (cb2 + 1e-5)
    tn_cb12[np.isnan(tn_cb12)] = 0
    inv_ca12 = 1/(res_dist + 1e-5)
    np.fill_diagonal(inv_ca12, 0)
    res_id_short = np.arange(0, res_id.max()+1, 1)
    is_seq = np.abs(res_id_short[np.newaxis, ] - res_id_short[:,np.newaxis])
    is_self = is_seq == 0
    is_seq_0 = is_seq == 1
    is_seq_1 = is_seq > 5
    is_struct_0 = is_seq > 1
    is_caca_cbcb = cb_dist < res_dist
    feats_res = np.concatenate((tn_cb12[:,:,np.newaxis],
                       inv_ca12[:,:,np.newaxis],
                        is_caca_cbcb[:,:,np.newaxis],
                        is_self[:,:,np.newaxis],
                       is_seq_0[:,:,np.newaxis],
                       is_seq_1[:,:,np.newaxis],
                       is_struct_0[:,:,np.newaxis]), axis=2)
    #print(feats_res.shape, u, v)
    feats_res = feats_res[u,v]
    feats_all = np.concatenate((feats_at.squeeze(), feats_res), axis=-1)
    
    return u, v, feats_all


def calc_named(pdb_loc: str, chain, t: int = 9) -> pd.DataFrame:
    '''
    calculate structural features as dataframe
    '''
    name_i = ['disulfide', 'hydrophobic', 'cation_pi', 'arg_arg', 'salt_bridge', 'hbond', 'vdw']
    name_i += ['c', 'lj', 'e']
    name_i += ['cbcb', '1/caca', 'ca_vs_cb', 'self', 'is_seq', 'is_seq_not', 'is_struct']
    structure = atomium.open(pdb_loc).model
    pdb_ids = atomium_chain_pdb_list(structure, raw_id=True)
    residues = atomium_select(structure, None, pdb_ids)
    u, v, feats = read_struct(residues, chain, t)
    '''
    dataframe = pd.DataFrame(data=feats.numpy(), columns=name_i)
    dataframe['res1'] = u.numpy()
    dataframe['res2'] = v.numpy()
    '''
    dataframe=None
    return dataframe

