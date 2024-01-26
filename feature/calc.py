'''script for calculating residue - residue interactions'''
from typing import Union, Tuple, List, Optional

import sys
sys.path.append('..')
import atomium
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import torch as th
from biopandas.pdb import PandasPdb
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
    from .. parse import atomium_select, atomium_chain_pdb_list
    from .. parse import read_pdb_full
except ImportError as e:
    print(e)

aa_trans = {
    'MSE': 'MET',  # Methionine Selenomethionine
    'CYX': 'CYS',  # Cystine
    'SEC': 'CYS',  # Selenocysteine
    'PYL': 'LYS',  # Pyrrolysine
    'ALM': 'ALA',  # Alanine with added methyl group
    'CME': 'CYS',  # S,S-(2-hydroxyethyl)thiocysteine
    'CSO': 'CYS',  # S-Hydroxycysteine
    'OCS': 'CYS',  # Cysteic acid
    'SEP': 'SER',  # Phosphoserine
    'TPO': 'THR',  # Phosphothreonine
    'PTR': 'TYR',  # Phosphotyrosine
    # ...
}

def map_aa_name(resname):
    """
    map residue residue names into standrad ones
    """
    if resname in aa_trans:
        return aa_trans[resname]
    else:
        return aa_trans


nan_type = float('nan')
nan_xyz = (nan_type, nan_type, nan_type)
atom_id = {ch : i for i, ch in enumerate(CHARGE.keys())}
EPS = th.Tensor([78*1e-2]) # unit Farad / angsterm
CLIP_MAX = th.FloatTensor([1,1,1,1,1,1,1,10,10,10,10,1,1,1,1,1,1])
CLIP_MIN = th.FloatTensor([0,0,0,0,0,0,0,-10,-10,-10,1e-20,0,0,0,0,0,0])
side_chain_atom_names = ['CA', 'C', 'N', 'O']
invalid_location = {'B','C','D'}
FEATNAME = [
    'disulfide', 'hydrophobic', 'cation_pi', 'arg_arg', 'salt_bridge', 'hbond', 'vdw',
    #'cx','cy','cz', 'ljx', 'ljy', 'ljz',
    '1/caca', 'ca_vs_cb', 'self', 'is_seq', 'is_seq_not', 'is_struct'
]

def fill_missing_part_by_index(missing_index: int,
                               biopandas_df: str,
                               chain: Optional[str] = None):
    hetdf = biopandas_df.df['HETATM'] if chain is None else biopandas_df.df['HETATM'].loc[biopandas_df.df['HETATM']['chain_id'] == chain]
    missing_df = hetdf[hetdf['residue_number'] == missing_index]
    if missing_df.empty:
        raise KeyError(f"missing residue {missing_index} not found")
    # convert miss to df
    missing_df['record_name'] = 'ATOM'
    missing_df['residue_name'] = missing_df['residue_name'].apply(lambda x: aa_trans[x] if x in aa_trans.keys() else x)
    return missing_df

def is_atom_in_group(atoms: List[str], group: set, size: int):
    arr = th.zeros(size, dtype=th.bool)
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
                chain: Optional[str] = None,
                t: Optional[float] = None,
                indices_to_read: Optional[list] = None
                ) -> Tuple[th.Tensor]:
    '''
    params:
        pdb_loc (str, set, atomium.Model): path to structure, atomium selection 
        chain (str): one letter pdb code 
        t (float): contact distance threshold
    return u, v for feats
    '''
    if isinstance(pdb_loc, str):
        data = PandasPdb().read_pdb(pdb_loc)#
    # elif isinstance(pdb_loc, atomium.structures.Model):
    #     data = pdb_loc.residues()
    # elif isinstance(pdb_loc, list):
    #     data = pdb_loc
    else:
        raise KeyError(f'wrong pdb_loc type {type(pdb_loc)}')
    if not isinstance(t, (int, float, type(None))):
        raise ValueError(f'threshold must be number or None, given: {type(t)}')
    else: 
        if isinstance(t, (int, float)) and t < 5:
            print('dumb threshold')
    atoms = data.df['ATOM']
    hetatm = data.df['HETATM']
    hetatm = hetatm[hetatm.residue_name.isin(aa_trans)]
    hetatm['residue_name'] = hetatm['residue_name'].apply(map_aa_name)
    #hetatm = hetatm[hetatm.residue_name == 'MSE']
    data = pd.concat([atoms, hetatm], axis=0)
    data.sort_values(['chain_id','residue_number', 'atom_number'], inplace=True)
    
    if chain is not None:
        data = data.loc[data['chain_id'] == chain].copy()
        if data.shape[0] == 0:
            raise ValueError(f"no data in chain {chain}")
    if indices_to_read is not None:
        data = data.loc[data['residue_number'].isin(indices_to_read)].copy()
    atoms, name = [], []
    ca_xyz, cb_xyz = [], []
    residues, residues_name = [], []
    res_per_res = list()
    is_side_chain = []
    res_at_num = []
    skip_c = 0
    for resi, (_, residue) in enumerate(data.groupby(['chain_id', 'residue_number'])):
        missing_cb = True
        missing_ca = True
        num_atoms = residue.shape[0]
        res_at_num.append(num_atoms)
        residues.append(resi)
        res_per_res.append(residue.iloc[0].residue_name)
        for _, atom in residue.iterrows():
            n = atom.atom_name
            if atom.alt_loc in invalid_location:
                skip_c += 1
                continue
            if atom.atom_name == 'CA':
                ca_xyz.append((atom.x_coord, atom.y_coord, atom.z_coord))
                missing_ca = False
            elif atom.atom_name == 'CB':
                cb_xyz.append((atom.x_coord, atom.y_coord, atom.z_coord))
                missing_cb = False
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

    # assign parameters to atoms
    num_atoms = len(name)
    num_residues = len(res_per_res)
    #print('skipped: ', skip_c)

    name_base = [n[0] for n in name]
    at_charge = [CHARGE[n] for n in name_base]
    at_vdw = [SIGMA[n] for n in name_base]
    atom_arr = [atom_id[n] for n in name_base]
    at_eps = [EPSILON[n] for n in name_base]
    # convert to tensors
    res_id = th.LongTensor(residues)
    res_ca = th.FloatTensor(ca_xyz)
    res_dist = th.cdist(res_ca, res_ca)
    res_cb = th.FloatTensor(cb_xyz)
    # print(minlength, chainlength)
    if num_residues != res_ca.shape[0]:
        raise ValueError(f"number of residues is different then CA atoms {num_residues} and {res_ca.shape[0]}")
    # check variuos atom/residue types
    # hydrophobic
    is_res_hf = [True if r in HYDROPHOBIC else False for r in residues_name]
    # hydrogen bonds
    is_at_hb_a = [False]*num_atoms
    is_at_hb_d = [False]*num_atoms
    is_at_hb_ac = [True if at.startswith("C") else False for at in name]
    is_at_hb_ad = [True if at.startswith('NH') else False for at in name]
    is_res_ar = [True if r in AROMATIC else False for r in residues_name]
    is_res_cpi = [True if at in CATION_PI else False for at in name]
    is_res_arg = [True if r in {'ARG'} else False for r in residues_name]
    # salt bridge
    is_at_sb_c1 = [True if at in SALT_BRIDGE_C1 else False for at in name]
    is_res_sb_c1 = [True if at in {'ARG', 'LYS'} else False for at in residues_name]
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
    at_is_side = th.BoolTensor(is_side_chain)
    # hbonds acceptor donors
    at_is_hba = th.BoolTensor(is_at_hb_a) | th.BoolTensor(is_at_hb_ac)
    at_is_hbd = th.BoolTensor(is_at_hb_d) | th.BoolTensor(is_at_hb_ad)
    #vdw
    at_is_vdw = th.BoolTensor(is_at_vdw) | th.BoolTensor(is_at_vdw_other)
    # set inverse of the atom self distance to zero to avoid nan/inf when summing
    sigma_radii = (sigma.view(-1, 1) + sigma.view(1, -1))
    '''
    at_dist_inv = 1/(at_dist + 1e-6)
    at_dist_inv.fill_diagonal_(0) 
    atat_charge = th.FloatTensor(at_charge)
    atat_charge = atat_charge.view(-1, 1) * atat_charge.view(1, -1)
    sigma_coeff = (sigma.view(-1, 1) + sigma.view(1, -1))/2
    
    epsilon = th.sqrt(epsilon.view(-1, 1) * epsilon.view(1, -1))
    
    lj_r = sigma_coeff*at_dist_inv * (at_dist < 10)
    lj6 = th.pow(lj_r, 6) 
    lj12 = th.pow(lj_r, 12)
    '''
    # binary interactions
    disulfde = (at_id == 4) & (at_dist < 2.2)
    hydrophobic = (at_dist < 5.0) & (at_is_side == False) & th.BoolTensor(is_res_hf)
    cation_pi = (at_dist < 6) & th.BoolTensor(is_res_cpi)
    arg_arg = (at_dist < 5.0) & th.BoolTensor(is_res_arg)
    vdw = ((at_dist - sigma_radii) < 0.5) & at_is_vdw
    # hbonds
    hbond = at_is_hba.view(-1, 1) & at_is_hbd.view(1, -1) & (at_dist < 3.5)
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
    '''
    feats = feats.float()
    coulomb_energy =  (1.0/3.14*EPS) * atat_charge * at_dist_inv
    lenard_jones_energy = epsilon* (lj12 - lj6) 
    energy_sum = coulomb_energy + lenard_jones_energy
    '''
    # change feature resolution
    # residue level features
    efeat_list = list()
    first_dim_split = feats.split(res_at_num, 0)
    for i in range(len(res_at_num)):
        efeat_list.extend(list(first_dim_split[i].split(res_at_num, 1)))
    if t is None:
        t = res_dist.max() + 1
    u, v = th.where(res_dist < t)
    uv = th.where(res_dist < t)[0]
    # sum only for residues within threshold
    feats_at = th.cat([efeat_list[e].sum((0,1), keepdim=True) for e in uv], dim=0)
    th.clamp(feats_at, min=0, max=1, out=feats_at)
    # gather residue level feature, such as edge criteria
    cb_dist = th.cdist(res_cb, res_cb)
    inv_ca12 = 1/(res_dist + 1e-5)
    inv_ca12.fill_diagonal_(0)
    res_id_short = th.arange(0, num_residues)
    is_seq = th.abs(res_id_short.unsqueeze(0) - res_id_short.unsqueeze(1))
    is_self = is_seq == 0
    is_seq_0 = is_seq == 1
    is_seq_1 = is_seq > 5
    is_struct_0 = is_seq > 1
    is_caca_cbcb = cb_dist < res_dist
    # inv_ca12 = inv_ca12[(~th.isnan(inv_ca12).any(axis=1)) and (~th.isnan(inv_ca12).any(axis=0))] 
    is_caca_cbcb = is_caca_cbcb[~th.isnan(is_caca_cbcb).any(axis=1)]
    # print(data.residue_number.head(1), data.residue_number.tail(1))
    # print(f" inv_ca12: {inv_ca12.shape}, is_caca_cbcb: {is_caca_cbcb.shape}, is_self: {is_self.shape}, is_seq_0: {is_seq_0.shape}, is_seq_1: {is_seq_1.shape}, is_struct_0: {is_struct_0.shape}")
    feats_res = th.cat((inv_ca12.unsqueeze(2),
                        is_caca_cbcb.unsqueeze(2),
                        is_self.unsqueeze(2),
                       is_seq_0.unsqueeze(2),
                       is_seq_1.unsqueeze(2),
                       is_struct_0.unsqueeze(2)), dim=2)
    feats_res = feats_res[u,v]
    feats_all = th.cat((feats_at.float().squeeze(), feats_res), dim=-1)
    return u, v, feats_all, res_per_res


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
    res_id = th.arange(0, num_residues)
    res_ca = th.from_numpy(caxyz)
    res_dist = th.cdist(res_ca, res_ca)
    res_cb = th.from_numpy(cbxyz)
    # Atomic level
    at_xyz = th.from_numpy(atomxyz)
    at_dist = th.cdist(at_xyz, at_xyz)
    at_id = th.LongTensor(at_eps)
    sigma = th.FloatTensor(at_vdw)
    epsilon = th.FloatTensor(at_eps)
    # check variuos atom/residue types
    #is atom from side chain
    at_is_side = is_atom_in_group(atomname, BACKBONE, num_atoms)
    # hydrophobic
    is_res_hf = is_atom_in_group(resname, HYDROPHOBIC, num_atoms)
    # hydrogen bonds
    is_res_hf = is_atom_in_group(resname, HYDROPHOBIC, num_atoms)
    # hydrogen bonds
    is_at_hb_a = th.zeros(num_atoms, dtype=th.bool)
    is_at_hb_d = th.zeros(num_atoms, dtype=th.bool)
    is_sulphur = [True if at.startswith('S') else False for at in atomname]
    is_at_hb_ac = [True if at.startswith("C") else False for at in atomname]
    is_at_hb_ad = [True if at.startswith('NH') else False for at in atomname]
    is_res_ar = is_atom_in_group(resname, AROMATIC, num_atoms)
    is_res_cpi = is_atom_in_group(atomname, CATION_PI, num_atoms)
    is_res_arg = is_atom_in_group(resname, {'ARG'}, num_atoms)
    # salt bridge
    is_at_sb_c1 = is_atom_in_group(atomname, SALT_BRIDGE_C1, num_atoms)
    is_res_sb_c1 = is_atom_in_group(resname, {'ARG', 'LYS'}, num_atoms)
    is_at_sb_c2 = is_atom_in_group(resname, {'ARG', 'GLU'}, num_atoms)
    is_res_sb_c2 = is_atom_in_group(resname, SALT_BRIDGE_C2, num_atoms)
    # van der Waals
    is_at_vdw_other = th.zeros(num_atoms, dtype=th.bool)
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
    is_at_vdw  = is_atom_in_group(atombase, {'C', 'S'}, num_atoms)
    is_at_sulphur = th.BoolTensor(is_sulphur)
    is_at_hb_ac = th.BoolTensor(is_at_hb_ac)
    is_at_hb_ad = th.BoolTensor(is_at_hb_ad)
    # hbonds acceptor donors
    at_is_hba = is_at_hb_a | is_at_hb_ac
    at_is_hbd = is_at_hb_d | is_at_hb_ad
    #vdw
    at_is_vdw = is_at_vdw | is_at_vdw_other
    atat_charge = th.FloatTensor(at_charge)
    atat_charge = atat_charge.view(-1, 1) * atat_charge.view(1, -1)
    sigma_coeff = (sigma.view(-1, 1) + sigma.view(1, -1))/2
    sigma_radii = (sigma.view(-1, 1) + sigma.view(1, -1))
    epsilon = th.sqrt(epsilon.view(-1, 1) * epsilon.view(1, -1))
    # binary interactions
    disulfde = is_at_sulphur & (at_dist < 3)
    hydrophobic = (at_dist < 5.0) & ~at_is_side & is_res_hf
    cation_pi = (at_dist < 6) & is_res_cpi
    arg_arg = (at_dist < 5.0) & is_res_arg
    vdw = ((at_dist - sigma_radii) < 0.5) & at_is_vdw
    # hbonds
    hbond = at_is_hba.view(-1, 1) & at_is_hbd.view(1, -1) & (at_dist < 3.5)
    # salt bridge
    sb_tmp1 = is_at_sb_c1.view(-1, 1) & is_at_sb_c2.view(1, -1)
    sb_tmp2 = is_res_sb_c1.view(-1, 1) & is_res_sb_c2.view(1, -1)
    salt_bridge = sb_tmp1 & (at_dist < 5.0) & sb_tmp2
    
    feats = th.cat((disulfde.unsqueeze(2),
                   hydrophobic.unsqueeze(2),
                   cation_pi.unsqueeze(2),
                   arg_arg.unsqueeze(2),
                   salt_bridge.unsqueeze(2),
                   hbond.unsqueeze(2),
                   vdw.unsqueeze(2)), dim=2)
    feats = feats.float()
    # change feature resolution
    # from atomic level to residue level
    efeat_list = list()
    first_dim_split = feats.split(resatnum, 0)
    for i in range(len(resatnum)):
        efeat_list.extend(list(first_dim_split[i].split(resatnum, 1)))
    u, v = th.where(res_dist < t)
    uv = th.where(res_dist < t)[0]
    # sum only for residues within threshold
    feats_at = th.cat(tuple(efeat_list[e].sum((0,1), keepdim=True) for e in uv), dim=0)
    feats_at 
    # gather residue level feature, such as edge criteria
    cb_dist = th.cdist(res_cb, res_cb)
    inv_ca12 = 1/(res_dist + 1e-5)
    inv_ca12.fill_diagonal_(0)
    res_id_short = th.arange(0, res_id.max()+1, 1)
    is_seq = th.abs(res_id_short.unsqueeze(0) - res_id_short.unsqueeze(1))
    is_self = is_seq == 0
    is_seq_0 = is_seq == 1
    is_seq_1 = is_seq > 5
    is_struct_0 = is_seq > 1
    is_caca_cbcb = cb_dist < res_dist
    feats_res = th.cat((inv_ca12.unsqueeze(2),
                        is_caca_cbcb.unsqueeze(2),
                        is_self.unsqueeze(2),
                       is_seq_0.unsqueeze(2),
                       is_seq_1.unsqueeze(2),
                       is_struct_0.unsqueeze(2)), dim=2)
    feats_res = feats_res[u,v]
    feats_all = th.cat((feats_at.squeeze(), feats_res), dim=-1)
    #feats_all = th.where(feats_all < CLIP_MIN, CLIP_MIN, feats_all)
    #feats_all = th.where(feats_all > CLIP_MAX, CLIP_MAX, feats_all)
    return u, v, feats_all


def calc_named(pdb_loc: str, chain, t: float = 9) -> pd.DataFrame:
    '''
    calculate structural features as dataframe
    Params:
        pdb_loc (str) path to pdb file
        chain (str or None) chain to use if None all chains are used
        t (float) threshold to ca-ca distance
    '''
    
    structure = atomium.open(pdb_loc).model
    pdb_ids = atomium_chain_pdb_list(structure, raw_id=True)
    residues = atomium_select(structure, None, pdb_ids)
    u, v, feats = read_struct(residues, chain, t)

    dataframe = pd.DataFrame(data=feats.numpy(), columns=FEATNAME)
    dataframe['res1'] = u.numpy()
    dataframe['res2'] = v.numpy()
    return dataframe

def calculate_interactions(path_pdb: str, t: float = 9) -> pd.DataFrame:

    structdata = read_pdb_full(path_pdb)
    u, v, feats = calc_struct_properties(*structdata, t=t)
    dataframe = pd.DataFrame(data=feats.numpy(), columns=FEATNAME)
    dataframe['res1'] = u.numpy()
    dataframe['res2'] = v.numpy()
    return dataframe

def edge_embedding_to_3d_tensor(u,v,emb):
    num_nodes = u.max()+1
    contacts = th.zeros((num_nodes, num_nodes, emb.shape[-1]))
    emb_normed = emb/emb.norm(2, dim=1, keepdim=True)
    contacts[u,v] = emb_normed
    return contacts

