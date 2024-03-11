'''script for calculating residue - residue interactions'''
from typing import Union, Tuple, List, Optional

import sys
sys.path.append('..')
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import torch as th
from torch import linalg as LA
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
@th.jit.script
def distance(xyz1: th.Tensor, xyz2: Optional[th.Tensor] = None):

    if xyz2 is not None:
        return th.cdist(xyz1, xyz2)
    else:
        return (xyz1.unsqueeze(0) - xyz1.unsqueeze(1)).pow(2).sum(-1).sqrt()


@th.jit.script
def dihedral(n: th.Tensor, ca: th.Tensor, c: th.Tensor):
    """
    Args:
        nitrogen: torch.Tensor [num_atoms, xyz]
        carbon_alpha: torch.Tensor
        carbon: torch.Tensor
    N - [ CA - C - N - CA ] - C
             b1  b2   b3  vectors 
               n1   n2    planes
    Returns:
        torch.FloatTensor: phi
        torch.FloatTensor: psi
    """

    b1 = ca - c
    b2 = c.roll(1) - n.roll(1)
    b3 = n.roll(1) - ca.roll(1)
    n1 = th.cross(b1, b2)
    n1 /= LA.vector_norm(n1, ord=2, dim=1, keepdim=True)
    n2 = th.cross(b2, b3)
    n2 /= LA.vector_norm(n2, ord=2, dim=1, keepdim=True)
    # normalize b2 
    #b2 /= LA.vector_norm(b2, ord=2, dim=1, keepdim=True)
    #m1 = th.cross(n1, b2)
    #x = (n1 * n2).sum(1)
    #y = (m1 * n2).sum(1)
    b_cross23 = th.cross(b2, b3)
    b_cross12 = th.cross(b1, b2)
    b2_norm = LA.vector_norm(b2, ord=2, dim=1, keepdim=True)
    b_cross1223 = (b_cross12*b_cross23).sum(1, keepdim=True).sqrt()
    b21 = b2_norm*(b1*b_cross23).sum(1, keepdim=True).sqrt()
    dihedral = th.atan2(b21, b_cross1223)
    phi = dihedral
    psi = dihedral.roll(1)
    # fill borders
    phi[0] = 0
    psi[0] = 0
    return phi, psi


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

def is_atom_in_group(atoms: List[str], group: set, size: int) -> th.BoolTensor:
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

def read_struct(pdb_loc: str,
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
    atoms, name = list(), list()
    ca_xyz, cb_xyz = list(), list()
    cg_xyz = list()
    c_xyz, n_xyz = list(), list()
    residues, residues_name = list(), list()
    res_per_res = list()
    is_side_chain = list()
    res_at_num = list()
    skip_c = 0
    for resi, (_, residue) in enumerate(data.groupby(['chain_id', 'residue_number'])):
        missing_cd = True
        missing_cg = True
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
            elif atom.atom_name == "C":
                c_xyz.append((atom.x_coord, atom.y_coord, atom.z_coord))
            elif atom.atom_name == 'N':
                n_xyz.append((atom.x_coord, atom.y_coord, atom.z_coord))
            elif atom.atom_name == 'CG':
                missing_cg = False
                cg_xyz.append((atom.x_coord, atom.y_coord, atom.z_coord))
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

    # assign parameters to atoms
    num_atoms = len(name)
    num_residues = len(res_per_res)
    #print('skipped: ', skip_c)

    name_base = [n[0] for n in name]
    #at_charge = [CHARGE[n] for n in name_base]
    at_vdw = [SIGMA[n] for n in name_base]
    #atom_arr = [atom_id[n] for n in name_base]
    at_eps = [EPSILON[n] for n in name_base]
    # convert to tensors
    res_ca = th.FloatTensor(ca_xyz)
    res_dist = distance(res_ca)
    res_cb = th.FloatTensor(cb_xyz)
    res_cg = th.FloatTensor(cg_xyz)
    res_c = th.FloatTensor(c_xyz)
    res_n = th.FloatTensor(n_xyz)
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
    is_res_ar = is_atom_in_group(residues_name, AROMATIC, num_atoms)
    is_res_cpi = is_atom_in_group(name, CATION_PI, num_atoms)
    is_res_arg = is_atom_in_group(residues_name, 'ARG', num_atoms)
    # salt bridge
    is_at_sb_c1 = is_atom_in_group(name, SALT_BRIDGE_C1, num_atoms)
    is_res_sb_c1 = is_atom_in_group(residues_name, {'ARG', 'LYS'}, num_atoms)
    is_at_sb_c2 = is_atom_in_group(name, {'ARG', 'GLU'}, num_atoms)
    is_res_sb_c2 = is_atom_in_group(residues_name, SALT_BRIDGE_C2, num_atoms)
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
    at_dist = distance(at_xyz)
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
    # change feature resolution
    # residue level features
    #breakpoint()
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
    cb_dist = distance(res_cb)
    res_id_short = th.arange(0, num_residues)
    is_seq = th.abs(res_id_short.unsqueeze(0) - res_id_short.unsqueeze(1))
    is_self = is_seq == 0
    is_seq_0 = is_seq == 1
    is_seq_1 = is_seq > 5
    is_struct_0 = is_seq > 1
    
    feats_res = th.stack((is_self,
                       is_seq_0,
                       is_seq_1,
                       is_struct_0), dim=2)
    feats_res = feats_res[u,v]
    # geometic and others
    phi, psi = dihedral(res_n, res_ca, res_c)
    chi1, chi2 = dihedral(res_ca, res_cb, res_cg)
    distmx = th.stack((res_dist, cb_dist), dim=2)
    nfeats = th.stack((phi, psi, chi1, chi2), dim=1).squeeze(-1)
    efeats = th.cat((feats_at.float().squeeze(), feats_res), dim=-1)
    return u, v, efeats, nfeats, res_per_res, distmx



def edge_embedding_to_3d_tensor(u,v,emb):
    num_nodes = u.max()+1
    contacts = th.zeros((num_nodes, num_nodes, emb.shape[-1]))
    emb_normed = emb/emb.norm(2, dim=1, keepdim=True)
    contacts[u,v] = emb_normed
    return contacts

if __name__ == "__main__":
    file = "../tests/data/3sxw.pdb.gz"
    _, _, _, _ = read_struct(file, "A", t=7)