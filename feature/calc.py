import torch as th
from params import ATAT_INT, num_atat_feats
import atomium

def read_struct(pdb_loc, chain, t):
    
    data = atomium.open(pdb_loc).model
    atoms = []
    name = []
    CA_xyz = []
    residues = []
    i = 0
    for res in data.chain(chain).residues():
        for atom in res.atoms():
            n = atom.name
            if n == 'CA':
                CA_xyz.append(atom.location)
            elif len(n) == 3:
                n = n[:2]
            name.append(n)
            atoms.append(atom.location)
            residues.append(i)
            i += 1
            
    AT_xyz = th.FloatTensor(atoms)
    AT_dist = th.cdist(AT_xyz, AT_xyz)
    CA_dist = th.cdist(CA_xyz, CA_xyz)
    
    num_resid = len(CA_xyz)
    xyz = th.FloatTensor(atoms)
    dist = th.cdist(xyz, xyz)
    res_num = th.LongTensor(residues)
    atom_id = [AT_INT[at] for at in name]
    atom_id = th.LongTensor(atom_id)
    dist_where = (dist < 2)
    res1, res2 = th.nonzero(dist_where, as_tuple=True)
    aa1, aa2 = res_num[res1], res_num[res2]
    at1, at2 = atom_id[res1], atom_id[res2]
    inter = aa1 != aa2
    inter1, inter2 = aa1[inter], aa2[inter]
    
    print('num:', inter.shape)
    bond_list = list()
    bond_dict = {i : [] for i in range(num_resid)}
    for i in range(inter1.shape[0]-1):
        rid1, rid2 = inter1[i].item(), inter2[i].item()
        atid1, atid2 = at1[i].item(), at2[i].item()
        #print(rid1, ' - ', rid2, ': ', INT_AT[atid1], '-',  INT_AT[atid2])
        atat_name = INT_AT[atid1]+'-'+INT_AT[atid2]
        bond_list.append(atat_name)
        bond_dict[rid1].append(atat_name)
        
    bond_feat = bond_to_vector(bond_dict)
    return CA_dist, bond_feat


def bond_to_vector(bond_dict):
    stack = list()
    for res, bond_list in bond_dict.items():
        
        bonds_unique = set(bond_list)
        bonds = th.zeros(num_atat_feats)
        
        if bonds_unique:
            
            bonds_id = [ATAT_INT[b] for b in bonds_unique if b in ATAT_INT]
            bonds[bonds_id] = 1
        stack.append(bonds.unsqueeze(0))
    feats = th.cat(stack, 0)
    return feats