import os
from functools import partial
from collections import namedtuple

import torch as th
import numpy as np
import atomium
import Bio.SeqUtils as bios
from scipy.spatial import distance_matrix

protein_letters_3to1 = bios.IUPACData.protein_letters_3to1_extended
protein_letters_3to1 = {k.upper() : v for k,v in protein_letters_3to1.items()}
protein_letters = 'ACDEFGHIKLMNPQRSTVWY' + 'X'
residue_to_num_dict = {res : num for num, res in enumerate(protein_letters)}
map_residue = lambda res : residue_to_num_dict[res.code]

protein_struct = namedtuple('protein_struct', ['path', 'chain', 'xyz', 'seq', 'ss', 'pdb_list'])

def get_atom_xyz(residue, atom_name):
    for a in residue.atoms():
        if a.name == atom_name:
            return a.location
    return (np.nan, np.nan, np.nan)

def get_ss_label(residue):
    '''
    E, H or C label from atomium
    '''
    if residue.helix:
        return 'H'
    elif residue.strand:
        return 'E'
    else:
        return 'C'

    
get_CA_xyz = partial(get_atom_xyz, atom_name='CA')
def parse_graph_data(path, chain):
    
    if not os.path.isfile(path):
        FileNotFoundError('no such file', path)
    file = atomium.open(path)
    chain = file.model.chain(chain)
    preparation_dict = dict()
    
    for i, r in enumerate(chain.residues()):
        r_atoms = r.atoms()
        try:
            res_one_letter =  protein_letters_3to1[r.name]
        except:
            res_one_letter = 'X'
        preparation_dict[i] = {'aa' : res_one_letter,
                                    'charge' : r.charge,
                                    'CA' : get_atom_xyz(r_atoms, 'CA'),
                                    'CB' : get_atom_xyz(r_atoms, 'CB'),
                                    'ss_label' : get_ss_label(r)
                                   }

        ca_xyz = np.asarray(list(map(lambda v : v['CA'], preparation_dict.values())))
        sequence = list(map(lambda v : v['aa'], preparation_dict.values()))
        ss = list(map(lambda v : v['ss_label'], preparation_dict.values()))
        
        ca_ca_matrix = distance_matrix(ca_xyz, ca_xyz)
    return ca_ca_matrix, sequence, ss

def parse_graph_data_torch(path, pdb_chain):
    '''
    parses specified PDB file and computes
    ca_ca_matrix and sequence as 1-21 numbers
    '''
    if not path.is_file():
        FileNotFoundError('no such file', path)
    path_str = str(path)
    file = atomium.open(path_str)
    chain = file.model.chain(pdb_chain)

    if chain is None:
        KeyError(f'no chain: {chain} for {path}')
    sequence = list(map(map_residue, chain.residues()))
    ca_xyz = list(map(get_CA_xyz, chain.residues()))
    

    ca_xyz = th.FloatTensor(ca_xyz)
    sequence = th.LongTensor(sequence)
    ca_ca_matrix = th.cdist(ca_xyz, ca_xyz)
    
    return ca_ca_matrix, sequence

def parse_sequence(path, pdb_chain):
    
    if not path.is_file():
        FileNotFoundError('no such file', path)
    path_str = str(path)
    file = atomium.open(path_str)
    chain = file.model.chain(pdb_chain)

    if chain is None:
        KeyError(f'no chain: {chain} for {path}')
    sequence = list(map(lambda res : res.code, chain.residues()))
    return sequence


def parse_xyz(path, chain, get_pdb_ss=False):

    
    if not os.path.isfile(path):
        FileNotFoundError('no such file', path)
        
    file = atomium.open(path)
    chain = file.model.chain(chain)
    sequence = list(map(lambda x: x.code, chain.residues()))
    ca_xyz = list(map(get_CA_xyz, chain.residues()))
    ca_xyz = th.FloatTensor(ca_xyz)
    if get_pdb_ss:
        secondary = list(map(get_ss_label, chain.residues()))
        return ca_xyz, sequence, secondary
    else:
        return ca_xyz, sequence
    
    
    
def parse_pdb_indices(path, chain):
    
    pdb_list = [s.id.split('.')[1] for s in atomium.open(path).model.chain(chain).residues()]
    pdb_list = [int(idx) for idx in pdb_list]
    pdb_list = th.LongTensor(pdb_list)
    return pdb_list


def read_struct(path, chain):

    
    xyz, seq, ss = parse_xyz(path, chain, get_pdb_ss=True)
    pdb_list = parse_pdb_indices(path, chain)
    data = protein_struct(path=path,
                         chain=chain,
                         xyz=xyz,
                         seq=seq,
                         ss=ss,
                         pdb_list=pdb_list)
    return data
    