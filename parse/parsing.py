import os
import warnings
from functools import partial
from collections import namedtuple
from typing import List, Union
try:
    import torch as th
except ImportError:
    warnings.warn('missing pytorch some functionalities with be broken')
import numpy as np
import atomium
try:
    import Bio.SeqUtils as bios
    protein_letters_3to1 = bios.IUPACData.protein_letters_3to1_extended
    protein_letters_3to1 = {k.upper() : v for k,v in protein_letters_3to1.items()}
except ImportError:
    pass
from scipy.spatial import distance_matrix


protein_letters = 'ACDEFGHIKLMNPQRSTVWY' + 'X'
residue_to_num_dict = {res : num for num, res in enumerate(protein_letters)}
map_residue = lambda res : residue_to_num_dict[res.code]
protein_struct = namedtuple('protein_struct', ['path', 'pdb_chain', 'chain', 'xyz', 'seq', 'ss', 'pdb_list'])


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

def _is_handle_valid(handle: Union[str, atomium.structures.Model]):
    if isinstance(handle, str):
        if not os.path.isfile(handle):
            raise FileNotFoundError(f'file {handle} missing')
        else:
            handle = atomium.open(handle).model
    return handle
    
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


def parse_xyz(handle: Union[str, atomium.structures.Model],
              chain: Union[str, None] = None,
              get_pdb_ss:bool = False):
    '''
    read structure and return its xyz coordinates
    if handle posses more then one model then lists are returned
    Params:
        path (str or atomium model)
        chain (str, None) default None if str read chain if None read all
    return:
        ca_xyz (torch.Tensor or List[torch.Tensor])
        sequence (list or List[str])
        
    '''
    # validate input arguments
    if isinstance(handle, str):
        if not os.path.isfile(handle):
            raise KeyError(f'path: {handle} doesn\'t exist')
        # read structure from file
        else:
            data = atomium.open(handle).models
    elif isinstance(handle, atomium.structures.Model):
        data = [handle]
    else:
        raise KeyError(f'invalid path argument type for path: {handle}')

    xyz_list = list()
    seq_list = list()
    sec_list = list()
    for model in data:
        if chain is not None:
            model = model.chain(chain)
        # read content
        sequence = list(map(lambda x: x.code, model.residues()))
        ca_xyz = list(map(get_CA_xyz, model.residues()))
        ca_xyz = th.FloatTensor(ca_xyz)
        seq_list.append(sequence)
        xyz_list.append(ca_xyz)
        if get_pdb_ss:
            secondary = list(map(get_ss_label, model.residues()))
    if get_pdb_ss:
        if len(data) == 1:
            return xyz_list.pop(), seq_list.pop(), sec_list.pop()
        else:
            return xyz_list, seq_list, sec_list
    else:
        if len(data) == 1:
            return xyz_list.pop(), seq_list.pop()
        else:
            return xyz_list, seq_list


def parse_pdb_indices(path, chain):
    
    if isinstance(path, str):
        if not os.path.isfile(path):
            raise KeyError(f'path: {path} doesn\'t exist')
        else:
            data = atomium.open(path).model
            chain = data.chain(chain)
    elif isinstance(path, atomium.structures.Model):
        chain = path.chain(chain)
    else:
        raise KeyError(f'invalid path arg type {path}')
    pdb_list = [
        s.id.split('.')[1] for s in chain.residues()
    ]
    return pdb_list


def read_struct(path, pdb_chain):
    '''
    path (str, context) location on structure file
    '''
    pdb, chain = pdb_chain.split('_')
    xyz, seq, ss = parse_xyz(path, chain, get_pdb_ss=True)
    pdb_list = parse_pdb_indices(path, chain)
    if not isinstance(path, str):
        path = 'none'
    data = protein_struct(path=path,
                          pdb_chain=pdb_chain,
                         chain=chain,
                         xyz=xyz,
                         seq=seq,
                         ss=ss,
                         pdb_list=pdb_list)
    return data
    