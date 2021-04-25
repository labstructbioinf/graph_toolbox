import os

import torch as th
import numpy as np
import atomium
import Bio.SeqUtils as bios
from scipy.spatial import distance_matrix

protein_letters_3to1 = bios.IUPACData.protein_letters_3to1_extended
protein_letters_3to1 = {k.upper() : v for k,v in protein_letters_3to1.items()}


def get_atom_xyz(atoms, atom_name):
    for a in atoms:
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

        ca_ca_matrix = distance_matrix(ca_xyz, ca_xyz)
    return ca_ca_matrix, sequence

def parse_graph_data_torch(path, chain):
    
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

        ca_xyz = th.FloatTensor(list(map(lambda v : v['CA'], preparation_dict.values())))
        sequence = list(map(lambda v : v['aa'], preparation_dict.values()))

        ca_ca_matrix = th.cdist(ca_xyz, ca_xyz)
    return ca_ca_matrix, sequence