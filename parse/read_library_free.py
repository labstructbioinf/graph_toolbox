import os
import re 
from collections import namedtuple

import numpy as np

split = re.compile('\s+')
struct_desc = namedtuple('struc_desc', ['resid', 'resname', 'xyz', 'resatom'])

def read_pdb_basic(path: str):

    assert os.path.isfile(path)
    with open(path, 'rt') as file:
        atom_data = list()
        for line in file.readlines():
            if line.startswith('ATOM'):
                atom_data.append(split.split(line))
        residues = []
        atoms = []
        xyz = []
        for line in atom_data:
            residues.append(line[3])
            atoms.append(line[11])
            xyz.append(line[6:9])
    arr = np.asarray(xyz, dtype=np.float32)
    return arr


def read_pdb(path: str):

    assert os.path.isfile(path)
    with open(path, 'rt') as file:
        atom_data = list()
        for line in file.readlines():
            if line.startswith('ATOM'):
                atom_data.append(split.split(line))
        resid = []
        resname = []
        atomresname = []
        xyz = []
        for line in atom_data:
            resid.append(line[3])
            atomresname.append(line[2])
            resname.append(line[3])
            xyz.append(line[6:9])
    arr = np.asarray(xyz, dtype=np.float32)
    data = struct_desc(resid=resid, resname=resname, xyz=xyz, resatom=atomresname)
    return data