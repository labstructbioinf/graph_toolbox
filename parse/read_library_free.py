import os
import re
import itertools 
from typing import Tuple
from collections import namedtuple

import numpy as np

split = re.compile('\s+')
struct_desc = namedtuple('struc_desc', ['resid', 'resname', 'xyz', 'resatom'])
cb_nan = np.array((np.nan, np.nan, np.nan), dtype=np.float32)

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
            resid.append(line[5])
            atomresname.append(line[2])
            resname.append(line[3])
            xyz.append(line[6:9])
    arr = np.asarray(xyz, dtype=np.float32)
    data = struct_desc(resid=resid, resname=resname, xyz=arr, resatom=atomresname)
    return data

def find_cba_atoms(sd: struct_desc) -> Tuple[np.array, np.array]:
    cb_xyz_list = list()
    ca_xyz_list = list()
    res_num_atom = list()
    c_nan = np.array((np.nan, np.nan, np.nan), dtype=np.float32)
    pos_start = 0
    for resid, resiter in itertools.groupby(sd.resid):
        num_resatom = len(list(resiter))
        pos_end = pos_start + num_resatom
        resatoms = itertools.islice(sd.resatom, pos_start, pos_end)
        res_ca, res_cb = cb_nan, cb_nan
        for i, at in enumerate(resatoms, start=pos_start):
            if at == 'CB':
                res_cb = sd.xyz[i]
            elif at == 'CA':
                res_ca = sd.xyz[i]

        cb_xyz_list.append(res_cb)
        ca_xyz_list.append(res_ca)
        res_num_atom.append(num_resatom)
        pos_start = pos_end
    cb = np.asarray(cb_xyz_list)
    ca = np.asarray(ca_xyz_list)
    return ca, cb, res_num_atom
        
def read_pdb_full(path: str):

    ds = read_pdb(path)
    ca, cb, res_atnum = find_cba_atoms(ds)
    return ds.resname, ds.resatom, res_atnum, ds.xyz, ca, cb
