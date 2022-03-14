import sys

import atomium
sys.path.append('..')
from feature import calc_named
from parse import parse_xyz

path = 'contactmap_mapped.pdb'
print(atomium.open(path).model.residues())
xyz, seq = parse_xyz(path, None)
print(xyz.shape)
dataframe = calc_named('contactmap_mapped.pdb', chain=None, t=9)