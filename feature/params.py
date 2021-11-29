HYDROPHOBIC = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'RPO', 'TYR'}
AROMATIC = {'TRP', 'TYR', 'PHE'}
CATION_PI = {'CG', 'CD', 'CE', 'CZ'}
SALT_BRIDGE_C1 = {'NH', 'NZ'}
SALT_BRIDGE_C2 = {'OE', 'OD'}
CHARGE = {'H' : 1,
          'C' : 4,
          'N' : -3,
          'O' : -2,
          'S' : -2}

VDW_RADIUS = {'H' : 1.2,
          'C' : 1.7,
          'N' : 1.55,
          'O' : 1.52,
          'S' : 1.8}

# averaged over multiple set of params
SIGMA = {
    'H' : 1.44,
    'C' : 3.16,
    'O' : 2.75,
    'N' : 2.4,
    'S' : 3.5
}
EPSILON = {
    'H' : 0,
    'C' : 1,
    'O' : 1,
    'N' : 1,
    'S' : 1
}
HYDROGEN_ACCEPTOR = {'NH', 'OH'}
HYDROGEN_DONOR = {'F', 'N', 'O'}