from typing import Dict

FEATNAME = [
    'disulfide',
    'hydrophobic', 
    'cation_pi', 
    'arg_arg', 
    'salt_bridge', 
    'hbond', 
    'vdw',
    #'cx','cy','cz', 'ljx', 'ljy', 'ljz',
    'self', 
    'is_seq', 
    'is_seq_not', 
    'is_struct'
]
NFEATNAME = ['psi', 'phi', 'chi1', 'chi2']
# source: http://www.mlb.co.jp/linux/science/garlic/doc/commands/dihedrals.html
C_GAMMA = {"CG", "CG1", 'OG', 'OG1', 'SG'}
C_DELTA = {"CD", "OD1", "ND1", "SD"}
BACKBONE = {"CA", "C", "N", "O"}
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

VDW_RADIUS = {
    'H' : 1.2,
    'C' : 1.7,
    'N' : 1.55,
    'O' : 1.52,
    'S' : 1.8
}
VDW_ATOMS = {
    'GLN' : {'NE1', 'OE1'},
    'ASN' : {'ND2', 'OD1'} # ASN atoms
}
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

# RING 2.0 Table 1
HYDROGEN_ACCEPTOR = {
    'ASN':{'OD1'},
    'ASP': {'OD1', 'OD2'},
    'GLN': {'OE1'},
    'GLN': {'OE1', 'OE2'},
    'HIS': {'ND1'},
    'MET': {'SD'},
    'SER': {'OG'},
    'THR': {'OG1'},
    'TYR': {'OH'}
}

HYDROGEN_DONOR = {
    'ARG' : {'NE', 'NH1', 'NH2'},
    'ASN' : {'ND2'},
    'CYS' : {'SG'},
    'GLN' : {'NE2'},
    'HIS' : {'NE2', 'ND1'},
    'LYS' : {'NZ'},
    'SER' : {'OG'},
    'THR' : {'OG1'},
    'TRP' : {'NE1'},
    'TYR' : {'OH'}
}


SS_MAP_EXT : Dict[str, int] = {
    'H' : 0,
    'B' : 1,
    'G' : 2,
    'I' : 3,
    'T' : 4,
    'S' : 5,
    '-' : 6,
    'C' : 6,
    ' ' : 6,
    '?' : 6,
    'E' : 7,

}

amino_acid_residues_extended_upper = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'MSE',  # asdasd
    'SEC',  # Selenocysteine
    'PYL',  # Pyrrolysine
    'ORN',  # Ornithine
    'CIT',  # Citrulline
    'HYL',  # Hydroxylysine
    'ABA',  # 4-aminobutyric acid
    'AAD',  # 2-Aminoadipic acid
    'AIB',  # α-Aminoisobutyric acid
    'NLE',  # Norleucine
]

aa_3to1 = {"ALA": "A",
           "ARG": "R",
           "ASN": "N",
           "ASP": "D",
           "CYS": "C",
           "GLU": "E",
           "GLN": "Q",
           "GLY": "G",
           "HIS": "H",
           "ILE": "I",
           "LEU": "L",
           "LYS": "K",
           "MET": "M",
           "PHE": "F",
           "PRO": "P",
           "SER": "S",
           "THR": "T",
           "TRP": "W",
           "TYR": "Y",
           "VAL": "V"}

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
ACIDS_ORDER : str = amino_acid_residues_extended_upper
ACIDS_MAP_DEF3 = {acid : nb for  nb, acid in enumerate(aa_3to1)}
ACIDS_3TO1 = aa_3to1
ACIDS_MAP_DEF = {acid : nb for  nb, acid in enumerate(ACIDS_3TO1.values())}
ACIDS_MAP_R = {nb : acid for  nb, acid in enumerate(ACIDS_ORDER)}

NUM_SS_LETTERS = len(set(SS_MAP_EXT.values()))
NUM_RES_LETTERS = len(set(ACIDS_ORDER))