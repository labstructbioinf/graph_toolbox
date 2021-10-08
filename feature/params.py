import itertools
aa_atoms = ['N','C','O','CA','CB','CG','CD','CE','CZ','OD','NH','NE','OG','ND','SG','OE','CH','NZ','OH','SD','OX']
aa_atoms = [a for a in aa_atoms if a not in {'OX', 'CA', 'N', 'O', 'C'}]
AT_INT = {k : i for i,k in enumerate(aa_atoms)}
INT_AT = {v:k for k, v in AT_INT.items()}

at_at_list = [[f'{at1}-{at2}' for at1 in aa_atoms] for at2 in aa_atoms]
AT_AT_name = list(itertools.chain(*at_at_list))
ATAT_INT = {k : i for i,k in enumerate(AT_AT_name)}
num_atat_feats = len(ATAT_INT)
