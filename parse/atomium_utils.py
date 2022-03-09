from typing import List

import atomium


def atomium_select(structure: atomium.structures.Model,
                   chain: str,
                   pdb_list: list) -> List[atomium.structures.Residue]:
    '''
    select subset of residues given by `pdb_list` from atomium.model object
    return: list[residues]
    '''
    pdb_id_list = [f'{chain}.{pdb_id}' for pdb_id in pdb_list]
    residues = list()
    for pdb_id in pdb_id_list:
        residues.append(structure.residue(id=pdb_id))
    for pdbid, res in zip(pdb_id_list, residues):
        if res is None:
            print(f'pdb_list: {pdb_list}')
            print('id\'s in structure', structure.residues())
            raise KeyError(f'residue is missing for pdbid {pdbid}')
    return residues


def atomium_chain_pdb_list(structure: atomium.structures.Model) -> List[str]:
    '''
    return structure residue pdb id's
    return list[str]
    '''
    pdb_list = []
    res = structure.residue()
    # find first residue
    while res.previous is not None:
        res = res.previous
    while res is not None:
        pdb_list.append(res.id)
        res = res.next
    pdb_list = [pl[2:] for pl in pdb_list]
    return pdb_list