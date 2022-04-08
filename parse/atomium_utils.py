from typing import List, Union

import atomium


def atomium_select(structure: atomium.structures.Model,
                   chain: Union[str, None],
                   pdb_list: list) -> List[atomium.structures.Residue]:
    '''
    select subset of residues given by `pdb_list` from atomium.model object
    chain 1 letter code, None or `infer`
    return: list[residues]
    '''
    if chain is not None and len(chain) == 1:
        pdb_id_list = [f'{chain}.{pdb_id}' for pdb_id in pdb_list]
    elif chain == 'infer':
        chains = structure.chains()
        if len(chains) == 1:
            chain = chains.pop().id
            pdb_id_list = [f'{chain}.{pdb_id}' for pdb_id in pdb_list]
        else:
            raise KeyError('cannot infer chain when structure poses more then one chain')
    else:
        pdb_id_list = pdb_list
    residues = list()
    for pdb_id in pdb_id_list:
        residues.append(structure.residue(id=pdb_id))
    for pdbid, res in zip(pdb_id_list, residues):
        if res is None:
            print(f'pdb_list: {pdb_list}')
            raise KeyError(f'residue is missing for pdbid {pdbid}')
    return residues


def atomium_chain_pdb_list(structure: atomium.structures.Model, raw_id=False) -> List[str]:
    '''
    return structure residue pdb id's
    raw_id (bool) if false cuts first two indices
    return list[str]
    '''
    pdb_list = []
    # find first residue
    res = structure.residue()
    while res.previous is not None:
        res = res.previous
    while res is not None:
        pdb_list.append(res.id)
        res = res.next
    if not raw_id:
        pdb_list = [pl[2:] for pl in pdb_list]
    return pdb_list