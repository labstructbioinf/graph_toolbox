# graph_toolbox
utilities for graph-like data processing and neural network related

# submodules
## IO
```python
save_json(data: Union[dict, list], file: str)
load_gpickle(file: str)
save_gpickle(obj, file: str)
unpack_gzip_to_pdb_atomium_obj(gzip_file_path: str)
```
## parse
read XYZ coordinates, sequence and optionally secondary structure
```python
parse_xyz(path: str, chain: str, get_pdb_ss:bool=False)-> th.FloatTensor, List[str]
```
select residues from structure
```python
atomium_select(structure: atomium.structures.Model, chain: str, pdb_list: list) -> List[atomium.structures.Residue]
```

## plot
```python
tensor_to_3d_position(tensor: th.FloatTensor) -> Dict[int, list]
```

## pymol
```python
PyMolColor
```

## feature
```python
read_struct(pdb_loc: Union[str,atomium.structures.Model], chain: Union[str, None], t: int)
```
