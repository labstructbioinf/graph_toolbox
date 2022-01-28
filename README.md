# graph_toolbox
utilities for graph-like data processing and neural network related

## IO
```python
save_json(data: Union[dict, list], file: str)
load_gpickle(file: str)
save_gpickle(obj, file: str)
unpack_gzip_to_pdb_atomium_obj(gzip_file_path: str)
```
## read
```python
parse_xyz(path: str, chain: str, get_pdb_ss:bool=False)-> th.FloatTensor, List[str]
```
## plot
```python
tensor_to_3d_position(tensor: th.FloatTensor) -> Dict[int, list]
```

## feature
```python
read_struct(pdb_loc: Union[str,atomium.structures.Model], chain: str, t: int)
```
