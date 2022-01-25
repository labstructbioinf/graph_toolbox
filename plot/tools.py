from typing import Dist
import torch as th

def tensor_to_3d_position(tensor: th.FloatTensor) -> Dict[int, list]:
    if tensor.ndim != 2:
        tensor = tensor.squeeze()
    xyz_list = tensor.tolist()
    pos_dict = {i : xyz for i, xyz in enumerate(xyz_list)}
    return pos_dict