import torch as th

def delete_indices(tensor, indices, return_mask=False):
    
    #if isinstance(th.)
    tensor[indices] = -1
    return th.nonzero(tensor != -1).squeeze()

def mask_sequential_connections(u,v):
    return th.abs(u - v) < 2
    
        
    
    