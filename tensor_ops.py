import torch as th

def delete_indices(tensor, indices, return_mask=False):
    
    #if isinstance(th.)
    tensor[indices] = -1
    return th.nonzero(tensor != -1).squeeze()

def mask_sequential_connections(u,v):
    return th.abs(u - v) < 2


def argwhere(condition):
    return np.nonzero(condition, as_tuple=True)



def diagonalize(R):
    '''
    transform set of points eg. resisues x,y,z coords
    to new coordinates where inertia tensor is diagonal
    '''
    R = R-R.mean(0)
    I = th.zeros((3,3), dtype=th.float)
    for i in range(R.shape[0]):
        ri = R[i, :]
        r = ri.pow(2).sum()
        r_mx = th.diagflat(r.repeat(3))
        rij = ri.unsqueeze(1) @ ri.unsqueeze(0)
        I += r_mx - rij
    _, P = th.eig(I, True)
    return R@P



def rmsd_chunk(r1, r2):
    '''
    calculate rmsd for two equal length Ca sequences
    '''
    r1, r2 = r1 - r1.mean(0), r2 - r2.mean(0)
    
    dist = (r1 - r2).pow(2).sum().sqrt()
    naive_norm = th.tensor(dist.size()).prod().float().sqrt()
    rmsd = dist.pow(2).sum().sqrt()/naive_norm
    return rmsd



def diff_slice(indices):
    '''
    split indices into continious groups
    '''
    indices_diff = (indices[1:] - indices[:-1]) > 2
    start = th.nonzero(xi).flatten()
    stop = start.clone()
    start = th.cat((th.tensor([0]), start)) 
    stop = th.cat((stop, th.tensor([indices.shape[0]])))
    seq1_slices = []
    for i,j in zip(start, stop):
        seq1_slices.append(th.unique_consecutive(x.narrow(0, i, j-i)))
    return seq1_slices
    
        
    
    