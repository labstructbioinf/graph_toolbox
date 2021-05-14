import dgl
import torch as th

def cm_to_eids(cm, threshold):
    
    adj  = th.nonzero(cm < threshold)
    u = adj[:, 0]
    v = adj[:, 1]
    return dgl.graph((u,v))