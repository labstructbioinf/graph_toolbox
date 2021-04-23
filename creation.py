import numpy as np
import dgl

def cm_to_eids(cm, threshold):
    
    adj  = np.argwhere(cm < threshold)
    u = adj[:, 0]
    v = adj[:, 1]
    return dgl.graph((u,v))