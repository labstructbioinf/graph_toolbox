'''block diagonal ops'''
from typing import List

import dgl
import torch


class SparseBlokDiag:

    blocks: List[torch.Tensor]

    def __init__(self, blocks: List[torch.Tensor]):

        assert isinstance(blocks, list)
        self.blocks = blocks
        self.sizelist = [blk.shape[0] for blk in blocks]
        self.size = sum(self.sizelist)
        
    def to_sparse(self, u: torch.Tensor, v: torch.Tensor):
        '''
        create sparse COO tensor from list of block diagonal matrices
        according to u, v indices
        '''

        sparse = torch.sparse_coo_tensor(size=(self.size, self.size))
        raise NotImplementedError()


    def from_graph(self, bg: dgl.DGLgraph):
        
        num_edges = bg.batch_num_edges(bg)
        u, v = bg.edges()
        sparse = torch.sparse_coo_tensor(size=(self.size, self.size))

        values = list()

        prev = 0
        for ind, e in enumerate(num_edges):
            ebegin = prev
            estop = e + prev
            ui, vi = u[ebegin:estop], v[ebegin:estop]
            block_match = self.blocks[ind][ui, vi]
            values.append(block_match)
            prev += e

        values = torch.cat(values, dim=0)
        indices = torch.stack((u, v), dim=1)
        sparse.indices = indices
        sparse.values = values
        return sparse
    
def sparse_from_graph(a, bg: dgl.DGLGraph):

    pass