'''block diagonal ops'''

import itertools
from typing import List

import dgl
import torch


class SparseBlokDiag:
    '''
    routines to avoid explicite creating a adj/dist matrices
    '''
    blocks: List[torch.Tensor]
    def __init__(self, blocks: List[torch.Tensor]):

        assert isinstance(blocks, (list, tuple))
        self.blocks = blocks
        # unpack from fastai
        if len(self.blocks[0]) > 1:
            self.blocks = list(itertools.chain(*self.blocks))
        self.sizelist = [blk.shape[0] for blk in self.blocks]
        self.size = sum(self.sizelist)
        
    def to_sparse(self, u: torch.Tensor, v: torch.Tensor):
        '''
        create sparse COO tensor from list of block diagonal matrices
        according to u, v indices
        '''

        sparse = torch.sparse_coo_tensor(size=(self.size, self.size))
        raise NotImplementedError()


    def from_graph(self, bg: dgl.DGLGraph) -> torch.Tensor:
        '''
        Returns:
            sparse_coo_tensor
        '''
        num_edges = bg.batch_num_edges()
        num_nodes = bg.batch_num_nodes()
        uu, vv = bg.edges()
        sparse = torch.sparse_coo_tensor(size=(self.size, self.size, 2))

        values = list()

        prev = 0
        graph_nodes = 0
        for ind, e in enumerate(num_edges):
            ebegin = prev
            estop = e + prev
            ui, vi = uu[ebegin:estop], vv[ebegin:estop]
            block_match = self.blocks[ind][ui-graph_nodes, vi-graph_nodes]
            #breakpoint()
            values.append(block_match)
            graph_nodes += num_nodes[ind]
            prev += e

        values = torch.cat(values, dim=0)
        indices = torch.stack((uu, vv), dim=1)
        sparse.indices = indices
        sparse.values = values
        return sparse
    
    def sparse_from_graph(self, arr: torch.Tensor, bg: dgl.DGLGraph) -> torch.Tensor:
        '''
        convert `arr` to sparse tensor matching edges from `bg`
        Args:
            arr: torch.Tensor with shape [N, N, ?]
        '''
        num_nodes = bg.num_nodes()
        assert arr.ndim >= 2
        assert arr.shape[0] == arr.shape[1]
        assert arr.shape[1] == num_nodes
        u, v = bg.edges()
        indices = torch.stack((u, v), dim=1)
        #breakpoint()
        sparse = torch.sparse_coo_tensor(indices=indices, values= arr[u, v])
        return sparse