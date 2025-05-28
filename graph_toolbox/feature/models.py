
from dataclasses import dataclass, asdict
from typing import List

import torch as th

@dataclass
class StructFeats:
    u: th.Tensor
    v: th.Tensor
    efeats: th.Tensor
    nfeats: th.Tensor
    sequence: List[str]
    distancemx: th.Tensor
    residueid: th.Tensor
    chainids: List[str]

    def asdict(self):
        return asdict(self)