from dataclasses import dataclass, asdict, field
from typing import List

import torch as th

from .params import INTERACTION_FEATS, CONTACT_TYPE_FEATS


@dataclass
class StructFeats:
    """
    container for structural features
    """

    u: th.Tensor
    v: th.Tensor
    efeats: th.Tensor
    nfeats: th.Tensor
    sequence: List[str]
    distancemx: th.Tensor
    residueid: th.Tensor
    chainids: List[str]
    with_interactions: bool = field(default=True)
    edge_feature_names: list[str] = field(init=False)

    def __post_init__(self):
        if self.with_interactions:
            self.edge_feature_names = StructFeats.edge_features(self.with_interactions)
        else:
            self.edge_feature_names = StructFeats.edge_features(self.with_interactions)

    @staticmethod
    def edge_features(with_interactions=False) -> list[str]:
        """returns edges features names"""
        if with_interactions:
            return INTERACTION_FEATS + CONTACT_TYPE_FEATS
        else:
            return CONTACT_TYPE_FEATS

    def asdict(self):
        return asdict(self)
