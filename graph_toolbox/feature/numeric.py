
from typing import Optional
import torch as th


@th.jit.script
def distance(xyz1: th.Tensor, xyz2: Optional[th.Tensor] = None):

    if xyz2 is not None:
        return th.cdist(xyz1, xyz2)
    else:
        return (xyz1.unsqueeze(0) - xyz1.unsqueeze(1)).pow(2).sum(-1).sqrt()


@th.jit.script
def dotproduct(xyz1, xyz2):
    return (xyz1 * xyz2).sum(-1, keepdim=True)


@th.jit.script
def calc_single_dihedral(x1, x2, x3, x4) -> th.Tensor:
    """
    Returns:
        angle: [num_atoms, 1]
    """
    # https://en.wikipedia.org/wiki/Dihedral_angle
    # https://en.wikipedia.org/w/index.php?title=Dihedral_angle&oldid=689165217#Angle_between_three_vectors
    b1 = x2 - x1
    b2 = x3 - x2
    b3 = x4 - x3
    cross12 = th.linalg.cross(b1, b2)
    #cross12 /= LA.vector_norm(cross12, ord=2, dim=1, keepdim=True)
    cross23 = th.linalg.cross(b2, b3)
    #cross23 /= LA.vector_norm(cross23, ord=2, dim=1, keepdim=True)
    b2 /= th.pow(b2, 2).sum(-1, keepdim=True).sqrt()
    cross_1223 = th.linalg.cross(cross12, cross23)
    dot_1223 = dotproduct(cross12, cross23)
    #dot_b2cross1223 = dotproduct(b2, cross_1223)
    #print(b2norm.shape, dot_1223.shape, cross_1223.shape)
    #angle = th.atan2( dot_b2cross1223,  b2norm*dot_1223)
    angle = th.atan2( dotproduct(b2,cross_1223), dot_1223)
    if angle.ndim == 1:
        angle = angle.unsqueeze(-1)
    return angle


#@th.jit.script
def backbone_dihedral(n: th.Tensor, ca: th.Tensor, c: th.Tensor):
    """
    Args:
        nitrogen: torch.Tensor [num_atoms, xyz]
        carbon_alpha: torch.Tensor
        carbon: torch.Tensor
        C - N - CA - C - N
         b1  b2   b3  vectors 
           n1   n2  planes
        phi: C-N-CA-C
        psi: N-CA-C-N
    Returns:
        torch.FloatTensor: phi
        torch.FloatTensor: psi
    """

    phi = calc_single_dihedral(c.roll(1, dims=0), n, ca, c)
    psi = calc_single_dihedral(n, ca, c, n.roll(-1, dims=0))
    omega = calc_single_dihedral(ca.roll(1, dims=0), n.roll(1, dims=0), c, ca)
    phi[0] = float('nan')
    psi[-1] = float('nan')
    omega[0] = float('nan')

    return th.cat((phi, psi, omega), dim=1)


def sidechain_dihedral(n, ca, cb, cg, cd):
    """
    same story as in backbone, but without roll
    """
    chi1 = calc_single_dihedral(n, ca, cb, cg)
    chi2 = calc_single_dihedral(ca, cb, cg, cd)
    return th.cat((chi1, chi2), dim=1)
