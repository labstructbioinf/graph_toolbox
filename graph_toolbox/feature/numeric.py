from typing import Optional
import torch as th

import math

def build_rpe_periods(n_freq: int = 16, p_min: float = 2.5, p_max: float = 50.0, device=None) -> th.Tensor:
    """Log-uniform periods in residues, returned in log space (keeps the bank positive and well
    conditioned if it is made learnable)."""
    return th.linspace(math.log(p_min), math.log(p_max), n_freq, device=device)



def compute_edge_features(res_index: th.Tensor, segment_id: th.Tensor) -> th.Tensor:
    """
    Args:
        res_index: Tensor (N,) of residue indices (e.g., [1, 2, 3... 100, 101...])
        segment_id: Tensor (N,) identifying the chain/helix (0 for central, 1 for env)
    """
    N = res_index.shape[0]
    
    # Broadcast to create N x N matrices
    i_idx = res_index.view(-1, 1).expand(N, N)
    j_idx = res_index.view(1, -1).expand(N, N)
    
    i_seg = segment_id.view(-1, 1).expand(N, N)
    j_seg = segment_id.view(1, -1).expand(N, N)

    # 1. Interaction Mask (Inter-helix vs Intra-helix)
    # This replaces generic "structural" features. 
    # It tells the GNN: "This edge is part of the packing interface."
    is_interaction = (i_seg != j_seg).float() 
    
    # 2. Relative Sequence Distance (Signed)
    # We only care about sequence distance if we are in the SAME segment.
    diff = (i_idx - j_idx).float()
    
    # Mask out cross-chain distances (set them to a dummy value, e.g., 999)
    # We use the interaction mask to zero out the diff where segments differ
    diff = diff * (1 - is_interaction) + (is_interaction * 999)
    abs_diff = th.abs(diff)

    # --- UPDATED BINARY FEATURES ---
    
    # Basic Topology
    is_self = (abs_diff == 0).float()
    is_peptide = (abs_diff == 1).float()
    
    # Helical Periodicity (The "CC Detector" Features)
    # Captures the i+3 and i+4 interactions defining the alpha-helix face
    is_turn_3 = (abs_diff == 3).float()
    is_turn_4 = (abs_diff == 4).float()
    
    # The Heptad Repeat (i to i+7)
    # Crucial for identifying the long-range order of coiled-coils
    is_heptad = (abs_diff == 7).float()

    # Long range (residues far away in sequence but close in space)
    is_long = (abs_diff > 7).float() * (1 - is_interaction)

    # --- RELATIVE POSITIONAL ENCODING (Continuous) ---
    
    # Instead of just binary buckets, give the GNN the raw value.
    # We use a sinusoidal encoding (like Transformers) to squash the distance
    # into a range [-1, 1] while preserving high-frequency info.
    # We map the distance k to: sin(k / timescale)
    
    # Only encode valid sequence distances (mask interactions)
    # We clamp diff to avoid issues with the '999' dummy value
    valid_diff = th.clamp(diff, min=-30, max=30) 
    
    rpe_sin = th.sin(valid_diff / 5.0) * (1 - is_interaction) # Scale of approx 1 turn
    rpe_cos = th.cos(valid_diff / 5.0) * (1 - is_interaction)

    # Stack all features
    # Shape: (N, N, 9)
    edge_feats = th.stack((
        is_self,            # 0: Self
        is_interaction,     # 1: Interface edge (CRITICAL for your task)
        is_peptide,         # 2: i, i+1
        is_turn_3,          # 3: i, i+3 (3-10 helix / tight turn)
        is_turn_4,          # 4: i, i+4 (Alpha helix)
        is_heptad,          # 5: i, i+7 (Coiled-coil register)
        is_long,            # 6: >7 (Distal)
        rpe_sin,            # 7: Continuous Pos Enc
        rpe_cos             # 8: Continuous Pos Enc
    ), dim=2)
    
    return edge_feats

def compute_edge_features_sparse_fixed(
    res_index: th.Tensor, 
    segment_id: th.Tensor, 
    u: th.Tensor, 
    v: th.Tensor, 
    num_rpe_freqs: int = 16
) -> th.Tensor:
    """
    Computes continuous, non-canonical-safe edge features using a sparse edge list representation.
    Designed directly for edge-augmented message passing layers like EGATConv.

    Args:
        res_index: Tensor of shape (N,) containing absolute residue indices.
        segment_id: Tensor of shape (N,) identifying the segment (e.g., 0 for helix, 1 for environment).
        u: Tensor of shape (E,) containing source node indices for every edge.
        v: Tensor of shape (E,) containing target node indices for every edge.
        num_rpe_freqs: Total dimensions allocated for Relative Positional Encoding (must be even).

    Returns:
        edge_feats: Tensor of shape (E, D) containing calculated features for each edge.
                    The total dimensionality is D = 4 + num_rpe_freqs.
                    With default num_rpe_freqs=16, the output shape is (E, 20).
    """
    # 1. Directional sequence difference (retains sign to preserve backbone polarity)
    seq_diff = (res_index[v] - res_index[u]).float()
    
    # 2. Interaction Mask (1.0 if edge crosses between helix and environment, 0.0 otherwise)
    is_cross_segment = (segment_id[u] != segment_id[v]).float().unsqueeze(-1)
    
    # 3. Multi-frequency Relative Positional Encoding (Continuous RPE)
    # Instead of hardcoded buckets, we use a geometric spectrum of frequencies.
    # Base is scaled to 100.0 to focus heavily on local/coiled-coil scale interactions.
    inv_freq = th.exp(
        th.arange(0, num_rpe_freqs, 2, dtype=th.float32, device=res_index.device) 
        * -(math.log(100.0) / num_rpe_freqs)
    )
    
    angles = seq_diff.unsqueeze(-1) * inv_freq
    rpe_sin = th.sin(angles)
    rpe_cos = th.cos(angles)
    
    # Concatenate sin/cos components and nullify RPE for cross-segment interactions
    rpe_feats = th.cat([rpe_sin, rpe_cos], dim=-1) * (1 - is_cross_segment)
    
    # 4. Topological and Directional Polarity Flags
    is_self = (seq_diff == 0).float().unsqueeze(-1)
    dir_forward = (seq_diff > 0).float().unsqueeze(-1) * (1 - is_cross_segment)
    dir_backward = (seq_diff < 0).float().unsqueeze(-1) * (1 - is_cross_segment)
    
    # Concatenate all components into the final edge feature tensor
    edge_feats = th.cat([
        is_cross_segment,  # Index 0: Interface boundary flag
        is_self,           # Index 1: Self-loop flag
        dir_forward,       # Index 2: Sequence direction (N -> C)
        dir_backward,      # Index 3: Sequence direction (C -> N)
        rpe_feats          # Indices 4 to (4 + num_rpe_freqs - 1): Continuous position spectrum
    ], dim=-1)
    
    return edge_feats



def compute_edge_features_sparse(
    res_index: th.Tensor, 
    segment_id: th.Tensor, 
    u: th.Tensor, 
    v: th.Tensor, 
    num_rpe_freqs: int = 16
) -> th.Tensor:
    """
    Computes continuous, non-canonical-safe edge features using a sparse edge list representation.
    Designed directly for edge-augmented message passing layers like EGATConv.

    Args:
        res_index: Tensor of shape (N,) containing absolute residue indices.
        segment_id: Tensor of shape (N,) identifying the segment (e.g., 0 for helix, 1 for environment).
        u: Tensor of shape (E,) containing source node indices for every edge.
        v: Tensor of shape (E,) containing target node indices for every edge.
        num_rpe_freqs: Total dimensions allocated for Relative Positional Encoding (must be even).

    Returns:
        edge_feats: Tensor of shape (E, D) containing calculated features for each edge.
                    The total dimensionality is D = 4 + num_rpe_freqs.
                    With default num_rpe_freqs=16, the output shape is (E, 20).
    """
    # 1. Directional sequence difference (retains sign to preserve backbone polarity)
    seq_diff = (res_index[v] - res_index[u]).float()
    
    # 2. Interaction Mask (1.0 if edge crosses between helix and environment, 0.0 otherwise)
    is_cross_segment = (segment_id[u] != segment_id[v]).float().unsqueeze(-1)
    
    # 3. Multi-frequency Relative Positional Encoding (Continuous RPE)
    # Instead of hardcoded buckets, we use a geometric spectrum of frequencies.
    # Base is scaled to 100.0 to focus heavily on local/coiled-coil scale interactions.
    inv_freq = th.exp(
        th.arange(0, num_rpe_freqs, 2, dtype=th.float32, device=res_index.device) 
        * -(math.log(100.0) / num_rpe_freqs)
    )
    
    angles = seq_diff.unsqueeze(-1) * inv_freq
    rpe_sin = th.sin(angles)
    rpe_cos = th.cos(angles)
    
    # Concatenate sin/cos components and nullify RPE for cross-segment interactions
    rpe_feats = th.cat([rpe_sin, rpe_cos], dim=-1) * (1 - is_cross_segment)
    
    # 4. Topological and Directional Polarity Flags
    is_self = (seq_diff == 0).float().unsqueeze(-1)
    dir_forward = (seq_diff > 0).float().unsqueeze(-1) * (1 - is_cross_segment)
    dir_backward = (seq_diff < 0).float().unsqueeze(-1) * (1 - is_cross_segment)
    
    # Concatenate all components into the final edge feature tensor
    edge_feats = th.cat([
        is_cross_segment,  # Index 0: Interface boundary flag
        is_self,           # Index 1: Self-loop flag
        dir_forward,       # Index 2: Sequence direction (N -> C)
        dir_backward,      # Index 3: Sequence direction (C -> N)
        rpe_feats          # Indices 4 to (4 + num_rpe_freqs - 1): Continuous position spectrum
    ], dim=-1)
    
    return edge_feats


@th.jit.script
def distance(xyz1: th.Tensor, xyz2: Optional[th.Tensor] = None):

    if xyz2 is not None:
        return th.cdist(xyz1, xyz2)
    else:
        return th.cdist(xyz1, xyz1)


@th.jit.script
def dotproduct(xyz1, xyz2):
    return (xyz1 * xyz2).sum(-1, keepdim=True)


@th.jit.script
def calc_single_dihedral(x1, x2, x3, x4) -> th.Tensor:
    """
    Returns:
        angle: [num_atoms, 1]
    """

    b1 = x2 - x1
    b2 = x3 - x2
    b3 = x4 - x3
    cross12 = th.linalg.cross(b1, b2)
    cross23 = th.linalg.cross(b2, b3)

    # keeps values in a comfortable range and avoids overflow on very long bonds
    cross12 = th.nn.functional.normalize(cross12, dim=-1)
    cross23 = th.nn.functional.normalize(cross23, dim=-1)

    # division by 0 risk
    #b2 /= th.pow(b2, 2).sum(-1, keepdim=True).sqrt()
    # fix:
    norm = b2.norm(2, [-1], True).clamp_min(1e-8)
    b2 = b2 / norm

    cross_1223 = th.linalg.cross(cross12, cross23)
    dot_1223 = dotproduct(cross12, cross23)
    
    angle = th.atan2(dotproduct(b2, cross_1223), dot_1223)
    
    if angle.ndim == 1:
        angle = angle.unsqueeze(-1)
    
    return angle


@th.jit.script
def backbone_dihedral(n: th.Tensor, ca: th.Tensor, c: th.Tensor):
    """
    Args:
        nitrogen: torch.Tensor [num_atoms, xyz]
        carbon_alpha: torch.Tensor
        carbon: torch.Tensor
        
        C - N - CA - C - N
         b1  b2   b3  vectors
           n1   n2  planes
           
        phi: C'-N-CA-C
        psi: N-CA-C-N'
      omaga: CA-C-N'-CA'  (this is VADAR post-bond nomenclature, the pre-bond CA'-C'-N-CA is also used)
        
    Returns:
        torch.FloatTensor: phi
        torch.FloatTensor: psi
        torch.FloatTensor: omega
    """

    phi = calc_single_dihedral(c.roll(1, dims=0), n, ca, c)
    psi = calc_single_dihedral(n, ca, c, n.roll(-1, dims=0))

    #omega = calc_single_dihedral(ca.roll(1, dims=0), n.roll(1, dims=0), c, ca)
    #FIX:
    omega = calc_single_dihedral(
                ca,   # CA(i)
                c,    # C (i)
                n.roll(-1, dims=0), # N (i+1)
                ca.roll(-1, dims=0) # CA(i+1)
    )              
    
    phi[0] = float("nan")
    psi[-1] = float("nan")
    omega[-1] = float("nan")

    return th.cat((phi, psi, omega), dim=1)


def sidechain_dihedral(n, ca, cb, cg, cd):
    """
    same story as in backbone, but without roll
    """
    chi1 = calc_single_dihedral(n, ca, cb, cg)
    chi2 = calc_single_dihedral(ca, cb, cg, cd)
    return th.cat((chi1, chi2), dim=1)

#### TEST #####

# example N, CA, C, CB, CG, CD data for https://www.rcsb.org/structure/1AJQ
# residues 4..19

import torch as th

n = th.tensor([
    [ 0.310, 12.623,  6.552],
    [ 2.500, 11.546,  7.805],
    [ 4.705, 13.052,  7.253],
    [ 7.321, 15.356,  8.111],
    [ 8.312, 18.570,  7.644],
    [11.211, 20.475,  7.751],
    [12.292, 23.722,  7.679],
    [15.063, 25.898,  7.525],
    [15.959, 29.307,  8.257],
    [17.807, 32.291,  7.022],
    [19.237, 32.802,  9.434],
    [19.543, 30.453, 10.496],
    [17.401, 29.696, 12.011],
    [14.807, 27.429, 12.807],
    [11.810, 26.287, 11.258],
    [ 8.865, 24.479, 11.578],
], dtype=th.float32)

ca = th.tensor([
    [ 0.320, 12.666,  8.004],
    [ 3.759, 11.078,  8.314],
    [ 5.776, 13.952,  6.836],
    [ 7.628, 16.602,  8.873],
    [ 9.214, 19.385,  6.852],
    [11.966, 21.459,  8.569],
    [12.944, 24.763,  6.889],
    [15.851, 26.872,  8.306],
    [16.128, 30.586,  7.579],
    [19.195, 32.768,  6.924],
    [20.008, 32.823, 10.691],
    [19.614, 29.089, 11.032],
    [16.213, 29.432, 12.817],
    [14.036, 26.339, 12.201],
    [10.384, 26.340, 11.164],
    [ 8.389, 23.086, 11.602],
], dtype=th.float32)

c = th.tensor([
    [ 1.658, 12.240,  8.588],
    [ 4.883, 12.114,  8.146],
    [ 6.048, 15.111,  7.802],
    [ 8.636, 17.387,  8.009],
    [ 9.890, 20.421,  7.806],
    [12.695, 22.473,  7.682],
    [13.773, 25.720,  7.790],
    [16.122, 28.176,  7.552],
    [17.577, 31.047,  7.506],
    [19.908, 32.884,  8.259],
    [20.014, 31.449, 11.293],
    [18.357, 28.799, 11.857],
    [15.233, 28.460, 12.101],
    [12.572, 26.618, 12.291],
    [ 9.868, 24.896, 10.842],
    [ 6.988, 23.062, 11.024],
], dtype=th.float32)

cb = th.tensor([
    [ -0.823,  11.771,   8.504 ],   #  4 SER  – CB
    [  4.138,   9.769,   7.575 ],   #  5 SER  – CB
    [  5.279,  14.535,   5.511 ],   #  6 GLU  – CB
    [  8.265,  16.339,  10.240 ],   #  7 ILE  – CB
    [  8.410,  20.106,   5.773 ],   #  8 LYS  – CB
    [ 13.052,  20.747,   9.386 ],   #  9 ILE  – CB
    [ 11.855,  25.515,   6.140 ],   # 10 VAL  – CB
    [ 17.199,  26.276,   8.705 ],   # 11 ARG  – CB
    [ 15.192,  31.673,   8.114 ],   # 12 ASP  – CB
    [ 19.301,  34.103,   6.175 ],   # 13 GLU  – CB
    [ 19.396,  33.884,  11.655 ],   # 14 TYR  – CB
    [   th.nan,   th.nan,   th.nan ],# 15 GLY  – brak CB
    [ 15.493,  30.754,  13.037 ],   # 16 MET  – CB
    [ 14.324,  25.071,  13.022 ],   # 17 PRO  – CB
    [  9.992,  27.266,  10.026 ],   # 18 HIS  – CB
    [  8.413,  22.359,  12.944 ]    # 19 ILE  – CB
], dtype=th.float32)


cg = th.tensor([
    [  -2.019,   12.223,   7.865 ],   #  4 SER – OG
    [   4.453,   10.070,   6.199 ],   #  5 SER – OG
    [   5.753,   15.913,   5.127 ],   #  6 GLU – CG
    [   7.382,   15.472,  11.130 ],   #  7 ILE – CG1
    [   9.308,   20.939,   4.864 ],   #  8 LYS – CG
    [  12.494,   19.648,  10.268 ],   #  9 ILE – CG1
    [  12.408,   26.744,   5.406 ],   # 10 VAL – CG1
    [  17.061,   25.367,   9.878 ],   # 11 ARG – CG
    [  15.691,   32.158,   9.520 ],   # 12 ASP – CG
    [  18.044,   34.663,   5.557 ],   # 13 GLU – CG
    [  19.937,   35.273,  11.299 ],   # 14 TYR – CG
    [   th.nan,   th.nan,   th.nan ],  # 15 GLY – brak CG
    [  14.238,   30.661,  13.896 ],   # 16 MET – CG
    [  14.753,   25.670,  14.366 ],   # 17 PRO – CG
    [   9.978,   28.708,  10.409 ],   # 18 HIS – CG
    [   9.656,   22.466,  13.819 ]    # 19 ILE – CG1
], dtype=th.float32)

cd = th.tensor([
    [   th.nan,   th.nan,   th.nan ],  #  4 SER – brak CD
    [   th.nan,   th.nan,   th.nan ],  #  5 SER – brak CD
    [   6.295,   15.940,   3.716 ],   #  6 GLU – CD
    [   8.253,   14.445,  11.842 ],   #  7 ILE – CD1
    [   8.583,   22.060,   4.210 ],   #  8 LYS – CD
    [  13.578,   18.642,  10.648 ],   #  9 ILE – CD1
    [   th.nan,   th.nan,   th.nan ],  # 10 VAL – brak CD
    [  18.234,   24.447,   9.992 ],   # 11 ARG – CD
    [  16.709,   31.731,  10.055 ],   # 12 ASP – OD1  (kanoniczny δ)
    [  17.107,   35.322,   6.569 ],   # 13 GLU – CD
    [  21.182,   35.709,  11.739 ],   # 14 TYR – CD1
    [   th.nan,   th.nan,   th.nan ],  # 15 GLY – brak CD
    [  13.413,   32.211,  14.219 ],   # 16 MET – SD   (kanoniczny δ)
    [  15.420,   27.017,  14.097 ],   # 17 PRO – CD
    [  11.112,   29.498,  10.333 ],   # 18 HIS – ND1  (kanoniczny δ)
    [  10.984,   21.889,  13.315 ]    # 19 ILE – CD1
], dtype=th.float32)


# reference phi, psi, omega and chi1 values caluclated with https://pdb.indhinditech.com/ and http://vadar.wishartlab.com/

phi_ref = th.tensor([
    -82.11585,
    -88.43207,
    -79.78755,
    -122.76428,
    -102.04108,
    -115.21435,
    -106.61284,
    -103.52467,
    -84.48794,
    -63.10700,
    -107.03455,
     87.09579,
    -73.80208,
    -91.19301,
    -121.34759,
    -114.39235
], dtype=th.float32)

psi_ref = th.tensor([
   -18.17370,
   -20.22146,
   133.62032,
   121.65834,
   128.41147,
   111.41131,
   124.41387,
   136.14895,
  -173.04028,
   -11.97806,
    10.81421,
     0.07436,
   129.93947,
   141.27759,
   132.58415,
   123.57917
], dtype=th.float32)

omega_ref = th.tensor([
   -177.9,
   -169.3,
    170.6,
    178.8,
    177.5,
    178.6,
    175.8,
   -176.6,
   -179.4,
    175.5,
    176.4,
   -178.8,
   -171.2,
    173.6,
   -168.9,
   -178.5
], dtype=th.float32)

chi1_ref = th.tensor([
   -54.3,   
    69.0,   
   151.5,   
   -58.3,   
  -177.6,   
   -56.3,   
   173.1,   
   -79.2,   
    73.9,   
    -5.2,   
   -82.8,   
  th.nan,   # glycine!
  -178.6,   
    23.6,   
   -83.2,   
   -45.6    
], dtype=th.float32)

if __name__ == "__main__":

    # test backbone dihedral angles calculation
    bb_dih = backbone_dihedral(n, ca, c) # phi, psi, omega 
    
    assert th.allclose(th.rad2deg(bb_dih[1:, 0]), phi_ref[1:], rtol=1e-3, atol=1e-2)
    assert th.allclose(th.rad2deg(bb_dih[:-1, 1]), psi_ref[:-1], rtol=1e-3, atol=1e-2)
    assert th.allclose(th.rad2deg(bb_dih[:-1, 2]), omega_ref[:-1], rtol=1e-3, atol=1e-2)

    # test side-chain dihedral angles calculation
    side_dih = sidechain_dihedral(n, ca, cb, cg, cd) # chi1, chi2
    assert th.allclose(th.rad2deg(side_dih[:, 0]), chi1_ref, rtol=1e-3, atol=1e-2, equal_nan=True)
    # no data for chi2 test!
    

