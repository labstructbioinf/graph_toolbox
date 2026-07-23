import torch as th
import math

def compute_edge_features_rich(
    res_index: th.Tensor, 
    segment_id: th.Tensor, 
    chain_id: th.Tensor,
    u: th.Tensor, 
    v: th.Tensor, 
    num_rpe_freqs: int = 16,
    clamp_dist: float = 64.0,
    min_period: float = 2.5,
    max_period: float = 32.0
) -> th.Tensor:
    """
    Computes continuous edge features using a sparse edge list representation.
    Designed directly for edge-augmented message passing layers like EGATConv.

    MAJOR FIXES AND ADDITIONS APPLIED:
    1. Cross-Chain Handling (NEW): Added `chain_id` to resolve the "Cross-Chain Sequence 
       Distance Paradox". Sequence distances between different chains are now explicitly 
       zeroed out, and RPE/directional flags are nullified for inter-chain edges, 
       forcing the model to rely purely on 3D spatial features for multimeric interactions.
    2. Topological Self-Loops: Fixed `is_self` to use true node identity (`u == v`) 
       instead of sequence difference (`seq_diff == 0`). Prevents cross-chain 
       multimer collisions from being flagged as self-loops.
    3. Sequence Distance Clamping: Added `clamp_dist` to bound `seq_diff`. Prevents 
       unpredictable RPE phase-wrapping when evaluating edges between structurally 
       adjacent but sequentially distant residues in the discontinuous environment.
    4. Generic Band-Pass RPE: Replaced the NLP-style exponential frequency decay 
       with a dense, linearly spaced frequency spectrum. Blankets the specific 
       structural window (2.5 to 32 residues) to capture arbitrary non-canonical 
       periodicities via superposition without hardcoding biological motifs.
    5. Directional Flags: Modified forward/backward sequence direction flags to 
       explicitly exclude true topological self-loops (`u != v`) and cross-chain edges.

    Args:
        res_index: Tensor of shape (N,) containing absolute residue indices.
        segment_id: Tensor of shape (N,) identifying the segment (0: helix, 1: env).
        chain_id: Tensor of shape (N,) identifying the protein chain for each residue.
        u: Tensor of shape (E,) containing source node indices.
        v: Tensor of shape (E,) containing target node indices.
        num_rpe_freqs: Total dimensions for Relative Positional Encoding (must be even).
        clamp_dist: Maximum sequence distance to consider before capping.
        min_period: Minimum structural period to capture (Nyquist limit for turns).
        max_period: Maximum structural period to capture (Macro-repeats).

    Returns:
        edge_feats: Tensor of shape (E, 5 + num_rpe_freqs).
    """
    # Prevent silent dimension mismatches during hyperparameter sweeps
    assert num_rpe_freqs % 2 == 0, "num_rpe_freqs must be even for sin/cos pairing."

    # 1. Topological Flags
    is_self = (u == v).float().unsqueeze(-1)
    
    # 2. Boundary Masks
    is_cross_segment = (segment_id[u] != segment_id[v]).float().unsqueeze(-1)
    is_cross_chain = (chain_id[u] != chain_id[v]).float().unsqueeze(-1)
    
    # 3. Directional sequence difference
    seq_diff = (res_index[v] - res_index[u]).float()
    
    # Zero out sequence distance for cross-chain edges (biologically undefined)
    seq_diff = seq_diff * (1 - is_cross_chain).squeeze(-1) 
    seq_diff = th.clamp(seq_diff, min=-clamp_dist, max=clamp_dist)
    
    # 4. Generic Band-Pass Relative Positional Encoding
    # Convert periods to frequencies (f = 1/T)
    max_freq = 1.0 / min_period
    min_freq = 1.0 / max_period
    
    # Create a dense spectrum of frequencies blanketing the structural window
    freqs = th.linspace(max_freq, min_freq, steps=(num_rpe_freqs // 2), device=res_index.device)
    
    # Convert to angular frequencies (omega = 2 * pi * f)
    inv_freq = 2 * math.pi * freqs
    
    # Calculate angles based on sequence difference
    angles = seq_diff.unsqueeze(-1) * inv_freq
    
    # Concatenate sin/cos components
    rpe_feats = th.cat([th.sin(angles), th.cos(angles)], dim=-1)
    
    # Nullify RPE for BOTH cross-segment and cross-chain interactions.
    rpe_feats = rpe_feats * (1 - is_cross_segment) * (1 - is_cross_chain)
    
    # 5. Directional Polarity Flags
    # Cross-chain edges do not have a forward/backward N->C direction relative to each other.
    dir_forward = ((seq_diff > 0) & (u != v)).float().unsqueeze(-1) * (1 - is_cross_segment) * (1 - is_cross_chain)
    dir_backward = ((seq_diff < 0) & (u != v)).float().unsqueeze(-1) * (1 - is_cross_segment) * (1 - is_cross_chain)
    
    # 6. Concatenate final edge features
    edge_feats = th.cat([
        is_cross_segment,  # Index 0: Interface boundary flag
        is_cross_chain,    # Index 1: Chain boundary flag (CRITICAL for multimers)
        is_self,           # Index 2: True topological self-loop flag
        dir_forward,       # Index 3: Sequence direction (N -> C)
        dir_backward,      # Index 4: Sequence direction (C -> N)
        rpe_feats          # Indices 5+: Continuous position spectrum
    ], dim=-1)
    
    return edge_feats


import torch as th
import math

def compute_edge_features_sparse_v3(
    res_index: th.Tensor, 
    segment_id: th.Tensor, 
    chain_id: th.Tensor,
    u: th.Tensor, 
    v: th.Tensor, 
    num_rpe_freqs: int = 16,
    clamp_dist: float = 64.0
) -> th.Tensor:
    
    assert num_rpe_freqs % 2 == 0
    
    is_self = (u == v).float().unsqueeze(-1)
    is_cross_segment = (segment_id[u] != segment_id[v]).float().unsqueeze(-1)
    is_cross_chain = (chain_id[u] != chain_id[v]).float().unsqueeze(-1)
    
    # 1. Sequence Difference
    seq_diff_raw = (res_index[v] - res_index[u]).float()
    seq_diff = seq_diff_raw * (1 - is_cross_chain).squeeze(-1) 
    seq_diff = th.clamp(seq_diff, min=-clamp_dist, max=clamp_dist)
    
    # NEW: Pass normalized sequential distance explicitly so the MLP doesn't 
    # have to decode it from sine waves for distance regression.
    seq_dist_norm = (seq_diff.unsqueeze(-1) / clamp_dist)
    
    # 2. Revert to Exponential RPE (Better for regression tasks)
    inv_freq = th.exp(
        th.arange(0, num_rpe_freqs, 2, dtype=th.float32, device=res_index.device) 
        * -(math.log(100.0) / num_rpe_freqs)
    )
    angles = seq_diff.unsqueeze(-1) * inv_freq
    rpe_feats = th.cat([th.sin(angles), th.cos(angles)], dim=-1)
    rpe_feats = rpe_feats * (1 - is_cross_segment) * (1 - is_cross_chain)
    
    # 3. Polarity
    dir_forward = ((seq_diff > 0) & (u != v)).float().unsqueeze(-1) * (1 - is_cross_segment) * (1 - is_cross_chain)
    dir_backward = ((seq_diff < 0) & (u != v)).float().unsqueeze(-1) * (1 - is_cross_segment) * (1 - is_cross_chain)
    
    edge_feats = th.cat([
        is_cross_segment,  
        is_cross_chain,    
        is_self,           
        dir_forward,       
        dir_backward, 
        seq_dist_norm,     # <-- NEW: Explicit clamped distance
        rpe_feats          
    ], dim=-1)
    
    return edge_feats