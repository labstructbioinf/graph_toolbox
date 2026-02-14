import pandas as pd
import torch

def helix_indices_in_sample(env_indices: list[tuple], chain_id, pdb_index) -> list[int]:
    """
    Get the indices of the helices in using the env_indices and chain_id.
    Args:
        env_indices (list[tuple]): List of tuples containing environment indices.
        chain_id (str): The chain ID to filter by.
        pdb_index (list): List of PDB indices.
    Returns:
        list[int]: List of indices of helices in the sample.
    """
    helix_indices = []
    for emb_idx, (env_id, env_chain) in enumerate(env_indices):
        if env_id in pdb_index and env_chain == chain_id:
            helix_indices.append(emb_idx)
    return helix_indices


def helix_indices_from_series(row: pd.Series) -> list[int]:
    """
    Get the indices of the helices using `row` with values `env_index`, `chain_id` and `pdb_index`
    """
    return helix_indices_in_sample(env_indices=row.env_index, chain_id=row.chain_id, pdb_index=row.pdb_index)


def calculate_crossing_angle_torch(xyz: torch.Tensor, segment_id: torch.Tensor):
    """
    Calculates the signed crossing angle and centroid distance between a helix 
    and its environment using PCA (SVD) on coordinates.

    Args:
        xyz (torch.Tensor): Shape (N, 3) coordinates of CA atoms.
        segment_id (torch.Tensor): Shape (N,) vector where 0 = Helix, 1 = Environment.
    
    Returns:
        angle_rad (torch.Tensor): Scalar tensor, signed crossing angle in radians.
        dist (torch.Tensor): Scalar tensor, distance between centroids.
        valid (bool): False if environment is too small to calculate axis.
    """
    # 1. Separate the segments
    mask_helix = (segment_id == 0)
    mask_env = (segment_id == 1)
    
    coords_h = xyz[mask_helix]
    coords_e = xyz[mask_env]
    
    # 2. Validation: Need at least 3 points to define a line/plane variance
    if coords_h.shape[0] < 3 or coords_e.shape[0] < 3:
        return torch.tensor(0.0), torch.tensor(0.0), False

    # 3. Helper: Get Principal Axis via SVD
    def get_principal_axis(coords):
        # Center the coordinates
        centroid = torch.mean(coords, dim=0)
        centered = coords - centroid
        
        # SVD: U, S, Vh = torch.linalg.svd(X)
        # Vh[0] is the principal direction (eigenvector with largest variance)
        # Note: torch.linalg.svd returns Vh (V hermitian/transpose)
        # We wrap in no_grad because SVD gradients can be unstable if eigenvalues are close
        # but for fixed geometry it's fine.
        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        axis = Vh[0] 
        return axis, centroid

    axis_h, center_h = get_principal_axis(coords_h)
    axis_e, center_e = get_principal_axis(coords_e)

    # 4. Orient Axes (Handle Parallel vs Anti-Parallel Ambiguity)
    # We want the packing angle, which is typically acute (-90 to 90).
    # If the vectors point in opposite directions (dot < 0), flip one.
    if torch.dot(axis_h, axis_e) < 0:
        axis_e = -axis_e

    # 5. Calculate Magnitude (Unsigned Angle)
    # Clamp dot product to [-1, 1] to avoid NaNs from floating point errors
    dot_prod = torch.clamp(torch.dot(axis_h, axis_e), -1.0, 1.0)
    angle_rad = torch.acos(dot_prod)
    
    # 6. Determine Sign (Chirality)
    # This is the "Knobs-into-Holes" detector.
    # We define the sign based on the projection of the cross product 
    # onto the separation vector (Center_Env - Center_Helix).
    # Rule: Left-handed supercoiling (typical CC) usually yields Negative angles.
    
    separation_vec = center_e - center_h
    dist = torch.norm(separation_vec)
    
    cross_prod = torch.cross(axis_h, axis_e)
    
    # Sign is the sign of the dot product between (Axis_H x Axis_E) and Separation_Vec
    chirality_sign = torch.sign(torch.dot(cross_prod, separation_vec))
    
    # If perfectly parallel, cross_prod is 0, sign is 0. 
    # We default to positive (1.0) to avoid zeroing out the angle if it's very small.
    if chirality_sign == 0: 
        chirality_sign = 1.0

    signed_angle_rad = angle_rad * chirality_sign
    return signed_angle_rad, dist, True