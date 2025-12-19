import torch
import torch.nn.functional as F

def get_backbone_rotation_matrices(n: torch.Tensor, ca: torch.Tensor, c: torch.Tensor):
    """
    Constructs a local rotation matrix for each residue based on N, CA, C coordinates.
    
    Args:
        coords: Tensor of shape (Batch, Residues, 3, 3) 
                where the 3rd dim is atoms ordered [N, CA, C]
                and the 4th dim is [x, y, z].
                
    Returns:
        rot_matrices: Tensor of shape (Batch, Residues, 3, 3)
                      These matrices transform from the LOCAL frame to the GLOBAL frame.
                      Columns are the basis vectors [x_axis, y_axis, z_axis].
    """
    

    # 1. Define the first basis vector (u)
    # We use the direction from CA to N as the primary axis (X-axis)
    # This is an arbitrary choice, but must be consistent.
    v1 = n - ca
    u = F.normalize(v1, dim=-1)

    # 2. Define a second vector (v) to establish the plane
    # Direction from CA to C
    v2 = c - ca
    
    # 3. Construct the third basis vector (w) perpendicular to the N-CA-C plane (Z-axis)
    # Cross product of (CA->N) and (CA->C)
    # This captures the chirality and plane orientation
    cross_prod = torch.cross(u, v2, dim=-1)
    w = F.normalize(cross_prod, dim=-1)

    # 4. Construct the second basis vector (v) to ensure orthogonality (Y-axis)
    # Cross product of Z and X axes
    v = torch.cross(w, u, dim=-1)

    # 5. Stack to form the rotation matrix
    # Columns are [u, v, w] corresponding to [x, y, z] axes in the global frame.
    # shape: ( Residues, 3, 3)
    rot_matrices = torch.stack([u, v, w], dim=-1)
    
    return rot_matrices


def relative_rotation(R: torch.Tensor) -> torch.Tensor:
    """
    Calculates relative rotation for each pair of residues.
    
    Args:
        R: Tensor of shape (num_residues, 3, 3)
           Represents rotation matrices from local to global frame.
           
    Returns:
        torch.Tensor: (num_residues, num_residues, 3, 3)
                      Where output[i, j] = R_i.T @ R_j
                      This represents the rotation needed to go from Frame j to Frame i.
    """
    # 1. Prepare R_i (transpose and add dimension 1 for broadcasting)
    # Shape becomes: (N, 1, 3, 3)
    # We transpose the last two dimensions to get R_i^T
    Ri_T = R.unsqueeze(1).transpose(-1, -2)
    
    # 2. Prepare R_j (add dimension 0 for broadcasting)
    # Shape becomes: (1, N, 3, 3)
    Rj = R.unsqueeze(0)
    
    # 3. Perform Matrix Multiplication with Broadcasting
    # PyTorch will broadcast (N, 1) and (1, N) into (N, N)
    # The matmul operates on the last two dims (3, 3)
    pairwise_rotations = torch.matmul(Ri_T, Rj)
    
    return pairwise_rotations

def calculate_relative_rotation_matrix(n, ca, c):
    R = get_backbone_rotation_matrices(n, ca, c)
    Relij = relative_rotation(R)
    return Relij

# --- Example Usage ---
if __name__ == "__main__":
    # Create dummy data: 1 protein, 5 residues, 3 atoms (N, CA, C), 3D coords
    batch_size = 1
    num_residues = 5
    dummy_coords = torch.randn(num_residues, 3, 3)
    n, ca, c = torch.randn(num_residues, 3), torch.randn(num_residues, 3), torch.randn(num_residues, 3)
    # Get matrices
    R = get_backbone_rotation_matrices(n, ca, c)
    
    print("Input shape:", dummy_coords.shape)
    print("Output Rotation Matrix shape:", R.shape) # Expect (1, 5, 3, 3)
    
    # Verification: The columns should be orthogonal and unit length.
    # Check R * R_transpose is Identity
    identity_check = torch.matmul(R, R.transpose(-1, -2))
    print("\nOrthogonality check (first residue):\n", identity_check[0, 0])
    Relij = relative_rotation(R)
    print("relative rotation", Relij)
    print("shape", Relij.shape)
    breakpoint()