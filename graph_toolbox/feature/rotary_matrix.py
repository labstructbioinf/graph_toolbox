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

def calculate_relative_rotation_matrix(n: torch.Tensor, ca: torch.Tensor, c: torch.Tensor):
    """
    calculates relative rotations for each residue pair for a given tensor of atoms
    Returns:
        torch.Tensor: (N, N, 3, 3)
    """
    R = get_backbone_rotation_matrices(n, ca, c)
    Relij = relative_rotation(R)
    return Relij

def get_sidechain_rotation_matrices(ca: torch.Tensor, cb: torch.Tensor, cg: torch.Tensor):
    """
    Constructs a local rotation matrix for the side chain (chi1 level).
    
    Args:
        ca, cb, cg: Tensors of shape (Residues, 3) 
                    Note: For Glycine, CG is usually approximated or ignored.
                
    Returns:
        rot_matrices: (Residues, 3, 3) - Local side-chain frame to Global.
    """
    # 1. Primary axis (X-axis): CA -> CB bond
    v1 = cb - ca
    u = F.normalize(v1, dim=-1)

    # 2. Secondary vector: CB -> CG to define the chi1 plane
    v2 = cg - cb
    
    # 3. Z-axis: Normal to the CA-CB-CG plane
    cross_prod = torch.cross(u, v2, dim=-1)
    w = F.normalize(cross_prod, dim=-1)

    # 4. Y-axis: Orthonormal completion
    v = torch.cross(w, u, dim=-1)

    # Stack to form (Residues, 3, 3)
    rot_matrices = torch.stack([u, v, w], dim=-1)
    
    return rot_matrices

def calculate_relative_sidechain_rotation(ca: torch.Tensor, cb: torch.Tensor, cg: torch.Tensor):
    """
    Calculates relative rotations between all side-chain frames.
    """
    R_sc = get_sidechain_rotation_matrices(ca, cb, cg)
    
    # Reusing your efficient relative_rotation logic
    # Ri_T @ Rj
    Ri_T = R_sc.unsqueeze(1).transpose(-1, -2)
    Rj = R_sc.unsqueeze(0)
    
    return torch.matmul(Ri_T, Rj)
    

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
    
    ca, cb, cg = torch.randn(num_residues, 3), torch.randn(num_residues, 3), torch.randn(num_residues, 3)
    R = calculate_relative_sidechain_rotation(ca, cb, cg)
    identity_check = torch.matmul(R, R.transpose(-1, -2))
    print("\nOrthogonality check (first residue):\n", identity_check[0, 0])