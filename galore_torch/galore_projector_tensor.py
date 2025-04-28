import torch
from tensorly.decomposition import tucker
from tensorly import tenalg 

# The GaLoreProjector class in Python implements a projection method using orthogonal matrix
# decomposition for low-rank approximation of gradients for general tensors of dimension >2.
# We use tensor decomposition using tensorly library: https://tensorly.org/stable/index.html
class GaLoreProjectorTensor:
    """
    A class that represents a projector for the GaLore algorithm.

    Args:
        rank (int): The rank of the projector.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        update_proj_gap (int, optional): The number of iterations between updating the orthogonal matrix. Defaults to 200.
        scale (float, optional): The scaling factor for the projected gradients. Defaults to 1.0.
        proj_type (str, optional): The projection type (not used for tensor projector). Defaults to "std".
    """

    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type="std"):
        # proj_type is not used but added for compatibility with GaLoreProjector
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.transformed_low_rank = None
        self.orig_shape = None
        
    def project(self, full_rank_grad, iter):
        """
        Projects the full-rank gradients onto the low-rank subspace.

        Args:
            full_rank_grad (torch.Tensor): The full-rank gradients.
            iter (int): The current iteration.

        Returns:
            torch.Tensor: The transformed low-rank gradients.
        """
        if self.ortho_matrix is None and iter % self.update_proj_gap == 0:
            self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank)    
        self.transformed_low_rank = self.transform(self.ortho_matrix, full_rank_grad)
        return self.transformed_low_rank

    def project_back(self, low_rank_grad):
        """
        Projects the low-rank gradients back to the full-rank space.

        Args:
            low_rank_grad (torch.Tensor): The low-rank gradients.

        Returns:
            torch.Tensor: The full-rank gradients.
        """
        full_rank_grad = self.inverse_transform(self.ortho_matrix, self.transformed_low_rank)     
        return full_rank_grad * self.scale
        
    # tensor decomposition
    def get_orthogonal_matrix(self, weights, rank_all):
        """
        Computes the orthogonal matrix using tensor decomposition.

        Args:
            weights (torch.Tensor): The weights to decompose.
            rank_all (int): The desired rank of the decomposition.

        Returns:
            tuple: A tuple containing the core and factors of the orthogonal matrix.
        """
        module_params = weights
        if module_params.data.dtype != torch.float:
            matrix = module_params.data.float()
        else:
            matrix = module_params.data
            
        # Store original shape for later reshaping
        self.orig_shape = matrix.shape
            
        # For tensor decomposition, we need to specify rank as a list with one value per dimension
        # Convert integer rank to a list of ranks for each dimension
        if isinstance(rank_all, int):
            rank_list = [rank_all] * matrix.ndim
        else:
            rank_list = rank_all
            
        try:
            # Use the proper rank format for tensor decomposition
            tucker_tensor = tucker(matrix, rank=rank_list)
            return tucker_tensor
        except Exception as e:
            # Fallback to simpler approach if tensor decomposition fails
            print(f"Warning: Tucker decomposition failed: {e}")
            print(f"Using simpler approach for tensor with shape {matrix.shape}")
            
            # Reshape tensor to 2D for standard decomposition
            orig_shape = matrix.shape
            matrix_2d = matrix.reshape(orig_shape[0], -1)
            
            # Use SVD for 2D decomposition
            U, S, Vh = torch.linalg.svd(matrix_2d, full_matrices=False)
            
            # Truncate to rank
            U = U[:, :rank_all]
            S = S[:rank_all]
            Vh = Vh[:rank_all, :]
            
            # Return in a format compatible with multi_mode_dot
            factors = [U]
            core = torch.diag(S).mm(Vh)
            
            # Return a tuple similar to tucker decomposition
            return (core, factors)

    def transform(self, tensor, x):
        """
        Transforms the input tensor using the factors of the orthogonal matrix.

        Args:
            tensor (tuple): A tuple containing the core and factors of the orthogonal matrix.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        core, factors = tensor
        
        # Check if we're using the fallback approach (just one factor)
        if len(factors) == 1:
            # Handle the fallback SVD approach
            # Reshape input tensor to 2D
            orig_shape = x.shape
            x_2d = x.reshape(orig_shape[0], -1)
            
            # Project using the single factor
            result = torch.matmul(factors[0].t(), x_2d)
            return result
        else:
            # Regular Tucker decomposition approach
            return tenalg.multi_mode_dot(x, factors, transpose=True)

    def inverse_transform(self, tensor, x):
        """
        Inverse transforms the input tensor using the factors of the orthogonal matrix.

        Args:
            tensor (tuple): A tuple containing the core and factors of the orthogonal matrix.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The inverse transformed tensor.
        """
        core, factors = tensor
        
        # Check if we're using the fallback approach (just one factor)
        if len(factors) == 1:
            # Handle the fallback SVD approach
            result = torch.matmul(factors[0], x)
            
            # Reshape back to original tensor shape
            # We need to infer the original shape from the factor dimensions
            factor_shape = factors[0].shape
            if hasattr(self, 'orig_shape') and self.orig_shape is not None:
                result = result.reshape(self.orig_shape)
            else:
                # Try to infer a reasonable shape if original shape not stored
                result = result.reshape(factor_shape[0], -1)
                
            return result
        else:
            # Regular Tucker decomposition approach
            return tenalg.multi_mode_dot(x, factors)
