import numpy as np
from scipy.linalg import block_diag
from scipy.linalg import eigvalsh
from toqito.channels import depolarizing
from toqito.state_props import is_ppt
from toqito.channel_ops import apply_channel
from toqito.matrices import gen_pauli_z as clock, gen_pauli_x as shift
from toqito.matrix_props import is_positive_semidefinite
np.set_printoptions(precision=5, suppress=True)
from fid_vectors import fid_vector_list
import time
import jax
import jax.numpy as jnp
import pandas as pd


def depol_jax(rho, p):
    d = rho.shape[0]
    return (1-p)* rho + p* (jnp.eye(d) / d) * jnp.trace(rho)

def compute_ptm_jax(channel_func_single_arg, d):
    """
    Nimmt eine Funktion, die NUR rho akzeptiert: f(rho) -> rho_out
    """
    # Basis erstellen
    basis_matrices = get_orthonormal_basis_first_identity(d) # Deine Funktion von vorhin
    basis_stack = jnp.array(basis_matrices)
    
    vectorized_channel = jax.jit(jax.vmap(channel_func_single_arg))
    
    outputs_stack = vectorized_channel(basis_stack)
    
    basis_flat = basis_stack.reshape(basis_stack.shape[0], -1)
    outputs_flat = outputs_stack.reshape(outputs_stack.shape[0], -1)
    
    ptm = jnp.conj(basis_flat) @ outputs_flat.T
    return ptm


def get_orthonormal_basis_first_identity(d: int):
    """
    Generates the d^2 Orthonormal Generalized Gell-Mann matrices.
    IMPORTANT: The normalized Identity is placed at index 0.
    
    Args:
        d (int): Dimension of the Hilbert space.
        
    Returns:
        list[np.ndarray]: List of basis matrices E_0, ..., E_{d^2-1}.
    """
    basis = []
    
    basis.append(np.eye(d, dtype=complex) / np.sqrt(d))
    
    for k in range(d):
        for j in range(k):
            mat = np.zeros((d, d), dtype=complex)
            mat[j, k] = 1
            mat[k, j] = 1
            basis.append(mat / np.sqrt(2))
            
    for k in range(d):
        for j in range(k):
            mat = np.zeros((d, d), dtype=complex)
            mat[j, k] = -1j
            mat[k, j] = 1j
            basis.append(mat / np.sqrt(2))
            
    for l in range(1, d):
        mat = np.zeros((d, d), dtype=complex)
        coeff = np.sqrt(1 / (l * (l + 1)))
        for m in range(l):
            mat[m, m] = coeff
        mat[l, l] = -l * coeff
        basis.append(mat)
        
    return basis

def compute_ptm(channel_func, d: int) -> np.ndarray:
    """
    Computes the Pauli Transfer Matrix (PTM) for a black-box channel.
    
    Args:
        channel_func (callable): A function T(rho) -> rho_prime.
        d (int): Dimension of the system.
        
    Returns:
        np.ndarray: The d^2 x d^2 matrix M where M_ij = Tr(E_i @ T(E_j)).
    """
    basis = get_orthonormal_basis_first_identity(d)
    n = len(basis) # Should be d^2
    

    ptm = np.zeros((n, n), dtype=complex)
    

    for j in range(n):
        E_j = basis[j]
        

        output_rho = channel_func(E_j)
        
        for i in range(n):
            E_i = basis[i]
            

            val = np.trace(E_i @ output_rho)
            ptm[i, j] = val
            
    return ptm

def compute_ptm_vectorized(channel_func, d: int) -> np.ndarray:
    """
    Computes the Pauli Transfer Matrix (PTM) using vectorization.
    """
    basis_matrices = get_orthonormal_basis_first_identity(d)
    
    basis_flat = np.array([b.flatten() for b in basis_matrices])
    
    output_states = [channel_func(E_j) for E_j in basis_matrices]
        
    outputs_flat = np.array([rho.flatten() for rho in output_states])


    ptm = basis_flat.conj() @ outputs_flat.T
    
    return ptm

    


def heisenberg_weyl_toqito(x: int, z: int, d: int) -> np.ndarray:
    """
    Implements the Heisenberg-Weyl operator using toqito's generalized Pauli function.
    
    Toqito's `gen_pauli(k, 1, d)` corresponds to X.
    Toqito's `gen_pauli(k, 2, d)` corresponds to Z.
    
    However, toqito does not have a single function taking both x and z directly 
    as a combined W operator in one call, so we construct it similarly.
    """
    try:
        from toqito.matrices import gen_pauli
    except ImportError:
        return "Toqito not installed. Run `pip install toqito`."

    
    X_op = gen_pauli(x, 1, d)
    
    Z_op = gen_pauli(z, 2, d)
    
    return X_op @ Z_op


def depol_channel(rho):
    return apply_channel(rho,choi_depolarizing)

def get_antisymmetric_unitary(d):
    """
    Constructs the standard antisymmetric unitary matrix U for even dimension d.
    """
    if d % 2 != 0:
        raise ValueError("Breuer-Hall map requires an EVEN dimension.")
    

    sigma_block = np.array([[0, 1], [-1, 0]])
    

    return block_diag(*[sigma_block for _ in range(d // 2)])

def breuer_hall_map_single(op, U):
    """
    [AI-DOC] Applies the Breuer-Hall map to a single d x d operator.
    Phi(X) = Tr(X)I - X - U * X^T * U^dag
    """
    d = op.shape[0]
    I = np.eye(d)
    

    term1 = np.trace(op) * I
    

    term2 = -op
    


    term3 = - (U @ op.T @ U.conj().T)
    
    return term1 + term2 + term3

def check_breuer_hall_entanglement(rho, d):
    """
    Applies (Id ⊗ Phi_BH) to a bipartite state rho and checks positivity.
    
    Args:
        rho: The d^2 x d^2 density matrix of the bipartite state.
        d:   The dimension of a single subsystem (must be even).
        
    Returns:
        is_detected (bool): True if entanglement is detected (negative eigenvalues found).
        min_eig (float): The lowest eigenvalue (negative value indicates entanglement).
    """
    if rho.shape != (d**2, d**2):
        raise ValueError(f"State dimension {rho.shape} does not match d^2={d**2}.")


    U = get_antisymmetric_unitary(d)
    



    rho_prime = np.zeros_like(rho, dtype=complex)
    
    for i in range(d):
        for j in range(d):
           
            block = rho[i*d:(i+1)*d, j*d:(j+1)*d]
            

            mapped_block = breuer_hall_map_single(block, U)
            

            rho_prime[i*d:(i+1)*d, j*d:(j+1)*d] = mapped_block
            


    eigenvalues = eigvalsh(rho_prime)
    min_eig = np.min(eigenvalues)

    tolerance = -1e-10
    is_detected = min_eig < tolerance
    
    if is_detected:
        print('This Breuer Hall map says channel is not entanglement breaking')
    else:
        print('This Breuer Hall map can not make a statement')
    
    return is_detected, min_eig

def get_eigenvalues(matrix):
    """
    [AI-DOC] Returns the list of eigenvalues using NumPy.
    
    Args:
        matrix (np.ndarray): The square matrix.
        
    Returns:
        np.ndarray: Array of eigenvalues.
    """

    if np.allclose(matrix, matrix.conj().T):
        return np.linalg.eigvalsh(matrix)
    else:
        return np.linalg.eigvals(matrix)


def top_two_abs_values(arr: np.ndarray, decimals: int = 5):
    """
     Analyzes a 1D numpy array to find the counts of the highest 
    and second-highest absolute values, robust to floating point errors.

    Args:
        arr (np.ndarray): Input array of shape (n,).
        decimals (int): Number of decimal places to round to. 
                        Default is 10, which handles most numerical noise.

    Returns:
        dict: Statistics about the top two absolute values.
    """

    abs_vals = np.abs(arr)
    
le floating point noise
    abs_vals = np.round(abs_vals, decimals=decimals)
    
    unique_vals, counts = np.unique(abs_vals, return_counts=True)
    

    if len(unique_vals) == 0:
        return {
            "first_max_val": None, "first_max_count": 0,
            "second_max_val": None, "second_max_count": 0
        }


    first_max_val = unique_vals[-1]
    first_max_count = counts[-1]


    if len(unique_vals) >= 2:
        second_max_val = unique_vals[-2]
        second_max_count = counts[-2]
    else:

        second_max_val = None
        second_max_count = 0

    return {
        "first_max_val": first_max_val,
        "first_max_count": first_max_count,
        "second_max_val": second_max_val,
        "second_max_count": second_max_count
    }



    

def is_positive_semidefinite_jax(matrix, tol=1e-6):

    evals = jnp.linalg.eigvalsh(matrix)
    return jnp.min(evals) >= -tol


def compute_choi_jax(channel_func, d):
    """
    Berechnet die Choi-Matrix für einen beliebigen Kanal.
    
    Args:
        channel_func: Eine Funktion f(rho) -> rho (JAX-kompatibel).
        d: Dimension des Hilbertaums.
        
    Returns:
        C: Die Choi-Matrix der Dimension (d^2, d^2).
    """
    

    eye = jnp.eye(d, dtype=complex)
    input_states = jnp.einsum('ia,jb->ijab', eye, eye)
    

    inputs_flat = input_states.reshape(d*d, d, d)
    

    vectorized_channel = jax.vmap(channel_func)
    outputs_flat = vectorized_channel(inputs_flat) # Shape: (d^2, d, d)
    

    outputs_grid = outputs_flat.reshape(d, d, d, d)
    

    outputs_reordered = outputs_grid.transpose(0, 2, 1, 3)
    

    choi_matrix = outputs_reordered.reshape(d*d, d*d)
    
    return choi_matrix

    
    
    
    
    
    
    
    
    
    
    
