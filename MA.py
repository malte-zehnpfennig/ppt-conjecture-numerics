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
import numpy as np


# [AI] 1. Deine Funktion muss JAX-kompatibel sein (nutze jnp statt np)
# [AI] Beispiel:
def depol_jax(rho, p):
    # [AI] Wichtig: JAX Arrays sind "immutable".
    # [AI] Statt rho[0,0] = 1 musst du rho.at[0,0].set(1) nutzen, falls du schreibst.
    # [AI] Aber für normale Matrix-Ops (dot, trace) ist es fast gleich.
    d = rho.shape[0]
    # [AI] Beispiel Depolarizing:
    return (1-p)* rho + p* (jnp.eye(d) / d) * jnp.trace(rho)

def compute_ptm_jax(channel_func_single_arg, d):
    """
    [AI-DOC] Nimmt eine Funktion, die NUR rho akzeptiert: f(rho) -> rho_out
    """
    # [AI] Basis erstellen
    basis_matrices = get_orthonormal_basis_first_identity(d) # [AI] Deine Funktion von vorhin
    basis_stack = jnp.array(basis_matrices)
    
    # [AI] --- JAX MAGIE ---
    # [AI] Wir sagen JAX: "Vektorisiere diese Funktion, die EIN Argument nimmt"
    vectorized_channel = jax.jit(jax.vmap(channel_func_single_arg))
    
    # [AI] Jetzt rufen wir sie nur mit dem Stack auf (kein p nötig, ist schon drin)
    outputs_stack = vectorized_channel(basis_stack)
    
    # [AI] Rest wie gehabt...
    basis_flat = basis_stack.reshape(basis_stack.shape[0], -1)
    outputs_flat = outputs_stack.reshape(outputs_stack.shape[0], -1)
    
    ptm = jnp.conj(basis_flat) @ outputs_flat.T
    return ptm


def get_orthonormal_basis_first_identity(d: int):
    """
    [AI-DOC] Generates the d^2 Orthonormal Generalized Gell-Mann matrices.
    IMPORTANT: The normalized Identity is placed at index 0.
    
    Args:
        d (int): Dimension of the Hilbert space.
        
    Returns:
        list[np.ndarray]: List of basis matrices E_0, ..., E_{d^2-1}.
    """
    basis = []
    
    # [AI] 1. Normalized Identity (Placed FIRST)
    # [AI] E_0 = I / sqrt(d)
    basis.append(np.eye(d, dtype=complex) / np.sqrt(d))
    
    # [AI] 2. Symmetric Off-Diagonal
    for k in range(d):
        for j in range(k):
            mat = np.zeros((d, d), dtype=complex)
            mat[j, k] = 1
            mat[k, j] = 1
            basis.append(mat / np.sqrt(2))
            
    # [AI] 3. Anti-Symmetric Off-Diagonal
    for k in range(d):
        for j in range(k):
            mat = np.zeros((d, d), dtype=complex)
            mat[j, k] = -1j
            mat[k, j] = 1j
            basis.append(mat / np.sqrt(2))
            
    # [AI] 4. Diagonal (Traceless)
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
    [AI-DOC] Computes the Pauli Transfer Matrix (PTM) for a black-box channel.
    
    Args:
        channel_func (callable): A function T(rho) -> rho_prime.
        d (int): Dimension of the system.
        
    Returns:
        np.ndarray: The d^2 x d^2 matrix M where M_ij = Tr(E_i @ T(E_j)).
    """
    # [AI] 1. Get the basis E_0, ..., E_{d^2-1}
    basis = get_orthonormal_basis_first_identity(d)
    n = len(basis) # [AI] Should be d^2
    
    # [AI] 2. Initialize the empty PTM
    ptm = np.zeros((n, n), dtype=complex)
    
    # [AI] 3. Compute column by column (to minimize calls to channel_func)
    # [AI] We compute T(E_j) once, then project onto all E_i
    for j in range(n):
        E_j = basis[j]
        
        # [AI] Apply the black box channel
        output_rho = channel_func(E_j)
        
        # [AI] Project onto the dual basis elements E_i
        # [AI] Since basis is orthonormal Hermitian, dual is just E_i itself
        for i in range(n):
            E_i = basis[i]
            
            # [AI] M_ij = Tr(E_i @ T(E_j))
            # [AI] Note: Since basis is Hermitian, E_i^dagger = E_i.
            val = np.trace(E_i @ output_rho)
            ptm[i, j] = val
            
    return ptm

def compute_ptm_vectorized(channel_func, d: int) -> np.ndarray:
    """
    [AI-DOC] Computes the Pauli Transfer Matrix (PTM) using vectorization.
    """
    # [AI] 1. Generate the Basis (List of d^2 matrices)
    basis_matrices = get_orthonormal_basis_first_identity(d)
    
    # [AI] 2. Vectorize the Basis (Flatten matrices into vectors)
    # [AI] Stack them into a matrix B of shape (d^2, d^2)
    # [AI] Row i of B contains the flattened vector of Basis element i.
    basis_flat = np.array([b.flatten() for b in basis_matrices])
    
    # [AI] 3. Apply Channel to Basis (The only loop remaining)
    # [AI] We run the channel d^2 times.
    output_states = [channel_func(E_j) for E_j in basis_matrices]
        
    # [AI] 4. Vectorize the Outputs
    # [AI] Stack them into a matrix Phi of shape (d^2, d^2)
    outputs_flat = np.array([rho.flatten() for rho in output_states])
    
    # [AI] 5. Compute PTM via Matrix Multiplication
    # [AI] M_ij = Tr(E_i @ T(E_j)) -> Dot product of flattened vectors
    # [AI] This effectively computes all d^4 elements in one optimized BLAS call.
    ptm = basis_flat.conj() @ outputs_flat.T
    
    return ptm

    


def heisenberg_weyl_toqito(x: int, z: int, d: int) -> np.ndarray:
    """
    [AI-DOC] Implements the Heisenberg-Weyl operator using toqito's generalized Pauli function.
    
    Toqito's `gen_pauli(k, 1, d)` corresponds to X.
    Toqito's `gen_pauli(k, 2, d)` corresponds to Z.
    
    However, toqito does not have a single function taking both x and z directly 
    as a combined W operator in one call, so we construct it similarly.
    """
    try:
        from toqito.matrices import gen_pauli
    except ImportError:
        return "Toqito not installed. Run `pip install toqito`."

    # [AI] gen_pauli(ind, axis, dim)
    # [AI] axis=1 is X, axis=2 is Z.
    
    # [AI] Note: gen_pauli(x, 1, d) computes X^x directly
    X_op = gen_pauli(x, 1, d)
    
    # [AI] Note: gen_pauli(z, 2, d) computes Z^z directly
    Z_op = gen_pauli(z, 2, d)
    
    return X_op @ Z_op


def depol_channel(rho):
    return apply_channel(rho,choi_depolarizing)

def get_sic_povm(d, fiducial_vector):
    """
    [AI-DOC] Generates a SIC-POVM from a fiducial vector using Weyl-Heisenberg operators.
    """
    X = shift(d)
    Z = clock(d)
    
    povm_elements = []
    # [AI] Displacement operators D_{a,b} = X^a Z^b (ignoring phase factors for POVM)
    for a in range(d):
        for b in range(d):
            # [AI] D = (X^a) @ (Z^b)
            op = np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, b)
            vec = op @ fiducial_vector
            # [AI] POVM element E_k = 1/d * |psi><psi|
            povm_elements.append((1/d) * np.outer(vec, vec.conj()))
            
    return povm_elements

def get_antisymmetric_unitary(d):
    """
    [AI-DOC] Constructs the standard antisymmetric unitary matrix U for even dimension d.
    """
    if d % 2 != 0:
        raise ValueError("Breuer-Hall map requires an EVEN dimension.")
    
    # [AI] Define the 2x2 symplectic block [[0, 1], [-1, 0]]
    sigma_block = np.array([[0, 1], [-1, 0]])
    
    # [AI] Create the block diagonal matrix U = J ⊕ J ⊕ ... ⊕ J
    return block_diag(*[sigma_block for _ in range(d // 2)])

def breuer_hall_map_single(op, U):
    """
    [AI-DOC] Applies the Breuer-Hall map to a single d x d operator.
    Phi(X) = Tr(X)I - X - U * X^T * U^dag
    """
    d = op.shape[0]
    I = np.eye(d)
    
    # [AI] Term 1: Tr(X) * I
    term1 = np.trace(op) * I
    
    # [AI] Term 2: - X
    term2 = -op
    
    # [AI] Term 3: - U * X^T * U^dag
    # [AI] U is real-valued, so U_dagger = U.T = -U. We use conj().T for rigor.
    term3 = - (U @ op.T @ U.conj().T)
    
    return term1 + term2 + term3

def check_breuer_hall_entanglement(rho, d):
    """
    [AI-DOC] Applies (Id ⊗ Phi_BH) to a bipartite state rho and checks positivity.
    
    Args:
        rho: The d^2 x d^2 density matrix of the bipartite state.
        d:   The dimension of a single subsystem (must be even).
        
    Returns:
        is_detected (bool): True if entanglement is detected (negative eigenvalues found).
        min_eig (float): The lowest eigenvalue (negative value indicates entanglement).
    """
    if rho.shape != (d**2, d**2):
        raise ValueError(f"State dimension {rho.shape} does not match d^2={d**2}.")

    # [AI] Pre-calculate U once to save time
    U = get_antisymmetric_unitary(d)
    
    # [AI] Initialize the output matrix
    # [AI] We treat rho as a block matrix of d x d blocks and apply the map to each block.
    # [AI] (Id ⊗ Phi)( |i><j| ⊗ A ) = |i><j| ⊗ Phi(A)
    rho_prime = np.zeros_like(rho, dtype=complex)
    
    for i in range(d):
        for j in range(d):
            # [AI] Extract the (i, j)-th d x d block corresponding to subsystem B
            # [AI] Row indices: i*d to (i+1)*d
            # [AI] Col indices: j*d to (j+1)*d
            block = rho[i*d:(i+1)*d, j*d:(j+1)*d]
            
            # [AI] Apply the map to this block
            mapped_block = breuer_hall_map_single(block, U)
            
            # [AI] Place it back into the result matrix
            rho_prime[i*d:(i+1)*d, j*d:(j+1)*d] = mapped_block
            
    # [AI] Check eigenvalues of the resulting matrix
    # [AI] We use eigvalsh because the result should be Hermitian
    eigenvalues = eigvalsh(rho_prime)
    min_eig = np.min(eigenvalues)
    
    # [AI] If the minimum eigenvalue is significantly negative, it's entangled.
    # [AI] We use a small tolerance to avoid numerical noise (e.g., -1e-12).
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
    # [AI] Check if matrix is Hermitian (A == A^dagger)
    # [AI] If so, use 'eigh' which is numerically more stable for Hermitian matrices
    if np.allclose(matrix, matrix.conj().T):
        return np.linalg.eigvalsh(matrix)
    else:
        return np.linalg.eigvals(matrix)


def top_two_abs_values(arr: np.ndarray, decimals: int = 5):
    """
    [AI-DOC] Analyzes a 1D numpy array to find the counts of the highest 
    and second-highest absolute values, robust to floating point errors.

    Args:
        arr (np.ndarray): Input array of shape (n,).
        decimals (int): Number of decimal places to round to. 
                        Default is 10, which handles most numerical noise.

    Returns:
        dict: Statistics about the top two absolute values.
    """
    # [AI] 1. Compute absolute values
    abs_vals = np.abs(arr)
    
    # [AI] 2. CRITICAL STEP: Round to handle floating point noise
    # [AI] This groups 0.300000001 and 0.299999999 together as 0.3
    abs_vals = np.round(abs_vals, decimals=decimals)
    
    # [AI] 3. Find unique values and their counts
    unique_vals, counts = np.unique(abs_vals, return_counts=True)
    
    # [AI] 4. Handle Edge Case: Empty Array
    if len(unique_vals) == 0:
        return {
            "first_max_val": None, "first_max_count": 0,
            "second_max_val": None, "second_max_count": 0
        }

    # [AI] 5. Extract First Highest (Last element in sorted array)
    first_max_val = unique_vals[-1]
    first_max_count = counts[-1]

    # [AI] 6. Extract Second Highest (Second to last element)
    if len(unique_vals) >= 2:
        second_max_val = unique_vals[-2]
        second_max_count = counts[-2]
    else:
        # [AI] Edge Case: All numbers have the same absolute value
        second_max_val = None
        second_max_count = 0

    return {
        "first_max_val": first_max_val,
        "first_max_count": first_max_count,
        "second_max_val": second_max_val,
        "second_max_count": second_max_count
    }


def norm_condition(ptm, d):
    frobenius = np.linalg.norm(ptm, ord='fro')
    if frobenius > np.sqrt(d):
        print('Norm condition says channel is not e.b. since norm too big')
    elif frobenius <= np.sqrt(1/(d**2-1)+1):
        print('Norm condition says channel is e.b. since norm too small')
    else:
        print('Norm condition can not make a statement')
        
def norm_condition_power(ptm, ev, d, n):
    norm = np.linalg.norm(ev, ord=2*n)**(2*n)
    if (norm > d):
        print(f'Channel^{n} is not e.b.')
    else:
        print(f'Can not make a statement about this power {n} via norm condition')

def bound_norm_condition(ptm, ev, d):
    frobenius = np.linalg.norm(ptm, ord='fro')
    temp = top_two_abs_values(ev)
    g = temp["first_max_count"]
    D = temp["second_max_count"]
    if D != (d**2 -1):
        print('STOP! Rundungsfehler')
    max_2 = temp["second_max_val"]
    temp = np.log((d-g)/D)/(2*np.log(max_2))
    
    temp2 = -1/2*np.log(d**2-1)/np.log(np.sqrt(frobenius**2-1))
    
    print(f'Norm condition says the e.b. index is greater equal {temp}')
    if frobenius >= d/np.sqrt(d**2-1) and frobenius <= np.sqrt(2):
        print(f'Contraction condition says the e.b. index is smaller {temp2}')
    else:
        print('Contraction condition: no upper bound bound possible since norm not in range')
    return temp
        

# [AI] Hilfsfunktion für JAX (da JAX kein direktes is_positive_semidefinite hat)
def is_positive_semidefinite_jax(matrix, tol=1e-6):
    # [AI] Wir nutzen eigvalsh, da die Matrix Hermitesch sein sollte
    evals = jnp.linalg.eigvalsh(matrix)
    return jnp.min(evals) >= -tol

# [AI] Deine angepasste Funktion
def sic_condition_jax(channel_func, d):
    """
    [AI-DOC] channel_func: Deine 'channel_ready' Funktion (lambda), die nur ein Argument (rho) nimmt.
    """
    
    # [AI] 1. Fidelity Vector holen (darf numpy bleiben, wird gleich umgewandelt)
    # [AI] Annahme: fid_vector_list ist global verfügbar oder wird hier genutzt
    fid_vector = jnp.array(fid_vector_list[d-2]) 
    
    # [AI] Normieren
    fid_vector = fid_vector / jnp.linalg.norm(fid_vector)
    
    # [AI] SIC POVM generieren (Wir nehmen an get_sic_povm gibt eine Liste von Matrizen zurück)
    # [AI] Wir wandeln das Ergebnis direkt in JAX Arrays um
    sic_povm_np = get_sic_povm(d, fid_vector) 
    sic_povm = [jnp.array(M) for M in sic_povm_np]
    
    temp_sic = True
    
    # [AI] Schleife über die Elemente
    for i in sic_povm:
        # [AI] HIER: Generischer Aufruf statt depol_channel
        channel_out = channel_func(i)
        
        # [AI] Die Formel (unverändert)
        temp = (d + 1) * channel_out - jnp.eye(d) / d
        
        # [AI] Positivität prüfen
        if not is_positive_semidefinite_jax(temp):
            temp_sic = False
            # [AI] Optional: break, um Zeit zu sparen, wenn einer fehlschlägt
            break 

    if temp_sic:
        print('SICs say channel is entanglement breaking')
    else:
        print('SICs can not make a statement')
        
    return temp_sic


def compute_choi_jax(channel_func, d):
    """
    [AI-DOC] Berechnet die Choi-Matrix für einen beliebigen Kanal.
    
    Args:
        channel_func: Eine Funktion f(rho) -> rho (JAX-kompatibel).
        d: Dimension des Hilbertaums.
        
    Returns:
        C: Die Choi-Matrix der Dimension (d^2, d^2).
    """
    
    # [AI] 1. Erzeuge alle Basis-Matrizen |i><j| auf einmal
    # [AI] Wir nutzen einsum für ein äußeres Produkt aller Basisvektoren.
    # [AI] Das Ergebnis ist ein Tensor (d, d, d, d), wobei index [i,j] die Matrix |i><j| ist.
    eye = jnp.eye(d, dtype=complex)
    input_states = jnp.einsum('ia,jb->ijab', eye, eye)
    
    # [AI] 2. Flachklopfen für vmap: (d*d, d, d)
    # [AI] Jetzt haben wir eine Liste von d^2 Matrizen.
    inputs_flat = input_states.reshape(d*d, d, d)
    
    # [AI] 3. Kanal parallel auf alle |i><j| anwenden
    # [AI] Wir vektorisieren die Funktion, damit sie den ganzen Stapel auf einmal frisst.
    vectorized_channel = jax.vmap(channel_func)
    outputs_flat = vectorized_channel(inputs_flat) # [AI] Shape: (d^2, d, d)
    
    # [AI] 4. Reshape zurück in Block-Struktur: (d, d, d, d)
    # [AI] Dimensionen: (block_row, block_col, inner_row, inner_col)
    outputs_grid = outputs_flat.reshape(d, d, d, d)
    
    # [AI] 5. Die magische Transposition (WICHTIG!)
    # [AI] Wir müssen die Achsen vertauschen, um die Block-Matrix flach zu machen.
    # [AI] Von: (Block_Zeile, Block_Spalte, Innen_Zeile, Innen_Spalte)
    # [AI] Zu:  (Block_Zeile, Innen_Zeile, Block_Spalte, Innen_Spalte)
    # [AI] Damit liegen die Zeilen, die im Endbild nebeneinander gehören, auch im Speicher nebeneinander.
    outputs_reordered = outputs_grid.transpose(0, 2, 1, 3)
    
    # [AI] 6. Finales Reshape zur 2D-Matrix
    choi_matrix = outputs_reordered.reshape(d*d, d*d)
    
    return choi_matrix

    
dim = 6
for d in range(dim, dim +1, 1):
    p_thresh = d/(d+1) #for all p bigger equal this the depolarizing channel is e.b.
    p = p_thresh
    p = 0.8
    p = 0.001
    p = 0.84
    print(f'dim is {d}, threshold is {p_thresh}, p is {p}')
    if p == 0:
        print('Channel is not eeb')
    else:
        index_lower_bound = -np.log(d+1)/np.log(1-p)
        print(f'The e.b. index of this channel is bigger equal {index_lower_bound}')
    if p >= p_thresh:
        print(f'Depolarizing channel in d = {d} is e.b.')
    else: 
        print('Depolarizing channel is not e.b.')
    
    channel_ready = lambda rho: depol_jax(rho, p)
    
    # [AI] Jetzt passt es perfekt in den Solver
    ptm = compute_ptm_jax(channel_ready, d)
    ev = get_eigenvalues(ptm)

    check_breuer_hall_entanglement(compute_choi_jax(channel_ready, d), d)
    sic_condition_jax(channel_ready, d)
    # temp1 = bound_norm_condition(ptm, ev, d)
    # norm_condition(ptm, d)
    # start = int(temp1/2)
    # end = int(index_lower_bound)
    # # for n in range(start, end, max(int((end- start)/40), 1)):
    # #     norm_condition_power(ptm, ev, d, n)
    # print(np.linalg.norm(ptm, ord='fro'), np.sqrt(1/(d**2+1)+1), np.sqrt(2))
    # print()

    
    
    
    
    
    
    
    
    
    
    
