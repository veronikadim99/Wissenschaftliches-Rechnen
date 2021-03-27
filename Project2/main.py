import numpy as np
import tomograph


def gaussian_elimination(A: np.ndarray, b: np.ndarray, use_pivoting: bool = True) -> (np.ndarray, np.ndarray):
    """
    Gaussian Elimination of Ax=b with or without pivoting without use of numpy.linalg

    Arguments:
    A : matrix, representing left side of equation system of size: (m,m)
    b : vector, representing right hand side of size: (m, )
    use_pivoting : flag if pivoting should be used

    Return:
    A : reduced result matrix in row echelon form (type: np.ndarray, size: (m,m))
    b : result vector in row echelon form (type: np.ndarray, size: (m, ))

    Raised Exceptions:
    ValueError: if matrix and vector sizes are incompatible, matrix is not square or pivoting is disabled but necessary

    Side Effects:
    -
    """

    # Test if shape of matrix and vector is compatible and raise ValueError if not
    (n, n) = A.shape

    if not use_pivoting:
        for i in range(0, n):
            if A[[i], [i]] == 0:
                raise ValueError('Pivoting disabled but necessary')

    if len(A) != len(b): raise ValueError('The sizes are incompatible')
    if len(A[0]) != len(A): raise ValueError('The matrix is not square')

    # Perform gaussian elimination

    if use_pivoting:
        for i in range(0, n):
            for j in range(i + 1, n):
                if j > i:
                    if abs(A[[i], [i]]) < abs(A[[j], [i]]):
                        A[[i, j]] = A[[j, i]]
                        b[j], b[i] = b[i], b[j]
            for k in range(i + 1, n):
                if k > i:
                    factor = -(A[[k], [i]]) / (A[[i], [i]])
                    for j in range(i, n):
                        A[[k], [j]] = A[[k], [j]] + (factor * (A[[i], [j]]))
                b[k] = b[k] + (factor * (b[i]))

    if not use_pivoting:
        for i in range(0, n):
            for k in range(1, n):
                if k > i:
                    factor = -(A[[k], [i]]) / (A[[i], [i]])
                    for j in range(0, n):
                        A[[k], [j]] = A[[k], [j]] + (factor * (A[[i], [j]]))

                    b[k] = b[k] + (factor * (b[i]))

    return A, b


def back_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Back substitution for the solution of a linear system in row echelon form without use of numpy.linalg.

    Arguments:
    A : matrix in row echelon representing linear system
    b : vector, representing right hand side

    Return:
    x : solution of the linear system

    Raised Exceptions:
    ValueError: if matrix/vector sizes are incompatible or no/infinite solutions exist

    Side Effects:
    -
    """
    (n, n) = A.shape

    if len(A[0]) != len(b): raise ValueError('Matrix/vector sizes are incompatible')
    if A[len(A) - 1][len(A) - 1] == 0: raise ValueError('No/infinite solutions exist')

    # initialize solution vector with proper size
    x = np.zeros(n)

    # run backsubstitution and fill solution vector, raise ValueError if no/infinite solutions exist

    x[n - 1] = b[n - 1] / A[n - 1][n - 1]  # start with the last element
    for i in reversed(range(0, n)):
        k = 0
        for j in range(0, n):
            if j != i:
                k += -(A[[i], [j]] * (x[j]))
        x[i] = (b[i] + k) / (A[[i], [i]])

    return x


def compute_cholesky(M: np.ndarray) -> np.ndarray:
    """
    Compute Cholesky decomposition of a matrix without use of np.linalg.

    Arguments:
    M : matrix, symmetric and positive (semi-)definite

    Raised Exceptions:
    ValueError: L is not symmetric and psd

    Return:
    L :  Cholesky factor of M


    """

    (n, m) = M.shape
    T = np.zeros((m, n))

    if n != m: raise ValueError('The matrix does not meet the requirements for the Cholesky decomposition')
    for i in range(0, n):
        for j in range(0, m):
            T[[i], [j]] = M[[j], [i]]

    if not np.allclose(M, T, 1e-10, 1e-7):
        raise ValueError('The matrix does not meet the requirements for the Cholesky decomposition')

    # build the factorization and raise a ValueError in case of a non-positive definite input matrix

    L = np.zeros((n, n))
    # find L with all the formula from the skript
    # set first element of L
    L[[0], [0]] = (M[[0], [0]]) ** (1 / 2)
    # set the other elements from the first row
    for i in range(0, n):
        L[[i], [0]] = (M[[0], [i]]) / (L[[0], [0]])
    # diagonalelement

    # we are building L using the formula
    for i in range(1, n):
        sum_diag = 0
        # diagonal
        for j in range(0, i + 1):
            sum_diag = sum_diag + (L[[i], [j]]) ** (2)
        # raise valuer error if diagonal is negative number
        if (M[[i], [i]]) <= 0 or ((M[[i], [i]]) - sum_diag) < 0: raise ValueError('The matrix does not meet the '
                                                                                  'requirements for the Cholesky '
                                                                                  'decomposition')
        L[[i], [i]] = ((M[[i], [i]]) - sum_diag) ** (1 / 2)

        # the other numbers in the column
        for j in range(i + 1, n):
            sum_f = 0
            for k in range(0, n):
                if k < i:
                    sum_f += (L[[i], [k]]) * (L[[j], [k]])
            L[[j], [i]] = ((M[[j], [i]]) - sum_f) / (L[[i], [i]])

    return L


def solve_cholesky(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the system L L^T x = b where L is a lower triangular matrix

    Arguments:
    L : matrix representing the Cholesky factor
    b : right hand side of the linear system

    Raised Exceptions:
    ValueError: sizes of L, b do not match
    ValueError: L is not lower triangular matrix

    Return:
    x : solution of the linear system

    Forbidden:
    - numpy.linalg.*
    """

    # check the input for validity
    (n, m) = L.shape

    if len(L) != len(L[0]): raise ValueError('The matrix does not meet the requirements for the Cholesky decomposition')
    if len(b) != len(L): raise ValueError('The matrix does not meet the requirements for the Cholesky decomposition')
    for i in range(0, n):
        for j in range(1, n):
            if (j != i) & (j > i):
                if L[[i], [j]] != 0: raise ValueError('The matrix does not meet the requirements for the Cholesky decomposition')

    # solve the system by forward- and backsubstitution

    x = np.zeros(m)
    A = np.zeros((m, m))


    for i in range(0, m):
        for j in range(0, m):
            A[[i], [j]] = L[[j], [i]]

    # forward substiotution for L, find x
    x[0] = b[0] / L[0][0]  # start with the last element
    for i in range(1, m):
        var = 0
        for j in range(0, m):
            if j != i:
                var += -(L[[i], [j]] * (x[j]))
        x[i] = (b[i] + var) / (L[[i], [i]])

    x = back_substitution(A, x)

    return x


def setup_system_tomograph(n_shots: np.int, n_rays: np.int, n_grid: np.int) -> (np.ndarray, np.ndarray):
    """
    Set up the linear system describing the tomographic reconstruction

    Arguments:
    n_shots  : number of different shot directions
    n_rays   : number of parallel rays per direction
    n_grid   : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    L : system matrix
    g : measured intensities

    Raised Exceptions:
    -

    Side Effects:
    -

    """

    # initialize system matrix with proper size
    L = np.zeros((n_rays * n_shots, (n_grid) ** 2))
    # initialize intensity vector
    g = np.zeros(n_rays * n_shots)

    # iterate over equispaced angles, take measurements, and update system matrix and sinogram
    theta = 0
    intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays, theta)

    # fill first indexes of matrix and vector for theta = 0
    for j in range(len(ray_indices)):
        L[[ray_indices[j]], [isect_indices[j]]] = lengths[j]
    for i in range(0, len(intensities)):
        g[i] = intensities[i]
    # change value of theta and make new measurements with the new theta
    theta += np.pi / n_shots
    intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays, theta)
    counter = 0

    while counter < n_shots - 1:

        for i in range(0, len(isect_indices)):
            L[[ray_indices[i] + n_rays * (counter + 1)], [isect_indices[i]]] = lengths[i]

        for j in range(0, n_rays):
            g[j + (counter + 1) * n_rays] = intensities[j]
        counter = counter + 1
        theta += np.pi / n_shots
        intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays, theta)

    return [L, g]


def compute_tomograph(n_shots: np.int, n_rays: np.int, n_grid: np.int) -> np.ndarray:
    """
    Compute tomographic image

    Arguments:
    n_shots  : number of different shot directions
    n_rays   : number of parallel rays per direction
    n_grid   : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    tim : tomographic image

    Raised Exceptions:
    -

    Side Effects:
    -

    """

    # Setup the system describing the image reconstruction
    [L, g] = setup_system_tomograph(n_shots, n_rays, n_grid)

    LT = np.transpose(L)
    A = np.matmul(LT, L)
    b = np.dot(np.transpose(L), g)

    # solve for tomographic image using your Cholesky solver
    x = np.linalg.solve(A, b)

    # convert solution of linear system to 2D image
    tim = x.reshape(n_grid, n_grid)

    return tim


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")


