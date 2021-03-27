import numpy as np
from lib import timedcall, plot_2d

diag = 0
offDiag = 0
theta_rad = 0


def matrix_multiplication(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate product of two matrices a * b without the use of numpy.dot and numpy.matrix

    Arguments:
    a : first matrix
    b : second matrix

    Return:
    c : matrix product a * b

    Raised Exceptions:
    ValueError: if matrix sizes are incompatible

    Side Effects:
    -
    """
    n, m_a = a.shape
    m_b, p = b.shape

    # test if shape of matrices is compatible and raise error if not
    if m_a != m_b:
        raise ValueError('They should be equal')

    # initialize result matrix with zeros
    c = np.zeros((n, p))

    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(m_a):
                c[[i], [j]] = (c[[i], [j]]) + ((a[[i], [k]]) * (b[[k], [j]]))

    return c


def compare_multiplication(nmax: int, n: int) -> dict:
    """
    Compare performance of numpy matrix multiplication (np.dot()) and matrix_multiplication.

    Arguments:
    nmax : maximum matrix size to be tested
    n : step size for matrix sizes

    Return:
    tr_dict : numpy and matrix_multiplication timings and results {"timing_numpy": [numpy_timings],
    "timing_mat_mult": [mat_mult_timings], "results_numpy": [numpy_results], "results_mat_mult": [mat_mult_results]}

    Raised Exceptions:
    -

    Side effects:
    Generates performance plots.
    """

    x, y_mat_mult, y_numpy, r_mat_mult, r_numpy = [], [], [], [], []
    tr_dict = dict(timing_numpy=y_numpy, timing_mat_mult=y_mat_mult, results_numpy=r_numpy, results_mat_mult=r_mat_mult)

    for m in range(2, nmax, n):
        a = np.random.randn(m, m)
        b = np.random.randn(m, m)

        # Execute functions and measure the execution time
        time_mat_mult, result_mat_mult = timedcall(matrix_multiplication, a, b)
        time_numpy, result_numpy = timedcall(np.dot, a, b)

        # Add calculated values to lists
        x.append(m)
        y_numpy.append(time_numpy)
        y_mat_mult.append(time_mat_mult)
        r_numpy.append(result_numpy)
        r_mat_mult.append(result_mat_mult)

    # Plot the computed data
    plot_2d(x_data=x, y_data=[y_mat_mult, y_numpy], labels=["matrix_mult", "numpy"],
            title="NumPy vs. for-loop matrix multiplication",
            x_axis="Matrix size", y_axis="Time", x_range=[2, nmax])

    return tr_dict


def machine_epsilon(fp_format: np.dtype) -> np.number:
    """
    Calculate the machine precision for the given floating point type without the use of numpy.finfo

    Arguments:
    fp_format: floating point format, e.g. float32 or float64

    Return:
    eps : calculated machine precision

    Raised Exceptions:
    -

    Side Effects:
    Prints out iteration values.

    """

    # Create necessary variables for iteration
    var_one = 1.0
    var_two = 2.0
    i = 0
    eps = var_one
    # determine machine precision
    while (eps + var_one) > var_one:
        eps = eps / var_two
    eps = eps * var_two

    print('  i  |       2^(-i)        |  1 + 2^(-i)  ')
    print('  ----------------------------------------')

    print('{0:4.0f} |  {1:16.8e}   | equal 1'.format(i, eps))
    return eps


def rotation_matrix(theta: float) -> np.ndarray:
    """
    Create 2x2 rotation matrix around angle theta.

    Arguments:
    theta : rotation angle (in degrees)

    Return:
    r : rotation matrix

    Raised Exceptions:
    -

    Side Effects:
    -
    """

    # create empty matrix
    r = np.zeros((2, 2))

    # convert angle to radians
    global theta_rad
    theta_rad = theta * (np.pi / 180)

    r[0, 0] = np.cos(theta_rad)
    r[0, 1] = -(np.sin(theta_rad))
    r[1, 0] = np.sin(theta_rad)
    r[1, 1] = np.cos(theta_rad)

    # calculate diagonal terms of matrix
    global diag
    diag = (np.cos(theta_rad)) ** 2

    # off-diagonal terms of matrix
    global offDiag
    offDiag = -(np.sin(theta_rad)) ** 2

    return r


def inverse_rotation(theta: float) -> np.ndarray:
    """
    Compute inverse of the 2d rotation matrix that rotates a
    given vector by theta without use of numpy.linalg.inv and numpy.linalg.solve.

    Arguments:
    theta: rotation angle

    Return:
    Inverse of the rotation matrix


    """

    rotation_matrix(theta)
    m = np.zeros((2, 2))

    m[0, 0] = (np.cos(theta_rad)) / (diag - offDiag)
    m[0, 1] = (np.sin(theta_rad)) / (diag - offDiag)
    m[1, 0] = -(np.sin(theta_rad)) / (diag - offDiag)
    m[1, 1] = (np.cos(theta_rad)) / (diag - offDiag)

    return m
    
if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
