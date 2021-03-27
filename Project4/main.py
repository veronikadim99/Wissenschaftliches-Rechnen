import numpy as np


def lagrange_interpolation(x: np.ndarray, y: np.ndarray) -> (np.poly1d, list):
    """
    Generate Lagrange interpolation polynomial.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    polynomial: polynomial as np.poly1d object
    base_functions: list of base polynomials
    """

    assert (x.size == y.size)
    polynomial = np.poly1d(0)
    base_functions = []

    for i in range(len(x)):
        product = 1
        for k in range(len(x)):
            if k != i:
                product = product * np.poly1d([1, -x[k]]) / (x[i] - x[k])
        base_functions.append(product)
        polynomial = polynomial + (product * y[i])
        print(polynomial)
        print('base', base_functions)

    return polynomial, base_functions


def hermite_cubic_interpolation(x: np.ndarray, y: np.ndarray, yp: np.ndarray) -> list:
    """
    Compute hermite cubic interpolation spline

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points
    yp: derivative values of interpolation points

    Returns:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """
    A = np.zeros((4, 4))
    A[[0], [0]] = 1
    A[[1], [0]] = 1
    A[[2], [1]] = 1
    A[[3], [1]] = 1
    b = np.zeros(4)
    spline = []
    assert (x.size == y.size == yp.size)
    # compute piecewise interpolating cubic polynomials
    for i in range(len(x) - 1):
        A[[0], [1]] = x[i]
        A[[0], [2]] = (x[i]) ** 2
        A[[0], [3]] = (x[i]) ** 3
        A[[1], [1]] = x[i + 1]
        A[[1], [2]] = (x[i + 1]) ** 2
        A[[1], [3]] = (x[i + 1]) ** 3
        A[[2], [2]] = 2 * (x[i])
        A[[2], [3]] = 3 * ((x[i]) ** 2)
        A[[3], [2]] = 2 * (x[i + 1])
        A[[3], [3]] = 3 * ((x[i + 1]) ** 2)
        B = np.linalg.inv(A)
        b[0] = y[i]
        b[1] = y[i + 1]
        b[2] = yp[i]
        b[3] = yp[i + 1]

        spline.append(np.poly1d(np.flip(np.matmul(B, b))))

    return spline


def natural_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Intepolate the given function using a spline with natural boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """
    if len(x) <= 2:
        spline = []
        A = np.zeros((4, 4))
        A[[0], [2]] = 2
        A[[0], [3]] = 6 * (x[0])
        A[[1], [0]] = 1
        A[[1], [1]] = x[0]
        A[[1], [2]] = x[0] ** 2
        A[[1], [3]] = x[0] ** 3
        A[[2], [0]] = 1
        A[[2], [1]] = x[1]
        A[[2], [2]] = x[1] ** 2
        A[[2], [3]] = x[1] ** 3
        A[[3], [2]] = 2
        A[[3], [3]] = 6 * (x[1])
        b = np.zeros(4)
        b[0] = 0
        b[1] = y[0]
        b[2] = y[1]
        b[3] = 0

        spline.append(np.poly1d(np.flip(np.linalg.inv(A).dot(b))))

        return spline

    spline = []
    assert (x.size == y.size)
    A = np.zeros(((4 * len(x)) - 4, (4 * len(x)) - 4))

    b = np.zeros(4 * len(x) - 4)

    A[[0], [2]] = 2
    A[[0], [3]] = 6 * (x[0])
    counterC = 1
    j = 3
    i = 4
    c = 0
    A[[1], [0]] = 1
    A[[1], [1]] = x[0]
    A[[1], [2]] = x[0] ** 2
    A[[1], [3]] = x[0] ** 3
    A[[2], [0]] = 1
    A[[2], [1]] = x[1]
    A[[2], [2]] = x[1] ** 2
    A[[2], [3]] = x[1] ** 3
    A[[3], [1]] = 1
    A[[3], [2]] = x[1] * 2
    A[[3], [3]] = (x[1] ** 2) * 3
    A[[4], [2]] = 2
    A[[4], [3]] = x[1] * 6

    #construct linear system with natural boundary conditions
    while counterC + 1 < len(x) - 1:
        C = np.zeros((6, 4))

        C[[0], [1]] = -1
        C[[0], [2]] = -2 * x[counterC]
        C[[0], [3]] = -3 * (x[counterC]) ** 2
        C[[1], [2]] = -2
        C[[1], [3]] = -6 * x[counterC]
        C[[2], [0]] = 1
        C[[2], [1]] = x[counterC]
        C[[2], [2]] = x[counterC] ** 2
        C[[2], [3]] = x[counterC] ** 3
        C[[3], [0]] = 1
        C[[3], [1]] = x[counterC + 1]
        C[[3], [2]] = x[counterC + 1] ** 2
        C[[3], [3]] = x[counterC + 1] ** 3

        C[[4], [1]] = 1
        C[[4], [2]] = 2 * x[counterC + 1]
        C[[4], [3]] = 3 * (x[counterC + 1]) ** 2
        C[[5], [2]] = 2
        C[[5], [3]] = 6 * x[counterC + 1]

        c = c + 1

        A[j: j + 6, i: i + 4] = C
        i = i + 4
        j = j + 4
        counterC = counterC + 1

    A[[4 * len(x) - 5], [4 * len(x) - 6]] = 2
    A[[4 * len(x) - 5], [4 * len(x) - 5]] = 6 * (x[len(x) - 1])
    C = np.zeros((4, 4))
    C[[0], [1]] = -1
    C[[0], [2]] = -2 * x[counterC]
    C[[0], [3]] = -3 * (x[counterC]) ** 2
    C[[1], [2]] = -2
    C[[1], [3]] = -6 * x[counterC]
    C[[2], [0]] = 1
    C[[2], [1]] = x[counterC]
    C[[2], [2]] = x[counterC] ** 2
    C[[2], [3]] = x[counterC] ** 3
    C[[3], [0]] = 1
    C[[3], [1]] = x[counterC + 1]
    C[[3], [2]] = x[counterC + 1] ** 2
    C[[3], [3]] = x[counterC + 1] ** 3
    A[j: j + 4, i: i + 4] = C

    count = 0
    k = 1
    while count + 1 < len(y):
        b[k] = y[count]
        b[k + 1] = y[count + 1]
        count = count + 1
        k = k + 4

    A = np.linalg.inv(A)
    c = np.matmul(A, b)
    print('c', c)
    counter = 0
    while counter < len(c):
        spline.append(np.poly1d(np.flip(c[counter: counter + 4])))
        counter = counter + 4

    return spline


def periodic_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Interpolate the given function with a cubic spline and periodic boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    if len(x) <= 2:
        A = np.zeros((4, 4))

        spline = []
        A = np.zeros((4, 4))
        A[[2], [1]] = 1
        A[[2], [2]] = 2 * (x[1])
        A[[2], [3]] = 3 * (x[1] ** 2)

        A[[3], [2]] = 2
        A[[3], [3]] = 6 * x[1]
        A[[0], [0]] = 1
        A[[0], [1]] = x[0]
        A[[0], [2]] = x[0] ** 2
        A[[0], [3]] = x[0] ** 3
        A[[1], [0]] = 1
        A[[1], [1]] = x[1]
        A[[1], [2]] = x[1] ** 2
        A[[1], [3]] = x[1] ** 3
        print(A)

        b = np.zeros(4)
        b[0] = y[0]
        b[1] = y[1]
        print(b)
        print(np.linalg.inv(A).dot(b))

        spline.append(np.poly1d(np.flip(np.linalg.inv(A).dot(b))))

        return spline

    spline = []
    assert (x.size == y.size)
    A = np.zeros(((4 * len(x)) - 4, (4 * len(x)) - 4))
    xA = (4 * len(x)) - 4

    b = np.zeros(4 * len(x) - 4)
    A[[0], [1]] = 1
    A[[0], [2]] = 2 * (x[0])
    A[[0], [3]] = 3 * (x[0] ** 2)
    A[[0], [xA - 3]] = -1
    A[[0], [xA - 2]] = -2 * (x[len(x) - 1])
    A[[0], [xA - 1]] = -3 * (x[len(x) - 1] ** 2)
    A[[1], [2]] = 2
    A[[1], [3]] = 6 * x[0]
    A[[1], [xA - 1]] = x[len(x) - 1] * (-6)
    A[[1], [xA - 2]] = -2
    counterC = 1
    j = 4
    i = 4
    c = 0

    A[[2], [0]] = 1
    A[[2], [1]] = x[0]
    A[[2], [2]] = x[0] ** 2
    A[[2], [3]] = x[0] ** 3
    A[[3], [0]] = 1
    A[[3], [1]] = x[1]
    A[[3], [2]] = x[1] ** 2
    A[[3], [3]] = x[1] ** 3
    A[[4], [1]] = 1
    A[[4], [2]] = x[1] * 2
    A[[4], [3]] = (x[1] ** 2) * 3
    A[[5], [2]] = 2
    A[[5], [3]] = x[1] * 6

    #construct linear system with natural boundary conditions
    while counterC + 1 < len(x) - 1:
        C = np.zeros((6, 4))

        C[[0], [1]] = -1
        C[[0], [2]] = -2 * x[counterC]
        C[[0], [3]] = -3 * (x[counterC]) ** 2
        C[[1], [2]] = -2
        C[[1], [3]] = -6 * x[counterC]
        C[[2], [0]] = 1
        C[[2], [1]] = x[counterC]
        C[[2], [2]] = x[counterC] ** 2
        C[[2], [3]] = x[counterC] ** 3
        C[[3], [0]] = 1
        C[[3], [1]] = x[counterC + 1]
        C[[3], [2]] = x[counterC + 1] ** 2
        C[[3], [3]] = x[counterC + 1] ** 3

        C[[4], [1]] = 1
        C[[4], [2]] = 2 * x[counterC + 1]
        C[[4], [3]] = 3 * (x[counterC + 1]) ** 2
        C[[5], [2]] = 2
        C[[5], [3]] = 6 * x[counterC + 1]

        c = c + 1

        A[j: j + 6, i: i + 4] = C
        i = i + 4
        j = j + 4
        counterC = counterC + 1

    A[[4 * len(x) - 5], [4 * len(x) - 6]] = 2
    A[[4 * len(x) - 5], [4 * len(x) - 5]] = 6 * (x[len(x) - 1])
    C = np.zeros((4, 4))
    C[[0], [1]] = -1
    C[[0], [2]] = -2 * x[counterC]
    C[[0], [3]] = -3 * (x[counterC]) ** 2
    C[[1], [2]] = -2
    C[[1], [3]] = -6 * x[counterC]
    C[[2], [0]] = 1
    C[[2], [1]] = x[counterC]
    C[[2], [2]] = x[counterC] ** 2
    C[[2], [3]] = x[counterC] ** 3
    C[[3], [0]] = 1
    C[[3], [1]] = x[counterC + 1]
    C[[3], [2]] = x[counterC + 1] ** 2
    C[[3], [3]] = x[counterC + 1] ** 3
    A[j: j + 4, i: i + 4] = C

    count = 0
    k = 2

    while count + 1 < len(y):
        b[k] = y[count]
        b[k + 1] = y[count + 1]
        count = count + 1
        k = k + 4

    A = np.linalg.inv(A)
    c = np.matmul(A, b)

    counter = 0
    while counter < len(c):
        spline.append(np.poly1d(np.flip(c[counter: counter + 4])))
        counter = counter + 4

    return spline


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
