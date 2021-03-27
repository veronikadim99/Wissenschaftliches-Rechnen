import numpy as np


def find_root_bisection(f: object, lival: np.floating, rival: np.floating, ival_size: np.floating = -1.0,
                        n_iters_max: int = 256) -> np.floating:
    """
    Find a root of function f(x) in (lival, rival) with bisection method.

    Arguments:
    f: function object (assumed to be continuous), returns function value if called as f(x)
    lival: initial left boundary of interval containing root
    rival: initial right boundary of interval containing root
    ival_size: minimal size of interval / convergence criterion (optional)
    n_iters_max: maximum number of iterations (optional)

    Return:
    root: approximate root of the function
    """

    assert (n_iters_max > 0)
    assert (rival > lival)

    # set meaningful minimal interval size if not given as parameter, e.g. 10 * eps
    if ival_size is None:
        ival_size = np.finfo(np.float64.eps * 10)

    fl = f(lival)
    fr = f(rival)

    # make sure the given interval contains a root
    assert (not ((fl > 0.0 and fr > 0.0) or (fl < 0.0 and fr < 0.0)))

    n_iterations = 0
    while n_iterations < n_iters_max:
        fl = f(lival)
        x = (lival + rival) / 2
        n_iterations = n_iterations + 1
        if f(x) == 0:
            root = x
        if f(x) < ival_size:
            root = x
        if np.sign(f(x)) == np.sign(fl):
            lival = x
        else:
            rival = x
    root = x
    return root


def find_root_newton(f: object, df: object, start: np.inexact, n_iters_max: int = 256) -> (np.inexact, int):
    """
    Find a root of f(x)/f(z) starting from start using Newton's method.

    Arguments:
    f: function object (assumed to be continuous), returns function value if called as f(x)
    df: derivative of function f, also callable
    start: start position, can be either float (for real valued functions) or complex (for complex valued functions)
    n_iters_max: maximum number of iterations (optional)

    Return:
    root: approximate root, should have the same format as the start value start
    n_iterations: number of iterations
    """

    assert (n_iters_max > 0)

    root = start

    # chose meaningful convergence criterion eps, e.g 10 * eps

    n_iterations = 0
    eps = np.finfo(np.float64).eps * 10

    while n_iterations < n_iters_max:
        xn = start - (f(start) / df(start))
        root = xn

        start = xn
        if abs(f(xn)) < eps:
            return root, n_iterations
        n_iterations = n_iterations + 1

    return root, n_iterations


def surface_area(v: np.ndarray, f: np.ndarray) -> float:
    """
    Calculate the area of the given surface represented as triangles in f.

    Arguments:
    v: vertices of the triangles
    f: vertex indices of all triangles. f[i] gives 3 vertex indices for the three corners of the triangle i

    Return:
    area: the total surface area
    """

    area = 0.0
    # iterate over all triangles and sum up their area

    for i in range(0, len(f)):
        a = v[f[i, 0]]
        b = v[f[i, 1]]
        c = v[f[i, 2]]
        prod1 = b - a
        prod2 = c - a
        vector = np.cross(prod1, prod2)

        area = area + np.linalg.norm(vector) / 2

    return area


def surface_area_gradient(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Calculate the area gradient of the given surface represented as triangles in f.

    Arguments:
    v: vertices of the triangles
    f: vertex indices of all triangles. f[i] gives 3 vertex indices for the three corners of the triangle i

    Return:
    gradient: the surface area gradient of all vertices in v
    """

    gradient = np.zeros(v.shape)
    i = 0
    # iterate over all triangles and sum up the vertices gradients
    while i < len(f):
        vertexa = v[f[i, 1]] - v[f[i, 0]]
        vertexc = v[f[i, 2]] - v[f[i, 1]]
        vertexb = v[f[i, 2]] - v[f[i, 0]]
        grada = np.cross(np.cross(vertexa, vertexb), vertexc)
        gradb = np.cross(np.cross(-vertexa, vertexc), vertexb)
        gradc = np.cross(np.cross(-vertexb, -vertexc), vertexa)
        norma = np.linalg.norm(vertexc) / np.linalg.norm(-grada)
        normb = np.linalg.norm(vertexb) / np.linalg.norm(-gradb)
        normc = np.linalg.norm(vertexa) / np.linalg.norm(-gradc)
        gradient[f[i, 0]] = gradient[f[i, 0]] + (-grada * norma)
        gradient[f[i, 1]] = gradient[f[i, 1]] + (-gradb * normb)
        gradient[f[i, 2]] = gradient[f[i, 2]] + (-gradc * normc)
        i = i + 1

    return gradient


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
