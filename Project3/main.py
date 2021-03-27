import numpy as np
import lib
import matplotlib as mpl


def power_iteration(M: np.ndarray, epsilon: float = -1.0) -> (np.ndarray, list):
    """
    Compute largest eigenvector of matrix M using power iteration. It is assumed that the
    largest eigenvalue of M, in magnitude, is well separated.

    Arguments:
    M: matrix, assumed to have a well separated largest eigenvalue
    epsilon: epsilon used for convergence (default: 10 * machine precision)

    Return:
    vector: eigenvector associated with largest eigenvalue
    residuals : residual for each iteration step

    Raised Exceptions:
    ValueError: if matrix is not square

    Forbidden:
    numpy.linalg.eig, numpy.linalg.eigh, numpy.linalg.svd
    """

    A = ([[0, 2],
          [1, -1],
          [-1, -1]])

    A = np.asarray(A)
    A.shape = (3, 2)

    n, n = M.shape
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix not square")

    # set epsilon to default value if not set by user
    epsilon = np.dot(10, np.finfo(float).eps)

    # random vector of proper size to initialize iteration

    vector = np.zeros(n)

    # v = np.zeros(n)

    vector[0] = 1
    # Initialize residual list and residual of current eigenvector estimate
    residuals = []

    residual = 2.0 * epsilon

    counter = 0

    # Perform power iteration
    while residual > epsilon:
        if counter == 0:
            firstmulT = 1 / (np.matmul(vector.transpose(), vector))
            secundmulT = np.matmul(np.matmul(M.transpose(), vector.transpose()), vector)
            lamdaMax = firstmulT * secundmulT
            counter = 1

        vector = np.matmul(M, vector)
        vector = vector / (np.linalg.norm(vector))
        firstmulT = 1 / (np.matmul(vector.transpose(), vector))
        secundmulT = np.matmul(np.matmul(M.transpose(), vector.transpose()), vector)
        lMax = firstmulT * secundmulT

        residual = abs(lamdaMax - lMax)
        residual = residual / (lMax)
        residuals.append(residual)
        lamdaMax = lMax

        pass

    return vector, residuals


def load_images(path: str, file_ending: str = ".png") -> (list, int, int):
    """
    Load all images in path with matplotlib that have given file_ending

    Arguments:
    path: path of directory containing image files that can be assumed to have all the same dimensions
    file_ending: string that image files have to end with, if not->ignore file

    Return:
    images: list of images (each image as numpy.ndarray and dtype=float64)
    dimension_x: size of images in x direction
    dimension_y: size of images in y direction
    """

    images = []
    x = 0
    y = 0
    count = 0
    # read each image in path as numpy.ndarray and append to images

    listFornames = lib.list_directory(path)
    listFornames.sort()
    for oneim in listFornames:

        if oneim.endswith(file_ending):
            im = mpl.image.imread(path + oneim)
            images.append(np.asarray(im).astype('float64'))

            if count == 0:
                y = len(np.asarray(im))
                x = len(np.asarray(im)[0])
                count = 1
        else:
            continue

    # set dimensions according to first image in images
    dimension_y = y
    dimension_x = x

    return images, dimension_x, dimension_y


def setup_data_matrix(images: list) -> np.ndarray:
    """
    Create data matrix out of list of 2D data sets.

    Arguments:
    images: list of 2D images (assumed to be all homogeneous of the same size and type np.ndarray)

    Return:
    D: data matrix that contains the flattened images as rows
    """
    # initialize data matrix with proper size and data type
    firstim = images[0]
    D = np.zeros((len(images), len(np.asarray(firstim)[0]) * len(np.asarray(firstim))))

    # add flattened images to data matrix
    counter = 0
    for im in images:
        D[counter] = np.reshape(im, 11368)
        counter = counter + 1

    return D


def calculate_pca(D: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Perform principal component analysis for given data matrix.

    Arguments:
    D: data matrix of size m x n where m is the number of observations and n the number of variables

    Return:
    pcs: matrix containing principal components as rows
    svals: singular values associated with principle components
    mean_data: mean that was subtracted from data
    """

    # subtract mean from data / center data at origin

    mean_data = np.zeros(len(D[0]))

    for i in range(0, len(D[0])):
        meanValue = D[:, i].mean()
        mean_data[i] = meanValue

    # compute left and right singular vectors and singular values
    u, svals, pcs = np.linalg.svd(D - mean_data, full_matrices=False)

    return pcs, svals, mean_data


def accumulated_energy(singular_values: np.ndarray, threshold: float = 0.8) -> int:
    """
    Compute index k so that threshold percent of magnitude of singular values is contained in
    first k singular vectors.

    Arguments:
    singular_values: vector containing singular values
    threshold: threshold for determining k (default = 0.8)

    Return:
    k: threshold index
    """

    # normalize singular value magnitudes

    sumValues = np.sum(singular_values)
    normalized = np.dot(singular_values, (1 / sumValues))

    sum = 0
    k = 0
    for i in range(0, len(normalized)):
        sum = sum + normalized[i]
        if sum >= threshold:
            k = i + 1
            break

        else:
            continue

    return k


def project_faces(pcs: np.ndarray, images: list, mean_data: np.ndarray) -> np.ndarray:
    """
    Project given image set into basis.

    Arguments:
    pcs: matrix containing principal components / eigenfunctions as rows
    images: original input images from which pcs were created
    mean_data: mean data that was subtracted before computation of SVD/PCA

    Return:
    coefficients: basis function coefficients for input images, each row contains coefficients of one image
    """

    # initialize coefficients array with proper size
    coefficients = np.zeros((len(images), pcs.shape[0]))
    T = np.transpose(pcs)
    PT = np.matmul(pcs, T)
    images = setup_data_matrix(images)
    listToSort = []

    # iterate over images and project each normalized image into principal component basis
    for i in range(0, len(images)):
        images[i] = images[i] - mean_data
        x = np.matmul(pcs, images[i])
        coefficients[i] = np.linalg.solve(PT, x)
        coeff = coefficients[i]
        k = accumulated_energy(coeff)
        listToSort.append(k)
    listToSort.sort()

    return coefficients


def identify_faces(coeffs_train: np.ndarray, pcs: np.ndarray, mean_data: np.ndarray, path_test: str) -> (
        np.ndarray, list, np.ndarray):
    """
    Perform face recognition for test images assumed to contain faces.

    For each image coefficients in the test data set the closest match in the training data set is calculated.
    The distance between images is given by the angle between their coefficient vectors.

    Arguments:
    coeffs_train: coefficients for training images, each image is represented in a row
    path_test: path to test image data

    Return:
    scores: Matrix with correlation between all train and test images, train images in rows, test images in columns
    imgs_test: list of test images
    coeffs_test: Eigenface coefficient of test images
    """

    #load test data set

    imgs_test, x, y = load_images(path_test, ".png")

    # project test data set into eigenbasis

    coeffs_test = project_faces(pcs, list(np.asarray(imgs_test)), mean_data)

    # initialize scores matrix with proper size
    scores = np.zeros((coeffs_train.shape[0], coeffs_test.shape[0]))
    j = 0
    while j < coeffs_train.shape[0]:
        for i in range(0, coeffs_test.shape[0]):
            testNorm = np.linalg.norm(coeffs_test[i])
            trainNorm = np.linalg.norm(coeffs_train[j])
            multNorm = np.dot(testNorm, trainNorm)
            scores[[j], [i]] = (np.matmul(coeffs_train[j], coeffs_test[i])) / multNorm
            scores[[j], [i]] = np.arccos(scores[[j], [i]])
        j = j + 1

    return scores, imgs_test, coeffs_test


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
