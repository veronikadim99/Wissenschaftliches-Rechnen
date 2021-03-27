import numpy as np


def dft_matrix(n: int) -> np.ndarray:
    """
    Construct DFT matrix of size n.

    Arguments:
    n: size of DFT matrix

    Return:
    F: DFT matrix of size n

    Forbidden:
    - numpy.fft.*
    """
    # initialize matrix with proper size
    F = np.zeros((n, n), dtype='complex128')

    # create principal term for DFT matrix

    w = np.exp((2 * np.pi * (-1j)) / n)
    # fill matrix with values
    for i in range(0, n):
        F[[0], [i]] = 1
        F[[i], [0]] = 1
    for i in range(1, n):
        for j in range(1, n):
            F[[i], [j]] = w ** (i * j)
    # normalize dft matrix
    normalize = 1 / np.sqrt(n)
    F = normalize * F

    return F


def is_unitary(matrix: np.ndarray) -> bool:
    """
    Check if the passed in matrix of size (n times n) is unitary.

    Arguments:
    matrix: the matrix which is checked

    Return:
    unitary: True if the matrix is unitary
    """

    # check that F is unitary, if not return false

    Imatrix = np.eye(len(matrix), dtype=int)

    ConjMatrix = np.matrix.conjugate(matrix)
    TransposeMatrix = ConjMatrix.T
    product = np.dot(TransposeMatrix, matrix)
    unitary = np.allclose(product, Imatrix)

    return unitary


def create_harmonics(n: int = 128) -> (list, list):
    """
    Create delta impulse signals and perform the fourier transform on each signal.

    Arguments:
    n: the length of each signal

    Return:
    sigs: list of np.ndarrays that store the delta impulse signals
    fsigs: list of np.ndarrays with the fourier transforms of the signals
    """

    # list to store input signals to DFT
    sigs = []
    # Fourier-transformed signals
    fsigs = []
    # create signals and extract harmonics out of DFT matrix
    for i in range(0, n):
        e = np.zeros(n)
        e[i] = 1
        sigs.append(e)
        matrix = dft_matrix(n)
        product = np.dot(matrix, e)
        fsigs.append(product)

    return sigs, fsigs


def shuffle_bit_reversed_order(data: np.ndarray) -> np.ndarray:
    """
    Shuffle elements of data using bit reversal of list index.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    data: shuffled data array
    """
    dataCopy = np.copy(data)
    i = 0
    while i < len(data):
        binaryIndex = bin(i).replace('0b', '')
        while len(binaryIndex) < np.log2([len(data)]):
            binaryIndex = '0' + binaryIndex
        binaryIndex = ''.join(reversed(binaryIndex))

        dataCopy[int(binaryIndex, 2)] = data[i]
        i = i + 1

    data = dataCopy

    return data


def fft(data: np.ndarray) -> np.ndarray:
    """
    Perform real-valued discrete Fourier transform of data using fast Fourier transform.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    fdata: Fourier transformed data

    Note:
    This is not an optimized implementation but one to demonstrate the essential ideas
    of the fast Fourier transform.

    Forbidden:
    - numpy.fft.*
    """
    fdata = np.asarray(data, dtype='complex128')
    n = fdata.size
    if not n > 0 or (n & (n - 1)) != 0:
        raise ValueError

    # first step of FFT: shuffle data
    data = shuffle_bit_reversed_order(data)

    fdata = np.asarray(data, dtype='complex128')

    m = 0
    highTree = 0
    high = n
    while high > 1:
        high = high / 2
        highTree = highTree + 1

    while m < int(highTree):
        for k in range(0, 2 ** m):
            transformations = 0
            counter = 0
            while transformations < int(n / (2 ** (m + 1))):
                transformations = transformations + 1
                i = k + counter * (2 ** (m + 1))
                j = k + counter * (2 ** (m + 1)) + 2 ** m
                omega = (np.exp((-2 * np.pi * (1j) * k) / (2 ** (m + 1)))) * fdata[j]
                fdata[j] = fdata[i] - omega
                fdata[i] = fdata[i] + omega
                counter = counter + 1

        m = m + 1

    fdata = fdata / (np.sqrt(n))

    return fdata


def generate_tone(f: float = 261.626, num_samples: int = 44100) -> np.ndarray:
    """
    Generate tone of length 1s with frequency f (default mid C: f = 261.626 Hz) and return the signal.

    Arguments:
    f: frequency of the tone

    Return:
    data: the generated signal
    """

    # sampling range
    x_min = 0.0
    j = 0

    data = np.linspace(x_min, 2 * np.pi, num_samples)

    while j < num_samples:
        data[j] = np.sin(data[j] * f)

        j = j + 1

    return data


def low_pass_filter(adata: np.ndarray, bandlimit: int = 1000, sampling_rate: int = 44100) -> np.ndarray:
    """
    Filter high frequencies above bandlimit.

    Arguments:
    adata: data to be filtered
    bandlimit: bandlimit in Hz above which to cut off frequencies
    sampling_rate: sampling rate in samples/second

    Return:
    adata_filtered: filtered data
    """

    # translate band limit from Hz to data index according to sampling rate and data size
    bandlimit_index = int(bandlimit * adata.size / sampling_rate)

    # compute Fourier transform of input data
    newAdata = np.fft.fft(adata)
    #  set high frequencies above band limit to zero, make sure the almost symmetry of the transform is respected.
    newAdata[bandlimit_index + 1: len(newAdata) - bandlimit_index] = 0
    #  compute inverse transform and extract real component
    adata_filtered = np.fft.ifft(newAdata)
    adata_filtered = np.real(adata_filtered)

    return adata_filtered


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
