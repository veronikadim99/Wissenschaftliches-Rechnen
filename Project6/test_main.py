import os
import numpy as np
import unittest
import matplotlib.pyplot as plt
import matplotlib.image as image
from matplotlib import animation

from lib import fpoly, dfpoly, fractal_functions, generate_sampling, get_colors, generate_cylinder, load_object, \
    prepare_visualization, update_visualization, calculate_abs_gradient
from main import find_root_bisection, find_root_newton, surface_area, surface_area_gradient


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.isfile("data/data.npz"):
            cls.data = np.load("data/data.npz")
        else:
            raise IOError("Could not load data file 'data.npz' for tests.")

    @classmethod
    def tearDownClass(cls):
        cls.data.close()

    def test_1_find_root_bisection(self):

        x0 = find_root_bisection(lambda x: x ** 2 - 2, np.float64(-1.0), np.float64(2.0))
        self.assertTrue(np.isclose(x0, np.sqrt(2)))
        x1 = find_root_bisection(fpoly, np.float64(-1.0), np.float64(5.0))
        x2 = find_root_bisection(fpoly, np.float64(1.0), np.float64(4.0))
        x3 = find_root_bisection(fpoly, np.float64(4.0), np.float64(5.0))
        x = np.linspace(-1.0, 5.5, 1000)
        plt.plot(x, fpoly(x))
        plt.plot([x1, x2, x3], [0.0] * 3, 'ro')
        plt.grid(True)
        plt.show()

    def test_2_find_root_newton(self):
        x0, i0 = find_root_newton(lambda x: x ** 2 - 2, lambda x: 2 * x, np.float64(10.0))

        x1, i1 = find_root_newton(lambda x: x ** 2 - 2, lambda x: 2 * x, np.float64(5.0))

        x2, i2 = find_root_newton(lambda x: x ** 2 - 2, lambda x: 2 * x, np.float64(0.1))

        self.assertTrue(np.allclose(np.array([x0, x1, x2]), np.array([np.sqrt(2)] * 3)))

        x0, i0 = find_root_newton(fpoly, dfpoly, np.float64(-1.0))
        x1, i1 = find_root_newton(fpoly, dfpoly, np.float64(2.0))
        x2, i2 = find_root_newton(fpoly, dfpoly, np.float64(5.0))
        self.assertTrue(np.allclose(np.array([x0, x1, x2]), np.array([0.335125152578, 2.61080833945, 4.79087461944])))
        x = np.linspace(-1.0, 5.5, 1000)
        plt.plot(x, fpoly(x))
        plt.plot([x0, x1, x2], [0.0] * 3, 'ro')
        plt.grid(True)
        plt.show()
        x0, i0 = find_root_newton(lambda x: x ** 3 - 2 * x + 2, lambda x: 3 * x ** 2 - 2, np.float64(-0.4), 50)

    def test_3_surface_area(self):

        v, f, c = generate_cylinder(16, 8)
        a = surface_area(v, f)
        self.assertTrue(np.isclose(a, 4.99431224361))

    def test_4_surface_area_gradient(self):
        v, f, c = load_object("data/wave")
        gradient = surface_area_gradient(v, f)
        gradient = gradient.flatten()
        gradient = gradient / np.linalg.norm(gradient)
        reference = self.data["gradient"].flatten()
        reference = reference / np.linalg.norm(reference)
        print('ref', reference)
        self.assertTrue(np.allclose(reference, gradient))


if __name__ == '__main__':
    unittest.main()
