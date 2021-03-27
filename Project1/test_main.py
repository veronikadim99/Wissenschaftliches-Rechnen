import random
from unittest import TestCase

import numpy as np

from main import compare_multiplication, matrix_multiplication, machine_epsilon, rotation_matrix, inverse_rotation


class Test(TestCase):
    def test_compare_multiplication(self):
        r_dict = compare_multiplication(200, 40)
        for r in zip(r_dict["results_numpy"], r_dict["results_mat_mult"]):
            self.assertTrue(np.allclose(r[0], r[1]))


class Test(TestCase):
    def test_matrix_multiplication(self):
        a = np.random.randn(11, 11)
        c = np.random.randn(20, 20)
        self.assertTrue(np.allclose(np.dot(a, a), matrix_multiplication(a, a)))
        self.assertRaises(ValueError, matrix_multiplication, a, c)


class Test(TestCase):
    def test_machine_epsilon(self):
        a = random.random()
        eps_function = machine_epsilon(a)
        eps_finfo = np.finfo(a).eps
        assert eps_finfo == eps_function


class Test(TestCase):
    def test_inverse_rotation(self):
        theta = random.randint(0,360)
        rotation_matrix1 = np.array(((np.cos(np.radians(theta)), -np.sin(np.radians(theta))),
                                    (np.sin(np.radians(theta)), np.cos(np.radians(theta)))))
        rotation_matrix2 = rotation_matrix(theta)
        self.assertTrue(np.allclose(rotation_matrix1,rotation_matrix2))
        inverse1 = np.linalg.inv(rotation_matrix2)
        inverse2 = inverse_rotation(theta)
        self.assertTrue(np.allclose(inverse1,inverse2))
