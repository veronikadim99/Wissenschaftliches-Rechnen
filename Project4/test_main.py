from unittest import TestCase
import numpy as np
import os
import unittest
from scipy import interpolate
from lib import plot_function, plot_function_interpolations, plot_spline, animate, linear_animation, cubic_animation, \
    runge_function, pad_coefficients
from main import natural_cubic_interpolation


class Test(TestCase):
    def test_natural_cubic_interpolation(self):
        # x-values to be interpolated

        keytimes = np.linspace(0, 200, 11)
        # y-values to be interpolated
        keyframes = [np.array([0., -0.05, -0.2, -0.2, 0.2, -0.2, 0.25, -0.3, 0.3, 0.1, 0.2]),
                     np.array([0., 0.0, 0.2, -0.1, -0.2, -0.1, 0.1, 0.1, 0.2, -0.3, 0.3])] * 5

        keyframes.append(keyframes[0])
        splines = []

        for i in range(11):  # Iterate over all animated parts
            x = keytimes
            y = np.array([keyframes[k][i] for k in range(11)])

            spline = natural_cubic_interpolation(x, y)
            if len(spline) == 0:
                animate(keytimes, keyframes, linear_animation(keytimes, keyframes))
                self.fail("Natural cubic interpolation not implemented.")
            splines.append(spline)
