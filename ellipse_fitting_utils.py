import os, sys, copy, itertools, matplotlib
import numpy as np
import matplotlib.pyplot as plt

# NOTE: astropy
from astropy import (
    units,
    constants
)
from astropy.io import fits
from astropy import stats

# NOTE: scipy
from scipy import interpolate

# NOTE:
from lmfit import (
    minimize,
    Parameters,
    fit_report,
)

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patheffects
# ---------------------------------------------------------------------------- #

# # NOTE:
# path = os.environ["GitHub"] + "/utils"
# sys.path.append(path)

import matplotlib_utils as matplotlib_utils
import pickle_utils as pickle_utils
# ---------------------------------------------------------------------------- #

def continuous_angle_conversion(angle):
    return (np.arctan2(np.sin(angle), np.cos(angle)) * 180 / np.pi) % 360


# ========== #
# NOTE: ellipse fitting with ...
# ========== #
# class Interpolator:
#
#     def __init__(self, image):
#
#         self.image = image
#
#         x = np.arange(image.shape[1])
#         y = np.arange(image.shape[0])
#
#         self.image_interp = interpolate.RegularGridInterpolator(
#             (x, y), image
#         )

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

def func(a, parameters, angles):

    # NOTE:
    x0 = parameters["x0"]
    y0 = parameters["y0"]
    #print("centre =", (x0, y0))

    # NOTE:
    e = parameters["ellipticity"]
    angle = parameters["angle"]

    # NOTE:
    b = a * np.sqrt(1.0 - e**2.0)

    # NOTE:
    r = np.divide(
        a * b,
        np.sqrt(
            np.add(
                a**2.0 * np.sin(angles - angle)**2.0,
                b**2.0 * np.cos(angles - angle)**2.0
            )
        )
    )
    x = r * np.cos(angles) + x0
    y = r * np.sin(angles) + y0

    # NOTE: m = 1
    if np.logical_and(
        parameters["a_1"] is not None,
        parameters["b_1"] is not None,
    ):
        a_1 = parameters["a_1"]
        b_1 = parameters["b_1"]
        r_1 = np.add(
            a_1 * np.cos(1.0 * (angles - angle)),
            b_1 * np.sin(1.0 * (angles - angle)),
        )
        x_1 = r_1 * np.cos(angles)
        y_1 = r_1 * np.sin(angles)

        x += x_1
        y += y_1

    # NOTE: m = 3
    if np.logical_and(
        parameters["a_3"] is not None,
        parameters["b_3"] is not None,
    ):
        a_3 = parameters["a_3"]
        b_3 = parameters["b_3"]
        r_3 = np.add(
            a_3 * np.cos(3.0 * (angles - angle)),
            b_3 * np.sin(3.0 * (angles - angle)),
        )
        x_3 = r_3 * np.cos(angles)
        y_3 = r_3 * np.sin(angles)

        x += x_3
        y += y_3

    # NOTE: m = 4
    if np.logical_and(
        parameters["a_4"] is not None,
        parameters["b_4"] is not None,
    ):
        a_4 = parameters["a_4"]
        b_4 = parameters["b_4"]
        r_4 = np.add(
            a_4 * np.cos(4.0 * (angles - angle)),
            b_4 * np.sin(4.0 * (angles - angle)),
        )
        x_4 = r_4 * np.cos(angles)
        y_4 = r_4 * np.sin(angles)

        x += x_4
        y += y_4

    # NOTE:
    idx = np.logical_or(np.isnan(x), np.isnan(y))
    if np.sum(idx) > 0.0:
        raise NotImplementedError()

    return x, y


class Sample:

    def __init__(
        self,
        image: np.ndarray,
        error: np.ndarray = None,
        mask: np.ndarray = None,
    ):

        # NOTE:
        self.image = image

        # NOTE:
        self.shape = image.shape#;print("shape =", self.shape)

        # NOTE:
        x = np.arange(image.shape[1])
        y = np.arange(image.shape[0])
        self.image_interp = interpolate.RegularGridInterpolator(
            (x, y),
            image,
            bounds_error=False,
            fill_value=0.0
        )

        # NOTE:
        if error is None:
            self.error_interp = None
        else:
            self.error_interp = interpolate.RegularGridInterpolator(
                (x, y),
                error,
                bounds_error=False,
                fill_value=0.0
            )

        # NOTE:
        if mask.shape == self.shape:
            self.mask = mask
        else:
            raise NotImplementedError()

        # NOTE:
        #self.angles = np.linspace(0.0, 2.0 * np.pi, 100)

    def extract(
        self,
        a: np.float,
        parameters,
        condition:bool = True # NOTE: auto-masking
    ):

        # NOTE:
        # circunference = 2.0 * np.pi * a
        # n = np.max([
        #     int(circunference / 2.0),
        #     len(parameters.keys()) + 4
        # ])

        # NOTE: area of a circle
        area = np.pi * a**2.0

        # # NOTE: area of an ellipse
        # b = a * np.sqrt(1.0 - parameters["ellipticity"]**2.0)
        # #area = np.pi * a * b

        # NOTE: circumference of an ellipse or circle if b = a.
        b = a
        C = np.round(2.0 * np.pi * np.sqrt((a**2.0 + b**2.0) / 2.0), 1)

        # NOTE: counterclockwise from the positive x-axis
        #n = np.min([500, int(area)]);print("n = ", n)
        n = np.min([500, int(C)])#;print("n = ", n)
        angles = np.linspace(
            0.0, 2.0 * np.pi, n
        )[1:]

        # NOTE:
        x, y = func(a=a, parameters=parameters, angles=angles)



        # # NOTE:
        # plt.figure()
        # plt.imshow(self.image, cmap="jet")
        # plt.plot(x, y, marker=".", color="w")
        # #plt.xlim(0, self.shape[1])
        # #plt.ylim(0, self.shape[0])
        # plt.xlim(parameters["x0"] - 2.0 * a, parameters["x0"] + 2.0 * a)
        # plt.ylim(parameters["y0"] - 2.0 * a, parameters["y0"] + 2.0 * a)
        # plt.show()
        # #exit()

        # NOTE:
        points = np.stack(
            arrays=(y, x), axis=-1
        )


        # NOTE:
        points_interp = self.image_interp(points)
        if self.mask is not None:
            for i, point in enumerate(points):
                y_index = int(point[0])
                x_index = int(point[1])
                if self.mask[y_index, x_index]:
                    points_interp[i] = np.nan

        # NOTE:
        if self.error_interp is not None:
            points_errors_interp = self.error_interp(points)
        else:
            points_errors_interp = None

        # NOTE:
        if condition:
            idx_init = np.isnan(points_interp)
            mean, _, std = stats.sigma_clipped_stats(
                points_interp[~idx_init], sigma=3.0
            )
            idx = points_interp > mean + 3.0 * std
            points_interp[idx] = np.nan


        return points_interp, points_errors_interp, (x, y), angles


class IsophoteFitter:

    def __init__(
        self,
        sample: Sample,
        a: np.float
    ):

        # NOTE:
        self.sample = sample

        # NOTE:
        self.a = a

        # NOTE:
        self.parameters = None

        # NOTE:
        self.n = 0


    def fit(
        self,
        parameters_0,
        fixed_centre=False,
        harmonics:list = False, #NOTE: examples: [], ["m3", "m4"]
    ):

        def fit_func(params):

            # NOTE:
            parameters = {
                "x0":params['x0'].value,
                "y0":params['y0'].value,
                "angle":params['angle'].value,
                "ellipticity":params['ellipticity'].value,
                "a_1":None if "a_1" not in params.keys()
                    else params['a_1'].value,
                "b_1":None if "b_1" not in params.keys()
                    else params['b_1'].value,
                "a_3":None if "a_3" not in params.keys()
                    else params['a_3'].value,
                "b_3":None if "b_3" not in params.keys()
                    else params['b_3'].value,
                "a_4":None if "a_4" not in params.keys()
                    else params['a_4'].value,
                "b_4":None if "b_4" not in params.keys()
                    else params['b_4'].value,
            }

            # NOTE:
            self.parameters = parameters

            # NOTE:
            y_fit, y_errors_fit, _, _ = self.sample.extract(
                a=self.a, parameters=parameters
            )
            #exit()

            # NOTE:
            if y_errors_fit is None:
                #y_errors_fit = 0.05 * y_fit
                raise NotImplementedError()

            # NOTE:
            residuals = (y_fit - np.nanmean(y_fit)) / y_errors_fit
            #print("chi_square =", np.sum(residuals**2.0))

            # NOTE:
            self.n += 1
            #print("n =", self.n)

            # NOTE:
            idx = np.logical_or(np.isnan(residuals), np.isinf(residuals))
            residuals[idx] = 0.0
            #print("residuals =", residuals)

            # NOTE:
            #print("chi-square =", np.sum(residuals**2.0))


            return residuals

        # NOTE:
        params = Parameters()
        params.add(
            'x0',
            value=parameters_0["x0"],
            vary=False if fixed_centre else True,
        )
        params.add(
            'y0',
            value=parameters_0["y0"],
            vary=False if fixed_centre else True,
        )
        params.add(
            'angle',
            value=parameters_0["angle"],
            min=-np.pi/2.0,
            max=np.pi/2.0,
            # min=0.0,
            # max=np.pi,
        )
        params.add(
            'ellipticity',
            value=parameters_0["ellipticity"],
            min=0.0,
            max=1.0,
        )

        """
        # NOTE:
        if harmonics:
            # params.add(
            #     'a_1',
            #     value=0.0,
            # )
            # params.add(
            #     'b_1',
            #     value=0.0,
            # )
            params.add(
                'a_3',
                value=0.0,
            )
            params.add(
                'b_3',
                value=0.0,
            )
            params.add(
                'a_4',
                value=0.0,
            )
            params.add(
                'b_4',
                value=0.0,
            )
        """
        if not harmonics:
            pass
        else:
            if "m1" in harmonics:
                params.add(
                    'a_1',
                    value=0.0 if parameters_0["a_1"] is None else parameters_0["a_1"],
                )
                params.add(
                    'b_1',
                    value=0.0 if parameters_0["b_1"] is None else parameters_0["b_1"],
                )
            # if "m2" in harmonics:
            #     params.add(
            #         'a_2',
            #         value=0.0,
            #     )
            #     params.add(
            #         'b_2',
            #         value=0.0,
            #     )
            if "m3" in harmonics:
                params.add(
                    'a_3',
                    value=0.0 if parameters_0["a_3"] is None else parameters_0["a_3"],
                )
                params.add(
                    'b_3',
                    value=0.0 if parameters_0["b_3"] is None else parameters_0["b_3"],
                )
            if "m4" in harmonics:
                params.add(
                    'a_4',
                    value=0.0 if parameters_0["a_4"] is None else parameters_0["a_4"],
                )
                params.add(
                    'b_4',
                    value=0.0 if parameters_0["b_4"] is None else parameters_0["b_4"],
                )

        # NOTE:
        result = minimize(
            fit_func,
            params,
            #method="lbfgsb",
            #args=(),
            max_nfev=20000,
        )



        # # NOTE:
        # plt.figure()
        # plt.plot(result.residual, linestyle="None", marker="o", color="b")
        # plt.show()
        # exit()

        # print(
        #     result.params['x0'].value,
        #     result.params['y0'].value,
        #     result.params['angle'].value,
        #     result.params['ellipticity'].value,
        #     result.params['a_1'].value,
        #     result.params['b_1'].value,
        #     result.params['a_3'].value,
        #     result.params['b_3'].value,
        #     result.params['a_4'].value,
        #     result.params['b_4'].value,
        # )

        return (
            self.parameters, result,
        )


"""
class MultipleIsophoteFitter:

    def __init__(self, sample, a_array):

        # NOTE:
        self.sample = sample

        # NOTE:
        self.a_array = a_array

        # NOTE:
        self.n = 0

    def fit(
        self,
        list_of_parameters_0,
        fixed_centre=False,
        harmonics=False
    ):

        def fit_func(params):

            # NOTE:
            l_a1 = np.zeros(shape=len(self.a_array))
            l_b1 = np.zeros(shape=len(self.a_array))
            l_a3 = np.zeros(shape=len(self.a_array))
            l_b3 = np.zeros(shape=len(self.a_array))
            l_a4 = np.zeros(shape=len(self.a_array))
            l_b4 = np.zeros(shape=len(self.a_array))

            # NOTE:
            residuals = []
            for n, a in enumerate(self.a_array):
                parameters = {
                    "x0":params[
                        'ell_{}_x0'.format(n)
                    ].value,
                    "y0":params[
                        'ell_{}_y0'.format(n)
                    ].value,
                    "angle":params[
                        'ell_{}_angle'.format(n)
                    ].value,
                    "ellipticity":params[
                        'ell_{}_ellipticity'.format(n)
                    ].value,
                    "a_1":None if "ell_{}_a_1".format(n) not in params.keys()
                        else params['ell_{}_a_1'.format(n)].value,
                    "b_1":None if "ell_{}_b_1".format(n) not in params.keys()
                        else params['ell_{}_b_1'.format(n)].value,
                    "a_3":None if "ell_{}_a_3".format(n) not in params.keys()
                        else params['ell_{}_a_3'.format(n)].value,
                    "b_3":None if "ell_{}_b_3".format(n) not in params.keys()
                        else params['ell_{}_b_3'.format(n)].value,
                    "a_4":None if "ell_{}_a_4".format(n) not in params.keys()
                        else params['ell_{}_a_4'.format(n)].value,
                    "b_4":None if "ell_{}_b_4".format(n) not in params.keys()
                        else params['ell_{}_b_4'.format(n)].value,
                }

                # NOTE:
                l_a1[n] = parameters["a_1"]
                l_b1[n] = parameters["b_1"]
                l_a3[n] = parameters["a_3"]
                l_b3[n] = parameters["b_3"]
                l_a4[n] = parameters["a_4"]
                l_b4[n] = parameters["b_4"]

                # NOTE:
                y_fit, y_errors_fit, _, _ = self.sample.extract(a=a, parameters=parameters)

                # NOTE:
                if y_errors_fit is None:
                    y_errors_fit = 0.05 * y_fit

                # NOTE:
                residuals.append(
                    [(y_fit - np.mean(y_fit)) / y_errors_fit]
                )

            # NOTE:
            residuals = np.hstack(residuals)

            # NOTE:
            idx = np.logical_or(np.isnan(residuals), np.isinf(residuals))
            residuals[idx] = 0.0

            # NOTE:
            self.n += 1
            print(
                "n =", self.n
            )
            #print(l)

            # NOTE:
            chi_square = np.sum(np.square(residuals))
            print("chi_square =", chi_square)

            # NOTE:
            reg_term_a1 = 0.1 * np.sum(np.square(l_a1))
            reg_term_b1 = 0.1 * np.sum(np.square(l_b1))
            reg_term_a3 = 0.1 * np.sum(np.square(l_a3))
            reg_term_b3 = 0.1 * np.sum(np.square(l_b3))
            reg_term_a4 = 0.1 * np.sum(np.square(l_a4))
            reg_term_b4 = 0.1 * np.sum(np.square(l_b4))
            # reg_term_a1 = 1.0 * np.sum(np.square(l_a1[1:] - l_a1[:-1]))
            # reg_term_b1 = 1.0 * np.sum(np.square(l_b1[1:] - l_b1[:-1]))
            # reg_term_a3 = 1.0 * np.sum(np.square(l_a3[1:] - l_a3[:-1]))
            # reg_term_b3 = 1.0 * np.sum(np.square(l_b3[1:] - l_b3[:-1]))
            # reg_term_a4 = 1.0 * np.sum(np.square(l_a4[1:] - l_a4[:-1]))
            # reg_term_b4 = 1.0 * np.sum(np.square(l_b4[1:] - l_b4[:-1]))
            reg_tems = np.sum([
                reg_term_a1,
                reg_term_b1,
                reg_term_a3,
                reg_term_b3,
                reg_term_a4,
                reg_term_b4,
            ])


            # # NOTE:
            # reg_term_1 = 1e6 * np.sum(np.square(l1[1:] - l1[:-1]))
            # reg_term_2 = 1e6 * np.sum(np.square(l2[1:] - l2[:-1]))
            # print("reg_terms = ", reg_term_1, reg_term_2)

            likelihood = chi_square #+ reg_tems
            print("likelihood =", likelihood)

            return likelihood

        # NOTE:
        params = Parameters()
        for n, parameters in enumerate(list_of_parameters_0):
            print("n =", n)
            params.add(
                'ell_{}_x0'.format(n),
                value=parameters["x0"],
                vary=False if fixed_centre else True,
            )
            params.add(
                'ell_{}_y0'.format(n),
                value=parameters["y0"],
                vary=False if fixed_centre else True,
            )
            params.add(
                'ell_{}_angle'.format(n),
                value=parameters["angle"],
                min=-np.pi,
                max=np.pi,
            )
            params.add(
                'ell_{}_ellipticity'.format(n),
                value=parameters["ellipticity"],
                min=-1.0,
                max=1.0,
            )

            # NOTE:
            if "a_1" and "b_1" in parameters.keys():
                params.add(
                    'ell_{}_a_1'.format(n),
                    value=parameters["a_1"],
                )
                params.add(
                    'ell_{}_b_1'.format(n),
                    value=parameters["b_1"],
                )
            if "a_3" and "b_3" in parameters.keys():
                params.add(
                    'ell_{}_a_3'.format(n),
                    value=parameters["a_3"],
                )
                params.add(
                    'ell_{}_b_3'.format(n),
                    value=parameters["b_3"],
                )
            if "a_4" and "b_4" in parameters.keys():
                params.add(
                    'ell_{}_a_4'.format(n),
                    value=parameters["a_4"],
                )
                params.add(
                    'ell_{}_b_4'.format(n),
                    value=parameters["b_4"],
                )

        # NOTE:
        result = minimize(
            fit_func,
            params,
            #args=(),
            method='lbfgsb',
            max_nfev=100000,
        )

        # NOTE:
        list_of_parameters = []
        list_of_parameters_errors = []
        for n in range(len(self.a_array)):
            parameters = {
                "x0":result.params[
                    'ell_{}_x0'.format(n)
                ].value,
                "y0":result.params[
                    'ell_{}_y0'.format(n)
                ].value,
                "angle":result.params[
                    'ell_{}_angle'.format(n)
                ].value,
                "ellipticity":result.params[
                    'ell_{}_ellipticity'.format(n)
                ].value,
                "a_1":None if "ell_{}_a_1".format(n) not in result.params.keys()
                    else result.params['ell_{}_a_1'.format(n)].value,
                "b_1":None if "ell_{}_b_1".format(n) not in result.params.keys()
                    else result.params['ell_{}_b_1'.format(n)].value,
                "a_3":None if "ell_{}_a_3".format(n) not in result.params.keys()
                    else result.params['ell_{}_a_3'.format(n)].value,
                "b_3":None if "ell_{}_b_3".format(n) not in result.params.keys()
                    else result.params['ell_{}_b_3'.format(n)].value,
                "a_4":None if "ell_{}_a_4".format(n) not in result.params.keys()
                    else result.params['ell_{}_a_4'.format(n)].value,
                "b_4":None if "ell_{}_b_4".format(n) not in result.params.keys()
                    else result.params['ell_{}_b_4'.format(n)].value,
            }
            list_of_parameters.append(parameters)

            errors = {
                "x0":result.params[
                    'ell_{}_x0'.format(n)
                ].stderr,
                "y0":result.params[
                    'ell_{}_y0'.format(n)
                ].stderr,
                "angle":result.params[
                    'ell_{}_angle'.format(n)
                ].stderr,
                "ellipticity":result.params[
                    'ell_{}_ellipticity'.format(n)
                ].stderr,
                "a_1":None if "ell_{}_a_1".format(n) not in result.params.keys()
                    else result.params['ell_{}_a_1'.format(n)].stderr,
                "b_1":None if "ell_{}_b_1".format(n) not in result.params.keys()
                    else result.params['ell_{}_b_1'.format(n)].stderr,
                "a_3":None if "ell_{}_a_3".format(n) not in result.params.keys()
                    else result.params['ell_{}_a_3'.format(n)].stderr,
                "b_3":None if "ell_{}_b_3".format(n) not in result.params.keys()
                    else result.params['ell_{}_b_3'.format(n)].stderr,
                "a_4":None if "ell_{}_a_4".format(n) not in result.params.keys()
                    else result.params['ell_{}_a_4'.format(n)].stderr,
                "b_4":None if "ell_{}_b_4".format(n) not in result.params.keys()
                    else result.params['ell_{}_b_4'.format(n)].stderr,
            }
            list_of_parameters_errors.append(errors)

        return list_of_parameters
"""

"""
class MultipleIsophoteFitterConstantHarmonics:

    def __init__(
        self,
        sample,
        a_array,
    ):

        # NOTE:
        self.sample = sample

        # NOTE:
        self.a_array = a_array

        # NOTE:
        self.n = 0

    def fit(
        self,
        list_of_parameters_0,
        fixed_centre=False,
        #harmonics=False
    ):

        def fit_func(params):
            residuals = []
            for n, a in enumerate(self.a_array):
                parameters = {
                    "x0":params[
                        'ell_{}_x0'.format(n)
                    ].value,
                    "y0":params[
                        'ell_{}_y0'.format(n)
                    ].value,
                    "angle":params[
                        'ell_{}_angle'.format(n)
                    ].value,
                    "ellipticity":params[
                        'ell_{}_ellipticity'.format(n)
                    ].value,
                    "a_1":None if "a_1" not in params.keys()
                        else params['a_1'].value,
                    "b_1":None if "b_1" not in params.keys()
                        else params['b_1'].value,
                    "a_3":None if "a_3" not in params.keys()
                        else params['a_3'].value,
                    "b_3":None if "b_3" not in params.keys()
                        else params['b_3'].value,
                    "a_4":None if "a_4" not in params.keys()
                        else params['a_4'].value,
                    "b_4":None if "b_4" not in params.keys()
                        else params['b_4'].value,
                }
                # print(parameters["a_1"], parameters["b_1"])
                # print(parameters["a_3"], parameters["b_3"])
                # print(parameters["a_4"], parameters["b_4"])

                # NOTE:
                y_fit, y_errors_fit, _, _ = self.sample.extract(a=a, parameters=parameters)

                # NOTE:
                if y_errors_fit is None:
                    y_errors_fit = 0.05 * y_fit

                # NOTE:
                residuals.append(
                    [(y_fit - np.mean(y_fit)) / y_errors_fit]
                )


            # NOTE:
            residuals = np.hstack(residuals)

            # NOTE:
            idx = np.logical_or(np.isnan(residuals), np.isinf(residuals))
            residuals[idx] = 0.0

            # NOTE:
            self.n += 1
            print(
                "n =", self.n
            )

            # NOTE:
            chi_square = np.sum(np.square(residuals))
            print("chi_square =", chi_square)

            if chi_square == 0.0:
                return 1e8
            else:
                return chi_square

        # NOTE:
        params = Parameters()
        for n, parameters in enumerate(list_of_parameters_0):
            print("n =", n)
            params.add(
                'ell_{}_x0'.format(n),
                value=parameters["x0"],
                vary=False if fixed_centre else True,
            )
            params.add(
                'ell_{}_y0'.format(n),
                value=parameters["y0"],
                vary=False if fixed_centre else True,
            )
            params.add(
                'ell_{}_angle'.format(n),
                value=parameters["angle"],
                min=-np.pi,
                max=np.pi,
            )
            params.add(
                'ell_{}_ellipticity'.format(n),
                value=parameters["ellipticity"],
                min=-1.0,
                max=1.0,
            )
        params.add(
            'a_1',
            value=0.0,
        )
        params.add(
            'b_1',
            value=0.0,
        )
        params.add(
            'a_3',
            value=0.0,
        )
        params.add(
            'b_3',
            value=0.0,
        )
        params.add(
            'a_4',
            value=0.0,
        )
        params.add(
            'b_4',
            value=0.0,
        )

        # NOTE:
        result = minimize(
            fit_func,
            params,
            #args=(),
            method='lbfgsb',
            max_nfev=100000,
        )

        print(
            result.params["a_1"].value,
            result.params["b_1"].value,
            result.params["a_3"].value,
            result.params["b_3"].value,
            result.params["a_4"].value,
            result.params["b_4"].value,
        )
        #exit()

        # NOTE:
        list_of_parameters = []
        for n in range(len(self.a_array)):
            parameters = {
                "x0":result.params[
                    'ell_{}_x0'.format(n)
                ].value,
                "y0":result.params[
                    'ell_{}_y0'.format(n)
                ].value,
                "angle":result.params[
                    'ell_{}_angle'.format(n)
                ].value,
                "ellipticity":result.params[
                    'ell_{}_ellipticity'.format(n)
                ].value,
                "a_1":result.params['a_1'].value,
                "b_1":result.params['a_1'].value,
                "a_3":result.params['a_3'].value,
                "b_3":result.params['b_3'].value,
                "a_4":result.params['a_4'].value,
                "b_4":result.params['b_4'].value,
                # "a_1":0.0,
                # "b_1":0.0,
                # "a_3":0.0,
                # "b_3":0.0,
                # "a_4":0.0,
                # "b_4":0.0,
            }
            list_of_parameters.append(parameters)

        return list_of_parameters
"""

class main:

    def __init__(
        self,
        image,
        error=None,
        mask=None,
        a_min=None,
        a_max=None,
        a_n=20,
        parameters_0=None,
        extract_condition=True
    ):

        # NOTE:
        self.image = image
        if error is None:
            self.error = None
        else:
            self.error = error

        # NOTE:
        self.mask = mask

        # NOTE:
        self.sample = Sample(
            image=self.image,
            error=self.error,
            mask=self.mask,
        )

        # NOTE:
        self.parameters_0 = parameters_0

        # NOTE:
        self.list_of_parameters = []
        self.list_of_parameters_errors = []

        # NOTE:
        self.a_min = a_min
        self.a_max = a_max
        # self.array = np.linspace(
        #     self.a_min, self.a_max, a_n
        # )
        self.array = np.logspace(
            np.log10(self.a_min),
            np.log10(self.a_max),
            a_n
        )

        # NOTE:
        self.parameters_previous = None

        # NOTE:
        self.extract_condition = extract_condition


        # self.array = []
        # a_previous = self.a_min
        # step=0.3
        # a_next = a_previous * (1.0 + step)
        # self.array.append(a_next)
        # a_previous = a_next
        # while a_previous < self.a_max:
        #     a_next = a_previous * (1.0 + step)
        #     self.array.append(a_next)
        #     a_previous = a_next
        # print(self.array)
        # self.array = np.asarray(self.array)

    """
    def fit_image_with_multifitter_and_constant_harmonics(
        self,
        list_of_parameters_0,
        fixed_centre=False,
        #harmonics=False,
        #refine=False
    ):

        # NOTE:
        fitter = MultipleIsophoteFitterConstantHarmonics(
            sample=self.sample, a_array=self.array
        )

        list_of_parameters = fitter.fit(
            list_of_parameters_0=list_of_parameters_0,
            fixed_centre=fixed_centre,
            #harmonics=harmonics
        )

        return list_of_parameters, None
    """

    def fit_image(
        self,
        fixed_centre=False,
        harmonics=False,
        refine=False
    ):

        chi_squares = []
        for i, a in enumerate(self.array):
            print("i =", i, "|", "a =", a)

            # NOTE:
            fitter = IsophoteFitter(
                sample=self.sample, a=a
            )

            # NOTE:
            # print(
            #     "parameters_0 =", self.parameters_0 if self.parameters_previous is None else self.parameters_previous
            # )
            parameters, result = fitter.fit(
                parameters_0=self.parameters_0 if self.parameters_previous is None else self.parameters_previous,
                fixed_centre=fixed_centre,
                harmonics=harmonics
            )
            print("parameters =", parameters)
            #print("goodness of fit =", result.chisqr)
            #exit()

            # NOTE:
            visualize = False
            if visualize:
                figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
                image = self.sample.image
                axes[0].imshow(
                    image,
                    cmap="jet",
                    aspect="auto",
                    # norm=matplotlib.colors.LogNorm(
                    #     vmin=2.5, vmax=500.0
                    # )
                )
                y_fit, y_errors_fit, (x, y), _ = self.sample.extract(
                    a=a, parameters=parameters, condition=self.extract_condition
                )
                axes[0].plot(x, y, marker=".", color="w")
                axes[0].set_xlim(0, image.shape[1])
                axes[0].set_ylim(0, image.shape[0])

                x_cen = parameters["x0"]
                y_cen = parameters["y0"]
                x_min = parameters["x0"] - 2.0 * a * np.cos(parameters["angle"])
                y_min = parameters["y0"] - 2.0 * a * np.sin(parameters["angle"])
                x_max = parameters["x0"] + 2.0 * a * np.cos(parameters["angle"])
                y_max = parameters["y0"] + 2.0 * a * np.sin(parameters["angle"])
                axes[0].plot(
                    [x_min, x_cen, x_max],
                    [y_min, y_cen, y_max],
                    marker="o",
                    color="grey"
                )
                axes[0].axvline(parameters["x0"], linestyle=":", color="grey")
                axes[0].axhline(parameters["y0"], linestyle=":", color="grey")

                axes[1].plot(y_fit, marker="o", color="black")
                axes[1].set_ylim(0.0, np.nanmax(image))
                plt.show()
                #exit()

            # NOTE:
            self.list_of_parameters.append(parameters)

            # NOTE:
            errors = {}
            for key in parameters.keys():
                if key in result.params.keys():
                    errors[key] = result.params[key].stderr
            self.list_of_parameters_errors.append(errors)

            # NOTE:
            #self.parameters_0 = parameters
            self.parameters_previous = parameters

            # NOTE:
            chi_squares.append(result.chisqr)

        # plt.figure()
        # plt.plot(self.array, chi_squares, marker="o", color="b")
        # #plt.show()

        if refine:
            list_of_parameters = self.fit_refinned(
                list_of_parameters_0=self.list_of_parameters,
                fixed_centre=True,
                harmonics=harmonics
            )
            self.list_of_parameters = list_of_parameters



        # NOTE:
        visualize = False
        if visualize:
            logscale = True
            figure, axes = plt.subplots(
                nrows=1, ncols=2, figsize=(18, 8)
            )
            image = self.sample.image
            axes[0].imshow(
                np.log10(image) if logscale else image,
                cmap="jet",
                aspect="auto",
                # # NOTE: __main__
                # norm=matplotlib.colors.LogNorm(
                #     vmin=0.05, vmax=10.0
                # )
                # NOTE: NGC 0057; F110W
                # norm=matplotlib.colors.LogNorm(
                #     vmin=2.5, vmax=500.0
                # )
                # NOTE: NGC 0057; F475X
                # norm=matplotlib.colors.LogNorm(
                #     vmin=0.25, vmax=5.0
                # )
                # # NOTE: NGC 0315; F110W
                # norm=matplotlib.colors.LogNorm(
                #     vmin=2.5, vmax=500.0
                # )
                # # NOTE: SPT2147; F200W
                # norm=matplotlib.colors.LogNorm(
                #     vmin=0.2, vmax=5.0
                # )
            )
            y_means = []
            y_stds = []
            for a, parameters in zip(self.array, self.list_of_parameters):
                y_fit, y_errors_fit, (x, y), angles = self.sample.extract(
                    a=a, parameters=parameters, condition=self.extract_condition
                )

                # NOTE:
                y_mean = np.nanmean(y_fit)
                y_means.append(y_mean)

                # NOTE:
                y_std = np.nanstd(y_fit)
                y_stds.append(y_std)

                # NOTE:
                axes[0].plot(
                    x,
                    y,
                    marker="o",
                    markersize=2.5,
                    color="w"
                )

                # NOTE:
                axes[1].errorbar(
                    angles * units.rad.to(units.deg),
                    y_fit,
                    yerr=0.05 * y_fit,
                    linestyle="None",
                    marker="o",
                    markersize=2.5,
                    color="black"
                )
                axes[1].axhline(
                    y_mean,
                    linestyle="--",
                    color="r"
                )

            axes[0].contour(
                np.log10(image) if logscale else image,
                #levels=y_means[::-1],
                levels=np.sort(np.log10(y_means)) if logscale else np.sort(y_means),
                colors="black"
            )

            # axes[0].plot(
            #     [247],
            #     [250],
            #     marker="o"
            # )
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            axes[1].set_yscale("log")
            axes[0].set_xlim(0, image.shape[1])
            axes[0].set_ylim(0, image.shape[0])

            x0 = self.list_of_parameters[0]["x0"]
            y0 = self.list_of_parameters[0]["y0"]
            axes[0].set_xlim(x0 - self.a_max, x0 + self.a_max)
            axes[0].set_ylim(y0 - self.a_max, y0 + self.a_max)

            #axes[1].set_ylim(self.a_min, self.a_max)
            # x_cen = parameters["x0"]
            # y_cen = parameters["y0"]
            # x_min = parameters["x0"] - 2.0 * a * np.cos(parameters["angle"])
            # y_min = parameters["y0"] - 2.0 * a * np.sin(parameters["angle"])
            # x_max = parameters["x0"] + 2.0 * a * np.cos(parameters["angle"])
            # y_max = parameters["y0"] + 2.0 * a * np.sin(parameters["angle"])
            # axes[0].plot(
            #     [x_min, x_cen, x_max],
            #     [y_min, y_cen, y_max],
            #     marker="o",
            #     color="grey"
            # )
            # axes[0].axvline(parameters["x0"], linestyle=":", color="grey")
            # axes[0].axhline(parameters["y0"], linestyle=":", color="grey")

            figure.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

            # === #
            # NOTE:
            # === #
            # figure, axes = plt.subplots(figsize=(8, 8))
            # plt.errorbar(
            #     self.array,
            #     y_means,
            #     yerr=y_stds,
            #     linestyle="None",
            #     marker="o",
            # )
            # plt.xscale("log")
            # #plt.yscale("log")
            # === #

            # === #
            # NOTE:
            # === #
            """
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
            ax.plot(
                [
                    parameters["angle"]
                    for parameters in self.list_of_parameters
                ],
                np.log(self.array),
                marker="o",
                color="black",
            )
            ax.plot(
                [
                    parameters["angle"] - 3.0 * errors["angle"]
                    for parameters, errors in zip(self.list_of_parameters, self.list_of_parameters_errors)
                ],
                np.log(self.array),
                color="black",
                alpha=0.5,
            )
            ax.plot(
                [
                    parameters["angle"] + 3.0 * errors["angle"]
                    for parameters, errors in zip(self.list_of_parameters, self.list_of_parameters_errors)
                ],
                np.log(self.array),
                color="black",
                alpha=0.5,
            )

            ax.plot(
                [
                    parameters["angle"] + np.pi
                    for parameters in self.list_of_parameters
                ],
                np.log(self.array),
                marker="o",
                color="black",
            )
            # ax.plot(
            #     [
            #         np.arctan2(parameters["b_1"], parameters["a_1"])
            #         for parameters in self.list_of_parameters
            #     ],
            #     np.log(self.array),
            #     marker="o",
            #     color="r",
            # )

            ax.set_rticks([])
            """
            # === #

        return self.list_of_parameters, self.list_of_parameters_errors

    """
    def fit_refinned(
        self,
        list_of_parameters_0,
        fixed_centre=False,
        harmonics=False
    ):


        # NOTE:
        fitter = MultipleIsophoteFitter(
            sample=self.sample, a_array=self.array
        )

        list_of_parameters = fitter.fit(
            list_of_parameters_0=list_of_parameters_0, fixed_centre=fixed_centre, harmonics=harmonics
        )

        return list_of_parameters
    """
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #


def plot(
    r,
    list_of_parameters,
    list_of_parameters_errors=None,
    list_of_parameters_i=None,
    color_i="r"
):

    fig, ax = plt.subplots(
        subplot_kw={'projection': 'polar'}, figsize=(8, 8)
    )
    ax.plot(
        [
            parameters["angle"]
            for parameters in list_of_parameters
        ],
        r,
        marker="o",
        color="black",
    )
    ax.plot(
        [
            parameters["angle"] + np.pi
            for parameters in list_of_parameters
        ],
        r,
        marker="o",
        color="black",
    )

    # NOTE:
    # ax.plot(
    #     [
    #         parameters["angle"] - 3.0 * errors["angle"]
    #         for parameters, errors in zip(self.list_of_parameters, self.list_of_parameters_errors)
    #     ],
    #     np.log(self.array),
    #     color="black",
    #     alpha=0.5,
    # )
    # ax.plot(
    #     [
    #         parameters["angle"] + 3.0 * errors["angle"]
    #         for parameters, errors in zip(self.list_of_parameters, self.list_of_parameters_errors)
    #     ],
    #     np.log(self.array),
    #     color="black",
    #     alpha=0.5,
    # )

    if list_of_parameters_i is not None:
        ax.plot(
            [
                parameters["angle"]
                for parameters in list_of_parameters_i
            ],
            r,
            marker="o",
            color=color_i,
        )
        ax.plot(
            [
                parameters["angle"] + np.pi
                for parameters in list_of_parameters_i
            ],
            r,
            marker="o",
            color=color_i,
        )

    ax.set_rticks([])

# NOTE:
def plot_list_of_parameters(
    list_of_parameters,
    list_of_parameters_errors,
    x=None,
    show_harmonics=True,
    show_centers=False,
    xscale="log"
):

    if x is None:
        x = len(list_of_parameters)
    else:
        pass


    # if show_centers:
    #     figure, axes = plt.subplots(
    #         nrows=1,
    #         ncols=2,
    #         figsize=(10, 5)
    #     )
    #
    #     # NOTE: axes = 0
    #     y = [
    #         parameters["x0"] for parameters in list_of_parameters
    #     ]
    #     if list_of_parameters_errors is not None:
    #         yerr = [
    #             errors["x0"] for errors in list_of_parameters_errors
    #         ]
    #     else:
    #         yerr = None
    #     axes[0].errorbar(
    #         x=x,
    #         y=y,
    #         yerr=yerr,
    #         linestyle="None",
    #         marker="o",
    #         color="black"
    #     )
    #
    #     # NOTE: axes = 1
    #     y = [
    #         parameters["y0"] for parameters in list_of_parameters
    #     ]
    #     if list_of_parameters_errors is not None:
    #         yerr = [
    #             errors["y0"] for errors in list_of_parameters_errors
    #         ]
    #     else:
    #         yerr = None
    #     axes[1].errorbar(
    #         x=x,
    #         y=y,
    #         yerr=yerr,
    #         linestyle="None",
    #         marker="o",
    #         color="black"
    #     )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(18, 6)
    )
    axes_flattened = np.ndarray.flatten(axes)


    # NOTE:
    keys = list(list_of_parameters[0].keys())

    # NOTE:
    y = [
        parameters["ellipticity"] for parameters in list_of_parameters
    ]

    axes_flattened[0].errorbar(
        x=x,
        y=y,
        yerr=[
            errors["ellipticity"] for errors in list_of_parameters_errors
        ] if list_of_parameters_errors is not None else None,
        linestyle="None",
        marker="o",
        color="black"
    )
    axes_flattened[1].errorbar(
        x=x,
        y=[
            parameters["angle"] for parameters in list_of_parameters
        ],
        yerr=[
            errors["angle"] for errors in list_of_parameters_errors
        ] if list_of_parameters_errors is not None else None,
        linestyle="None",
        marker="o",
        color="black"
    )

    # NOTE:
    if show_harmonics:
        if np.logical_and(
            "a_1" in keys,
            "b_1" in keys
        ):
            try:
                y_2 = [parameters["a_1"] for parameters in list_of_parameters]
                y_3 = [parameters["b_1"] for parameters in list_of_parameters]
                if list_of_parameters_errors is not None:
                    y_2_errors = [
                        errors["a_1"] for errors in list_of_parameters_errors
                    ]
                    y_3_errors = [
                        errors["a_1"] for errors in list_of_parameters_errors
                    ]
                else:
                    y_2_errors = None
                    y_3_errors = None
                y_2_min = np.min(y_2)
                y_3_min = np.min(y_3)
                y_2_max = np.max(y_2)
                y_3_max = np.max(y_3)
                ymin = np.min([y_2_min, y_3_min])
                ymax = np.max([y_2_max, y_3_max])
                value_max = np.max([
                    abs(y_2_min),
                    abs(y_2_max),
                    abs(y_3_min),
                    abs(y_3_max),
                ])
                axes_flattened[2].errorbar(
                    x,
                    y=y_2,
                    yerr=y_2_errors,
                    linestyle="None",
                    marker="o",
                    color="black"
                )
                axes_flattened[3].errorbar(
                    x,
                    y=y_3,
                    yerr=y_3_errors,
                    linestyle="None",
                    marker="o",
                    color="black"
                )
                #axes_flattened[2].set_ylim(-1.25 * value_max, 1.25 * value_max)
                #axes_flattened[3].set_ylim(-1.25 * value_max, 1.25 * value_max)
                axes_flattened[2].set_ylim(-5.0, 5.0)
                axes_flattened[3].set_ylim(-5.0, 5.0)
            except:
                pass
        if np.logical_and(
            "a_3" in keys,
            "b_3" in keys,
        ):
            try:
                y_4 = [parameters["a_3"] for parameters in list_of_parameters]
                y_5 = [parameters["b_3"] for parameters in list_of_parameters]
                if list_of_parameters_errors is not None:
                    y_4_errors = [
                        errors["a_3"] for errors in list_of_parameters_errors
                    ]
                    y_5_errors = [
                        errors["a_3"] for errors in list_of_parameters_errors
                    ]
                else:
                    y_4_errors = None
                    y_5_errors = None
                y_4_min = np.min(y_4)
                y_5_min = np.min(y_5)
                y_4_max = np.max(y_4)
                y_5_max = np.max(y_5)
                ymin = np.min([y_4_min, y_5_min])
                ymax = np.max([y_4_max, y_5_max])
                value_max = np.max([
                    abs(y_4_min),
                    abs(y_4_max),
                    abs(y_5_min),
                    abs(y_5_max),
                ])
                axes_flattened[4].errorbar(
                    x=x,
                    y=y_4,
                    yerr=y_4_errors,
                    linestyle="None",
                    marker="o",
                    color="black"
                )
                axes_flattened[5].errorbar(
                    x=x,
                    y=y_5,
                    yerr=y_5_errors,
                    linestyle="None",
                    marker="o",
                    color="black"
                )
                #axes_flattened[4].set_ylim(-1.25 * value_max, 1.25 * value_max)
                #axes_flattened[5].set_ylim(-1.25 * value_max, 1.25 * value_max)
                axes_flattened[4].set_ylim(-5.0, 5.0)
                axes_flattened[5].set_ylim(-5.0, 5.0)
            except:
                pass
        if np.logical_and(
            "a_4" in keys,
            "b_4" in keys,
        ):
            try:
                y_6 = [parameters["a_4"] for parameters in list_of_parameters]
                y_7 = [parameters["b_4"] for parameters in list_of_parameters]
                if list_of_parameters_errors is not None:
                    y_6_errors = [
                        errors["a_4"] for errors in list_of_parameters_errors
                    ]
                    y_7_errors = [
                        errors["a_4"] for errors in list_of_parameters_errors
                    ]
                else:
                    y_6_errors = None
                    y_7_errors = None
                y_6_min = np.min(y_6)
                y_7_min = np.min(y_7)
                y_6_max = np.max(y_6)
                y_7_max = np.max(y_7)
                ymin = np.min([y_6_min, y_7_min])
                ymax = np.max([y_6_max, y_7_max])
                value_max = np.max([
                    abs(y_6_min),
                    abs(y_6_max),
                    abs(y_7_min),
                    abs(y_7_max),
                ])
                axes_flattened[6].errorbar(
                    x=x,
                    y=y_6,
                    yerr=y_6_errors,
                    linestyle="None",
                    marker="o",
                    color="black"
                )
                axes_flattened[7].errorbar(
                    x=x,
                    y=y_7,
                    yerr=y_7_errors,
                    linestyle="None",
                    marker="o",
                    color="black"
                )
                #axes_flattened[6].set_ylim(-1.25 * value_max, 1.25 * value_max)
                #axes_flattened[7].set_ylim(-1.25 * value_max, 1.25 * value_max)
                axes_flattened[6].set_ylim(-5.0, 5.0)
                axes_flattened[7].set_ylim(-5.0, 5.0)
            except:
                pass

    # NOTE:
    for ax in axes_flattened:

        # NOTE:
        ax.minorticks_on()
        ax.tick_params(
            axis='y',
            which="major",
            length=6,
            right=True,
            top=True,
            direction="in",
            colors='black'
        )
        ax.tick_params(
            axis='y',
            which="minor",
            length=3,
            right=True,
            top=True,
            direction="in",
            colors='black'
        )
        ax.tick_params(
            axis='x',
            which="major",
            length=6,
            bottom=True,
            top=True,
            direction="in",
            colors='black',
        )
        ax.tick_params(
            axis='x',
            which="minor",
            length=3,
            bottom=True,
            top=True,
            direction="in",
            colors='black',
        )

        # NOTE:
        ax.set_xscale(xscale)

    axes_flattened[0].set_ylabel(r"$e$", fontsize=12.5)
    axes_flattened[1].set_ylabel(r"$\theta$", fontsize=12.5)
    axes_flattened[4].set_ylabel(r"$\alpha_3$", fontsize=12.5)
    axes_flattened[5].set_ylabel(r"$\beta_3$", fontsize=12.5)
    axes_flattened[6].set_ylabel(r"$\alpha_4$", fontsize=12.5)
    axes_flattened[7].set_ylabel(r"$\beta_4$", fontsize=12.5)

    axes_flattened[4].set_xlabel(r"$arcsec$", fontsize=12.5)
    axes_flattened[5].set_xlabel(r"$arcsec$", fontsize=12.5)
    axes_flattened[6].set_xlabel(r"$arcsec$", fontsize=12.5)
    axes_flattened[7].set_xlabel(r"$arcsec$", fontsize=12.5)





def plot_multipole_amplitudes_from_list_of_parameters(
    array,
    list_of_parameters,
    list_of_parameters_errors,
    show_harmonics=True,
    ylim=None,
    pixel_scale=1.0,
    name=None
):
    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15, 5)
    )
    figure_angles, axes_angles = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15, 5)
    )

    # NOTE:
    figure_k1_angles, axes_k1_angles = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(6, 5)
    )
    if name == "NGC0741":
        value = 264.81934742754027
    elif name == "NGC1129":
        value = 338.04154445784155
    elif name == "NGC2672":
        value = 204.46247911844327
    else:
        #value = None
        value = 0.0

    angles = np.array([
        parameters["angle"] for parameters in list_of_parameters
    ])
    #print(angles);exit()

    # NOTE:
    keys = list(list_of_parameters[0].keys())

    # NOTE:
    if show_harmonics:
        if np.logical_and(
            "a_1" in keys,
            "b_1" in keys
        ):
            if True:
                try:
                    a1 = np.array([
                        parameters["a_1"] for parameters in list_of_parameters
                    ])
                    b1 = np.array([
                        parameters["b_1"] for parameters in list_of_parameters
                    ])
                    k1 = np.hypot(a1, b1)
                    phi_1 = continuous_angle_conversion(angle=np.arctan2(b1, a1) + angles)
                    if list_of_parameters_errors is not None:
                        a1_errors = np.array([
                            errors["a_1"] for errors in list_of_parameters_errors
                        ])
                        b1_errors = np.array([
                            errors["b_1"] for errors in list_of_parameters_errors
                        ])
                        k1_errors = (a1 / k1)**2.0 * a1_errors**2.0 + (b1 / k1)**2.0 * b1_errors**2.0
                    else:
                        k1_errors = None
                    # y_2_min = np.min(y_2)
                    # y_3_min = np.min(y_3)
                    # y_2_max = np.max(y_2)
                    # y_3_max = np.max(y_3)
                    # ymin = np.min([y_2_min, y_3_min])
                    # ymax = np.max([y_2_max, y_3_max])
                    # value_max = np.max([
                    #     abs(y_2_min),
                    #     abs(y_2_max),
                    #     abs(y_3_min),
                    #     abs(y_3_max),
                    # ])
                    x1 = np.arange(k1.shape[0])
                    axes[0].errorbar(
                        x1 if array is None else array * pixel_scale,
                        y=k1 if array is None else k1/array * 100,
                        yerr=k1_errors if array is None else (k1/array)*(k1_errors/k1) * 100,
                        linestyle="None",
                        marker="o",
                        color="black",
                        label=r"$k_1$"
                    )
                    axes_angles[0].errorbar(
                        x1 if array is None else array * pixel_scale,
                        y=phi_1,
                        #yerr=,
                        linestyle="None",
                        marker="o",
                        color="black",
                        label=r"$k_1$"
                    )
                    def angle_difference(angle1, angle2):
                        diff = angle1 - angle2
                        idx_1 = diff > 180
                        idx_2 = diff < -180
                        diff[idx_1] -= 360
                        diff[idx_2] += 360
                        return diff

                    # directory = "/Users/ccbh87/Desktop/GitHub/utils/MASSIVE/metadata"
                    # pickle_utils.save_obj(directory=directory, filename="/{}_m1_angles".format(name), obj=[array * pixel_scale, phi_1, value])
                    # exit()
                    axes_k1_angles.errorbar(
                        array * pixel_scale,
                        #y=phi_1,
                        y=angle_difference(angle1=value, angle2=phi_1),
                        #yerr=,
                        linestyle="None",
                        marker="o",
                        color="navy",
                        label=r"$k_1$"
                    )
                    #axes_flattened[2].set_ylim(-value_max, value_max)
                    #axes_flattened[3].set_ylim(-value_max, value_max)
                except:
                    k1 = None
                    k1_errors = None
            else:
                pass
        if np.logical_and(
            "a_3" in keys,
            "b_3" in keys,
        ):
            if True:
                a3 = np.array([
                    parameters["a_3"] for parameters in list_of_parameters
                ])
                b3 = np.array([
                    parameters["b_3"] for parameters in list_of_parameters
                ])
                if any(_ is not None for _ in a3):
                    k3 = np.hypot(a3, b3)
                    phi_3 = continuous_angle_conversion(angle=np.arctan2(b3, a3) + angles)
                    if list_of_parameters_errors is not None:
                        a3_errors = np.array([
                            errors["a_3"] for errors in list_of_parameters_errors
                        ])
                        b3_errors = np.array([
                            errors["b_3"] for errors in list_of_parameters_errors
                        ])
                        k3_errors = (a3 / k3)**2.0 * a3_errors**2.0 + (b3 / k3)**2.0 * b3_errors**2.0
                    else:
                        k3_errors = None
                    # y_4_min = np.min(y_4)
                    # y_5_min = np.min(y_5)
                    # y_4_max = np.max(y_4)
                    # y_5_max = np.max(y_5)
                    # ymin = np.min([y_4_min, y_5_min])
                    # ymax = np.max([y_4_max, y_5_max])
                    # value_max = np.max([
                    #     abs(y_4_min),
                    #     abs(y_4_max),
                    #     abs(y_5_min),
                    #     abs(y_5_max),
                    # ])
                    x3 = np.arange(k3.shape[0])
                    axes[1].errorbar(
                        x3 if array is None else array * pixel_scale,
                        y=k3 if array is None else k3/array * 100,
                        #yerr=k3_errors if array is None else (k3/array)*(k3_errors/k3) * 100,
                        linestyle="None",
                        marker="o",
                        color="black",
                        label=r"$k_3$"
                    )
                    axes_angles[1].errorbar(
                        x3 if array is None else array * pixel_scale,
                        y=phi_3,
                        #yerr=,
                        linestyle="None",
                        marker="o",
                        color="black",
                        label=r"$k_3$"
                    )
                    #axes_flattened[4].set_ylim(-value_max, value_max)
                    #axes_flattened[5].set_ylim(-value_max, value_max)
                else:
                    k3 = None
                    k3_errors = None
            else:
                pass
        if np.logical_and(
            "a_4" in keys,
            "b_4" in keys,
        ):
            if True:
                a4 = np.array([
                    parameters["a_4"] for parameters in list_of_parameters
                ])
                b4 = np.array([
                    parameters["b_4"] for parameters in list_of_parameters
                ])
                if any(_ is not None for _ in a4):
                    k4 = np.hypot(a4, b4)
                    phi_4 = continuous_angle_conversion(angle=np.arctan2(b4, a4) + angles)
                    if list_of_parameters_errors is not None:
                        a4_errors = np.array([
                            errors["a_4"] for errors in list_of_parameters_errors
                        ])
                        b4_errors = np.array([
                            errors["b_4"] for errors in list_of_parameters_errors
                        ])
                        k4_errors = (a4 / k4)**2.0 * a4_errors**2.0 + (b4 / k4)**2.0 * b4_errors**2.0
                    else:
                        k4_errors = None
                    # y_6_min = np.min(y_6)
                    # y_7_min = np.min(y_7)
                    # y_6_max = np.max(y_6)
                    # y_7_max = np.max(y_7)
                    # ymin = np.min([y_6_min, y_7_min])
                    # ymax = np.max([y_6_max, y_7_max])
                    # value_max = np.max([
                    #     abs(y_6_min),
                    #     abs(y_6_max),
                    #     abs(y_7_min),
                    #     abs(y_7_max),
                    # ])
                    x4 = np.arange(k4.shape[0])
                    axes[2].errorbar(
                        x4 if array is None else array * pixel_scale,
                        y=k4 if array is None else k4/array * 100,
                        #yerr=k4_errors if array is None else (k4/array)*(k4_errors/k4) * 100,
                        linestyle="None",
                        marker="o",
                        color="black",
                        label=r"$k_4$"
                    )
                    axes_angles[2].errorbar(
                        x4 if array is None else array * pixel_scale,
                        y=phi_4,
                        #yerr=,
                        linestyle="None",
                        marker="o",
                        color="black",
                        label=r"$k_4$"
                    )
                    #axes_flattened[6].set_ylim(-value_max, value_max)
                    #axes_flattened[7].set_ylim(-value_max, value_max)
                else:
                    k4 = None
                    k4_errors = None
            else:
                pass

        # NOTE:
        if ylim is None:
            ylim = (0.0, 2.0)
            #ylim = (2e-3, 2e-1)


        # NOTE:
        for i, ax in enumerate(axes):

            # NOTE:
            ax.legend(loc=2, frameon=True, fontsize=15)

            # NOTE:
            ax.set_ylim(ylim)

            ax.set_xscale("log")
            ax.set_yscale("log")

            # NOTE:
            ax.minorticks_on()
            ax.tick_params(
                axis='y',
                which="major",
                length=6,
                right=True,
                top=True,
                direction="in",
                colors='black'
            )
            ax.tick_params(
                axis='y',
                which="minor",
                length=3,
                right=True,
                top=True,
                direction="in",
                colors='black'
            )
            ax.tick_params(
                axis='x',
                which="major",
                length=6,
                bottom=True,
                top=True,
                direction="in",
                colors='black',
            )
            ax.tick_params(
                axis='x',
                which="minor",
                length=3,
                bottom=True,
                top=True,
                direction="in",
                colors='black',
            )

            # NOTE:
            if i > 0:
                ax.set_yticks([])
            ax.set_xlabel(r"$a$ (arcsec)", fontsize=12.5)
        axes[0].set_ylabel(r"$\rm k_m / a \times 100$ (\%)", fontsize=12.5)

        # NOTE:
        for i, ax in enumerate(axes_angles):

            # NOTE:
            #ax.legend(loc=2, frameon=True, fontsize=15)

            # NOTE:
            ax.set_ylim(0.0, 360.0)
            ax.set_xscale("log")

            # NOTE:
            ax.minorticks_on()
            ax.tick_params(
                axis='y',
                which="major",
                length=6,
                right=True,
                top=True,
                direction="in",
                colors='black'
            )
            ax.tick_params(
                axis='y',
                which="minor",
                length=3,
                right=True,
                top=True,
                direction="in",
                colors='black'
            )
            ax.tick_params(
                axis='x',
                which="major",
                length=6,
                bottom=True,
                top=True,
                direction="in",
                colors='black',
            )
            ax.tick_params(
                axis='x',
                which="minor",
                length=3,
                bottom=True,
                top=True,
                direction="in",
                colors='black',
            )

            # NOTE:
            if i > 0:
                ax.set_yticks([])
            ax.set_xlabel(r"$a$ (arcsec)", fontsize=12.5)



        if value is not None:
            pass
            #axes_k1_angles.axhline(value, 0.4, 0.95, linewidth=3, color="b", alpha=0.75)
            axes_k1_angles.axhline(0.0, linestyle=":", linewidth=2, color="b", alpha=0.75)
        axes_k1_angles.set_ylim(0.0, 360.0)
        axes_k1_angles.set_xscale("log")
        axes_k1_angles.minorticks_on()
        axes_k1_angles.tick_params(
            axis='y',
            which="major",
            length=6,
            right=True,
            top=True,
            direction="in",
            colors='black'
        )
        axes_k1_angles.tick_params(
            axis='y',
            which="minor",
            length=3,
            right=True,
            top=True,
            direction="in",
            colors='black'
        )
        axes_k1_angles.tick_params(
            axis='x',
            which="major",
            length=6,
            bottom=True,
            top=True,
            direction="in",
            colors='black',
        )
        axes_k1_angles.tick_params(
            axis='x',
            which="minor",
            length=3,
            bottom=True,
            top=True,
            direction="in",
            colors='black',
        )
        axes_k1_angles.set_xlabel(r"$R$ (arcsec)", fontsize=15)
        #axes_k1_angles.set_ylabel(r"$\phi_1$ (deg)", fontsize=15)
        axes_k1_angles.set_ylabel(r"$\Delta\phi_1$ (deg)", fontsize=15)
        #axes_k1_angles.set_ylim(0.0, 360.0)
        axes_k1_angles.set_ylim(-180.0, 180.0)
        axes_k1_angles.set_xlim(9.0 * 10**-1, 5.0 * 10**1.0)
        #axes_k1_angles.set_yticks([0.0, 90.0, 180.0, 270.0, 360.0])
        axes_k1_angles.set_yticks([-180.0, -90.0, 0.0, 90.0, 180.0])
        axes_k1_angles.text(0.05, 0.95, name, transform=axes_k1_angles.transAxes, ha='left', va='top', weight="bold", fontsize=15, color='navy')

        figure.subplots_adjust(wspace=0.0, left=0.05, right=0.95)
        figure_angles.subplots_adjust(wspace=0.0, left=0.05, right=0.95)
        figure_k1_angles.subplots_adjust(wspace=0.0, right=0.975, top=0.975)

        return (
            k1,
            k1_errors,
            k3,
            k3_errors,
            k4,
            k4_errors,
        )





def plot_multipole_amplitudes_from_list_of_parameters_lite(
    array,
    list_of_parameters,
    list_of_parameters_errors,
    ylim=None,
    pixel_scale=1.0,
    name=None
):
    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15, 5)
    )
    figure_angles, axes_angles = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15, 5)
    )

    # NOTE:
    angles = np.array([
        parameters["angle"] for parameters in list_of_parameters
    ])

    # NOTE:
    keys = list(list_of_parameters[0].keys())

    # NOTE:
    if True:
        if np.logical_and(
            "a_1" in keys,
            "b_1" in keys,
        ):

            if np.logical_or(
                all(parameters["a_1"] is None for parameters in list_of_parameters),
                all(parameters["b_1"] is None for parameters in list_of_parameters)
            ):
                k1 = None
                k1_errors = None
            else:
                a1 = np.array([
                    parameters["a_1"] for parameters in list_of_parameters
                ])
                b1 = np.array([
                    parameters["b_1"] for parameters in list_of_parameters
                ])
                k1 = np.hypot(a1, b1)

                phi_1 = continuous_angle_conversion(
                    angle=np.arctan2(b1, a1) + angles
                )

                # NOTE:
                if list_of_parameters_errors is not None:
                    a1_errors = np.array([
                        errors["a_1"] for errors in list_of_parameters_errors
                    ])
                    b1_errors = np.array([
                        errors["b_1"] for errors in list_of_parameters_errors
                    ])
                    k1_errors = (a1 / k1)**2.0 * a1_errors**2.0 + (b1 / k1)**2.0 * b1_errors**2.0
                    if array is not None:
                        k1_errors = (k1 / array)*(k1_errors / k1) * 100
                else:
                    k1_errors = None


                x1 = np.arange(k1.shape[0])
                axes[0].errorbar(
                    x1 if array is None else array * pixel_scale,
                    y=k1 if array is None else k1/array * 100,
                    yerr=k1_errors,
                    linestyle="None",
                    marker="o",
                    color="black",
                    label=r"$k_1$"
                )
                axes_angles[0].errorbar(
                    x1 if array is None else array * pixel_scale,
                    y=phi_1,
                    linestyle="None",
                    marker="o",
                    color="black",
                    label=r"$k_1$"
                )

        if np.logical_and(
            "a_3" in keys,
            "b_3" in keys,
        ):

            if np.logical_or(
                all(parameters["a_3"] is None for parameters in list_of_parameters),
                all(parameters["b_3"] is None for parameters in list_of_parameters)
            ):
                k3 = None
                k3_errors = None
            else:
                a3 = np.array([
                    parameters["a_3"] for parameters in list_of_parameters
                ])
                b3 = np.array([
                    parameters["b_3"] for parameters in list_of_parameters
                ])
                k3 = np.hypot(a3, b3)

                phi_3 = continuous_angle_conversion(
                    angle=np.arctan2(b3, a3) + angles
                )

                # NOTE:
                if list_of_parameters_errors is not None:
                    a3_errors = np.array([
                        errors["a_3"] for errors in list_of_parameters_errors
                    ])
                    b3_errors = np.array([
                        errors["b_3"] for errors in list_of_parameters_errors
                    ])
                    k3_errors = (a3 / k3)**2.0 * a3_errors**2.0 + (b3 / k3)**2.0 * b3_errors**2.0
                    if array is not None:
                        k3_errors = (k3 / array)*(k3_errors / k3) * 100
                else:
                    k3_errors = None

                x3 = np.arange(k3.shape[0])
                axes[1].errorbar(
                    x3 if array is None else array * pixel_scale,
                    y=k3 if array is None else k3/array * 100,
                    yerr=k3_errors,
                    linestyle="None",
                    marker="o",
                    color="black",
                    label=r"$k_3$"
                )
                axes_angles[1].errorbar(
                    x3 if array is None else array * pixel_scale,
                    y=phi_3,
                    linestyle="None",
                    marker="o",
                    color="black",
                    label=r"$k_3$"
                )


        if np.logical_and(
            "a_4" in keys,
            "b_4" in keys,
        ):
            if np.logical_or(
                all(parameters["a_4"] is None for parameters in list_of_parameters),
                all(parameters["b_4"] is None for parameters in list_of_parameters)
            ):
                k4 = None
                k4_errors = None
            else:
                a4 = np.array([
                    parameters["a_4"] for parameters in list_of_parameters
                ])
                b4 = np.array([
                    parameters["b_4"] for parameters in list_of_parameters
                ])
                k4 = np.hypot(a4, b4)

                phi_4 = continuous_angle_conversion(
                    angle=np.arctan2(b4, a4) + angles
                )

                # NOTE:
                if list_of_parameters_errors is not None:
                    a4_errors = np.array([
                        errors["a_4"] for errors in list_of_parameters_errors
                    ])
                    b4_errors = np.array([
                        errors["b_4"] for errors in list_of_parameters_errors
                    ])
                    k4_errors = (a4 / k4)**2.0 * a4_errors**2.0 + (b4 / k4)**2.0 * b4_errors**2.0
                    if array is not None:
                        k4_errors = (k4 / array)*(k4_errors / k4) * 100
                else:
                    k4_errors = None


                x4 = np.arange(k4.shape[0])
                axes[2].errorbar(
                    x4 if array is None else array * pixel_scale,
                    y=k4 if array is None else k4/array * 100,
                    yerr=k4_errors,
                    linestyle="None",
                    marker="o",
                    color="black",
                    label=r"$k_4$"
                )
                axes_angles[2].errorbar(
                    x4 if array is None else array * pixel_scale,
                    y=phi_4,
                    linestyle="None",
                    marker="o",
                    color="black",
                    label=r"$k_4$"
                )

        # NOTE:
        if ylim is None:
            pass
            #ylim = (0.0, 2.0)
            #ylim = (2e-3, 2e-1)


        # NOTE:
        for i, ax in enumerate(axes):

            if i == 0:
                if k1 is None:
                    condition = False
                else:
                    condition = True
            if i == 1:
                if k3 is None:
                    condition = False
                else:
                    condition = True
            if i == 2:
                if k4 is None:
                    condition = False
                else:
                    condition = True

            # NOTE:
            if condition:
                ax.legend(loc=2, frameon=True, fontsize=15)

                # NOTE:
                ax.set_ylim(ylim)
                ax.set_xscale("log")
                #ax.set_yscale("log")

                # NOTE:
                ax.minorticks_on()
                ax.tick_params(
                    axis='y',
                    which="major",
                    length=6,
                    right=True,
                    top=True,
                    direction="in",
                    colors='black'
                )
                ax.tick_params(
                    axis='y',
                    which="minor",
                    length=3,
                    right=True,
                    top=True,
                    direction="in",
                    colors='black'
                )
                ax.tick_params(
                    axis='x',
                    which="major",
                    length=6,
                    bottom=True,
                    top=True,
                    direction="in",
                    colors='black',
                )
                ax.tick_params(
                    axis='x',
                    which="minor",
                    length=3,
                    bottom=True,
                    top=True,
                    direction="in",
                    colors='black',
                )

                # NOTE:
                if i > 0:
                    ax.set_yticks([])
                ax.set_xlabel(r"$a$ (arcsec)", fontsize=12.5)
        axes[0].set_ylabel(r"$\rm k_m / a \times 100$ (\%)", fontsize=12.5)

        # NOTE:
        for i, ax in enumerate(axes_angles):

            # NOTE:
            #ax.legend(loc=2, frameon=True, fontsize=15)

            # NOTE:
            ax.set_ylim(0.0, 360.0)
            ax.set_xscale("log")

            # NOTE:
            ax.minorticks_on()
            ax.tick_params(
                axis='y',
                which="major",
                length=6,
                right=True,
                top=True,
                direction="in",
                colors='black'
            )
            ax.tick_params(
                axis='y',
                which="minor",
                length=3,
                right=True,
                top=True,
                direction="in",
                colors='black'
            )
            ax.tick_params(
                axis='x',
                which="major",
                length=6,
                bottom=True,
                top=True,
                direction="in",
                colors='black',
            )
            ax.tick_params(
                axis='x',
                which="minor",
                length=3,
                bottom=True,
                top=True,
                direction="in",
                colors='black',
            )

            # NOTE:
            if i > 0:
                ax.set_yticks([])
            ax.set_xlabel(r"$a$ (arcsec)", fontsize=12.5)





        figure.subplots_adjust(
            wspace=0.0,
            left=0.05,
            right=0.95
        )
        figure_angles.subplots_adjust(
            wspace=0.0,
            left=0.05,
            right=0.95
        )


        return (
            k1,
            k1_errors,
            k3,
            k3_errors,
            k4,
            k4_errors,
        )

def visualize(
    array,
    sample,
    list_of_parameters,
    a_max=None,
    extract_condition=False,
    vmin=None,
    vmax=None,
):

    # NOTE:
    logscale = False

    # NOTE:
    figure, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(16, 8)
    )

    # NOTE:
    image = copy.copy(sample.image)
    if sample.mask is not None:
        image[sample.mask.astype(bool)] = np.nan

        image_temp = copy.copy(sample.image)
        image_temp[~sample.mask.astype(bool)] = np.nan
    else:
        image_temp = None

    # NOTE:
    def custom_colormap():
        # Define the colors
        # colors = ['white', 'black', 'red']
        # positions = [0.0, 0.5, 1.0]
        colors = ['grey', 'white', 'red', 'darkred', 'black']
        positions = [0.0, 0.25, 0.5, 0.75,1.0]

        # Create the colormap
        cmap = LinearSegmentedColormap.from_list('custom_colormap', list(zip(positions, colors)))

        return cmap

    # ========== #
    # NOTE: TEST - DELETE
    # ========== #
    """
    angles = np.linspace(0.0, 2.0 * np.pi, 100)[:-1]
    figure_temp, axes_temp = plt.subplots(figsize=(10, 8))
    axes_temp.imshow(
        np.log10(image) if logscale else image,
        cmap="jet",
        aspect="auto",
        vmin=0.0,
        vmax=3.0,
    )
    angles_temp = np.linspace(0.0, 360.0, len(array))
    for i, (a, parameters) in enumerate(zip(array, list_of_parameters)):
        #print(parameters)

        # y_fit, y_errors_fit, (x, y), angles = sample.extract(
        #     a=a, parameters=parameters, condition=extract_condition
        # )

        parameters_temp = {
            "x0":parameters["x0"],
            "y0":parameters["y0"],
            "ellipticity":0.8,
            "angle":angles_temp[i] * units.deg.to(units.rad),
            "a_1":5,
            "b_1":0.0,
            "a_3":None,
            "b_3":None,
            "a_4":None,
            "b_4":None,
        }
        print("angle =", angles_temp[i], "(deg)")
        x_temp, y_temp = func(a=a, parameters=parameters_temp, angles=angles)
        axes_temp.plot(
            x_temp,
            y_temp,
            marker="o",
            markersize=2.5,
            color="w"
        )
    axes_temp.set_xlim(0, image.shape[1])
    axes_temp.set_ylim(0, image.shape[0])
    #plt.show()
    #exit()
    """
    # ========== #
    # END
    # ========== #
    if vmin is None:
        vmin = 0.025
    if vmax is None:
        vmax = 5.0
    norm = matplotlib.colors.LogNorm(
        vmin=vmin, vmax=vmax
    )

    # plt.figure()
    # plt.imshow(np.log10(image))
    # plt.show()
    # exit()
    im = axes[0].imshow(
        np.log10(image) if logscale else image,
        cmap="jet",
        aspect="auto",
        norm=norm,
        # vmin=-0.8,
        # vmax=1.0,
        # # NOTE: __main__
        # norm=matplotlib.colors.LogNorm(
        #     vmin=0.05, vmax=10.0
        # )
        # # NOTE: NGC 0057; F110W
        # norm=matplotlib.colors.LogNorm(
        #     vmin=2.5, vmax=500.0
        # )
        # NOTE: NGC 0057; F475X
        # norm=matplotlib.colors.LogNorm(
        #     vmin=0.25, vmax=5.0
        # )
        # # NOTE: NGC 0315; F110W
        # norm=matplotlib.colors.LogNorm(
        #     vmin=2.5, vmax=500.0
        # )
        # # NOTE: SPT2147; F200W
        # norm=matplotlib.colors.LogNorm(
        #     vmin=0.2, vmax=5.0
        # )
        # # NOTE: SLACS
        # norm=matplotlib.colors.LogNorm(
        #     vmin=0.05, vmax=5.0
        # )
    )
    if image_temp is not None:
        axes[0].imshow(
            np.log10(image_temp) if logscale else image_temp,
            cmap="jet",
            aspect="auto",
            alpha=0.5,
        )
    list_of_angles = []
    list_of_y_fit = []
    list_of_y_errors_fit = []
    y_means = []
    y_stds = []
    residuals = []
    chi_squares = []
    for i, (a, parameters) in enumerate(zip(array, list_of_parameters)):
        if parameters is not None:
            if i == 0:
                m = sum(1 for value in parameters.values() if value is not None)
            y_fit, y_errors_fit, (x, y), angles = sample.extract(
                a=a, parameters=parameters, condition=extract_condition
            )
            list_of_angles.append(angles)
            list_of_y_fit.append(y_fit)
            list_of_y_errors_fit.append(y_errors_fit)

            if y_errors_fit is None:
                #y_errors_fit = 0.05 * y_fit
                raise NotImplementedError()

            # NOTE:
            y_mean = np.nanmean(y_fit)
            y_means.append(y_mean)

            # NOTE:
            y_std = np.nanstd(y_fit)
            y_stds.append(y_std)

            residuals.append(
                (y_fit - y_mean) / y_errors_fit
            )
            chi_squares.append(
                (y_fit - y_mean)**2.0 / y_errors_fit**2.0
            )
            #print(y_fit - y_mean, y_errors_fit)
            #print((y_fit - y_mean) / y_errors_fit);exit()

            # NOTE:
            axes[0].plot(
                x,
                y,
                marker="o",
                markersize=2.5,
                color="w"
            )

            # NOTE:
            axes[1].errorbar(
                angles * units.rad.to(units.deg),
                y_fit,
                yerr=y_errors_fit,
                linestyle="None",
                marker="o",
                markersize=2.5,
                color="black"
            )
            axes[1].axhline(
                y_mean,
                linestyle=":",
                color="black"
            )

            # phi_1 = continuous_angle_conversion(
            #     angle=np.arctan2(parameters["b_1"], parameters["a_1"]) + parameters["angle"]
            # )
            # print(phi_1)
            # axes[0].arrow(
            #     parameters["x0"],
            #     parameters["y0"],
            #     a * np.cos(phi_1 * units.deg.to(units.rad)),
            #     a * np.sin(phi_1 * units.deg.to(units.rad)),
            #     color="gray",
            #     head_width=2.0
            # )

    levels = np.sort(np.log10(y_means)) if logscale else np.sort(y_means)
    axes[0].contour(
        np.log10(image) if logscale else image,
        #levels=y_means[::-1],
        levels=levels,
        colors="black"
    )
    colors = [im.cmap(im.norm(level)) for level in levels][::-1]

    for i, (angles, y_fit, y_errors_fit) in enumerate(zip(list_of_angles, list_of_y_fit, list_of_y_errors_fit)):
        axes[1].errorbar(
            angles * units.rad.to(units.deg),
            y_fit,
            yerr=y_errors_fit,
            linestyle="None",
            marker="o",
            markersize=2.5,
            color=colors[i]
        )
    # axes[0].plot(
    #     [247],
    #     [250],
    #     marker="o"
    # )
    xticks = np.linspace(0, image.shape[1], 11)
    yticks = np.linspace(0, image.shape[0], 11)
    axes[0].set_xticks(xticks)
    axes[0].set_yticks(xticks)
    axes[1].set_xticks([0, 90, 180, 270, 360])
    axes[1].set_xlabel(r"$\phi$ (deg)", fontsize=15)
    axes[1].set_ylabel(r"$\rm I(\phi)$ [E/s]", fontsize=15)
    axes[1].tick_params(axis='y', labelsize=12.5)
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")
    for i, ax in enumerate(axes):
        ax.minorticks_on()
        ax.tick_params(
            axis='both',
            which="major",
            length=6,
            right=True,
            top=True,
            direction="in",
            colors='w' if i==0 else "black"
        )
        ax.tick_params(
            axis='both',
            which="minor",
            length=3,
            right=True,
            top=True,
            direction="in",
            colors='w' if i==0 else "black"
        )

    axes[1].set_yscale("log")

    # text = axes[0].text(
    #     0.05,
    #     0.95,
    #     "model 1",
    #     horizontalalignment='left',
    #     verticalalignment='center',
    #     transform=axes[0].transAxes,
    #     fontsize=25,
    #     weight="bold",
    #     color="w"
    # )
    # text.set_path_effects([patheffects.withStroke(linewidth=3, foreground='black')])

    # text = axes[0].text(
    #     0.7,
    #     0.95,
    #     "NGC 2274",
    #     horizontalalignment='left',
    #     verticalalignment='center',
    #     transform=axes[0].transAxes,
    #     fontsize=25,
    #     weight="bold",
    #     color="w"
    # )
    # text.set_path_effects([patheffects.withStroke(linewidth=3, foreground='black')])


    if a_max is None:
        axes[0].set_xlim(0, image.shape[1])
        axes[0].set_ylim(0, image.shape[0])
    else:
        x0 = list_of_parameters[0]["x0"]
        y0 = list_of_parameters[0]["y0"]
        axes[0].set_xlim(x0 - a_max, x0 + a_max)
        axes[0].set_ylim(y0 - a_max, y0 + a_max)

    # NOTE:
    figure.subplots_adjust(
        left=0.05, right=0.95, bottom=0.075, top=0.95, wspace=0.0
    )

    # # NOTE:
    # plt.figure()
    # colors = matplotlib_utils.colors_from(n=len(chi_squares))
    # x_init = 0.0
    # for i, (y_fit_i, y_errors_fit_i) in enumerate(zip(list_of_y_fit, list_of_y_errors_fit)):
    #     x = np.arange(len(y_fit_i))
    #     plt.plot(x + x_init, y_fit_i / y_errors_fit_i, linestyle="None", marker="o", color=colors[i])
    #     x_init += len(y_fit_i)
    # plt.yscale("log")

    # # NOTE:
    # plt.figure()
    # colors = matplotlib_utils.colors_from(n=len(residuals))
    # x_init = 0.0
    # for i, residuals_i in enumerate(residuals):
    #     x = np.arange(len(residuals_i))
    #     plt.plot(x + x_init, residuals_i, linestyle="None", marker="o", color=colors[i])
    #     x_init += len(residuals_i)
    # #plt.xscale("log")
    # plt.yscale("log")

    # # NOTE:
    # plt.figure()
    # colors = matplotlib_utils.colors_from(
    #     n=len(chi_squares)
    # )
    # x_init = 0.0
    # for i, (a, chi_squares_i) in enumerate(
    #     zip(array, chi_squares)
    # ):
    #     x = np.arange(len(chi_squares_i))
    #     plt.plot(x + x_init, chi_squares_i, linestyle="None", marker="o", color=colors[i])
    #     x_init += len(chi_squares_i)
    # #plt.xscale("log")
    # plt.yscale("log")

    y = []
    for i, (a, chi_squares_i) in enumerate(
        zip(array, chi_squares)
    ):
        n = len(chi_squares_i) - 1

        # NOTE:
        y_i = np.nansum(chi_squares_i[:-1]) / (n - m)
        y.append(y_i)

    figure_stats, axes_stats = plt.subplots()
    axes_stats.plot(
        array,
        y,
        linestyle="None",
        marker="o",
        color="black"
    )
    axes_stats.set_xscale("log")
    axes_stats.set_yscale("log")
    # directory = "./MASSIVE/metadata"
    # filename = "{}/xy_model_default.numpy".format(directory)
    # with open(filename, 'wb') as f:
    #     np.save(f, [x, y])
    # plt.show()
    # exit()


    # NOTE:
    #chi_squares_flattened = list(itertools.chain(*chi_squares))
    #plt.hist(chi_squares_flattened, bins=100, alpha=0.75)

    #plt.show();exit()

    return figure, axes, figure_stats, axes_stats



if __name__ == "__main__":

    pass
