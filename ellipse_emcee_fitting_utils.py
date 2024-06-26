import os, sys, copy
import numpy as np
import matplotlib.pyplot as plt
import corner as corner

from astropy import (
    units,
    constants
)
from astropy.io import fits
from astropy import stats
from scipy import interpolate


# ---------------------------------------------------------------------------- #

# NOTE:
path = os.environ["GitHub"] + "/utils"
sys.path.append(path)

#import matplotlib_utils as matplotlib_utils
import emcee_wrapper as emcee_wrapper
import emcee_plot_utils as emcee_plot_utils

# ---------------------------------------------------------------------------- #


class Dataset:

    def __init__(
        self,
        image,
        error,
        a,
        mappings=None,
        centre=None,
        extract_condition=True
    ):

        # NOTE:
        self.image = image
        if error is None:
            self.error = None
        else:
            self.error = error

        # NOTE:
        self.sample = Sample(
            image=self.image, error=self.error
        )

        # NOTE:
        self.a = a

        # NOTE:
        self.mappings = mappings

        # NOTE:
        self.centre = centre

        # NOTE:
        self.extract_condition = extract_condition



def func(
    a,
    parameters,
    angles
):

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
        image,
        error=None
    ):

        # NOTE: 2D
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
        #self.angles = np.linspace(0.0, 2.0 * np.pi, 100)

    def extract(
        self,
        a,
        parameters,
        condition=True,
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
        )

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

        # NOTE:
        if self.error_interp is not None:
            points_errors_interp = self.error_interp(points)
        else:
            points_errors_interp = None

        # NOTE:
        if condition:
            mean, _, std = stats.sigma_clipped_stats(
                points_interp, sigma=3.0
            )
            idx = points_interp > mean + 3.0 * std
            points_interp[idx] = np.nan


        return (
            points_interp,
            points_errors_interp,
            (x, y),
            angles,
        )



# NOTE:
def residuals_from(
    theta,
    dataset
):

    # NOTE:
    parameters = {
        "x0":None,
        "y0":None,
        "angle":None,
        "ellipticity":None,
        "a_1":None,
        "b_1":None,
        "a_3":None,
        "b_3":None,
        "a_4":None,
        "b_4":None,
    }
    for key, index in dataset.mappings.items():
        parameters[key] = theta[index]


    # NOTE: FIXED CENTRE
    if parameters["x0"] is None:
        if dataset.centre is None:
            raise NotImplementedError()
        else:
            parameters["x0"] = dataset.centre[0]
    if parameters["y0"] is None:
        if dataset.centre is None:
            raise NotImplementedError()
        else:
            parameters["y0"] = dataset.centre[1]

    #print("(x, y) =", (parameters["x0"], parameters["y0"]), "|", dataset.centre)

    # NOTE:
    y_fit, y_errors_fit, _, _ = dataset.sample.extract(
        a=dataset.a, parameters=parameters, condition=dataset.extract_condition
    )

    # NOTE:
    if y_errors_fit is None:
        raise NotImplementedError()

    # NOTE:
    residuals = (y_fit - np.nanmean(y_fit)) / y_errors_fit

    # NOTE:
    idx = np.logical_or(np.isnan(residuals), np.isinf(residuals))
    residuals[idx] = 0.0

    return residuals





def log_likelihood_func(
    dataset,
    theta
):

    # NOTE:
    if True:
        residuals = residuals_from(
            theta=theta, dataset=dataset,
        )

        # NOTE:
        figure_of_merit = -np.sum(np.square(residuals))
        print("figure_of_merit =", figure_of_merit)

        # NOTE:
        if np.logical_or.reduce((
            np.isnan(figure_of_merit), np.isinf(figure_of_merit), abs(figure_of_merit) == 0.0
        )):
            figure_of_merit = -1e99
    else:
        figure_of_merit = -1e99

    return figure_of_merit

# NOTE:
def fit_emcee(
    dataset,
    p0=None,
    limits=None,
    backend_filename="backend.h5",
    nwalkers=100,
    nsteps=1000,
    debug=False,
):

    # NOTE:
    search = emcee_wrapper.emcee_wrapper(
        dataset=dataset,
        log_likelihood_func=log_likelihood_func,
        limits=limits,
        p0=p0,
        nwalkers=nwalkers,
        backend_filename=backend_filename,
        parallization=False,
        eps=1e-3,
    )

    # NOTE:
    sampler = search.run(nsteps=nsteps, parallel=False)

    # NOTE: visualization
    visualize = False
    if visualize:
        import emcee_plot_utils as emcee_plot_utils
        import corner as corner

        discard = int(nsteps / 2.0)
        thin = 10
        chain = sampler.get_chain(
            discard=discard,
            thin=thin,
            flat=False
        )
        chain_flattened = sampler.get_chain(
            discard=discard, thin=thin, flat=True
        )
        def sanitize(chain):
            chain_temp = copy.copy(chain_flattened)
            if len(chain_flattened.shape) == 2:
                for i in range(chain_flattened.shape[1]):
                    mean, _, std = stats.sigma_clipped_stats(
                        chain_flattened[:, i], sigma=3.0
                    )
                    idx = np.logical_or(
                        chain_flattened[:, i] > mean + 5.0 * std,
                        chain_flattened[:, i] < mean - 5.0 * std,
                    )
                    chain_temp[idx, i] = np.nan
            else:
                raise ValueError()


            return chain_temp[~np.any(np.isnan(chain_temp), axis=1)]
        chain_flattened = sanitize(chain_flattened)

        emcee_plot_utils.plot_chain(
            chain=chain,
            ncols=2,
            figsize=(20, 10),
            truths=None
        )
        fig = corner.corner(
            chain_flattened#, labels=labels
        )
        plt.show()
        exit()

    return sampler

def sampler_subplots(sampler, discard=500, thin=10):

    chain = sampler.get_chain(
        discard=discard,
        thin=thin,
        flat=False
    )
    chain_flattened = sampler.get_chain(
        discard=discard, thin=thin, flat=True
    )
    def sanitize(chain):
        chain_temp = copy.copy(chain_flattened)
        if len(chain_flattened.shape) == 2:
            for i in range(chain_flattened.shape[1]):
                mean, _, std = stats.sigma_clipped_stats(
                    chain_flattened[:, i], sigma=3.0
                )
                idx = np.logical_or(
                    chain_flattened[:, i] > mean + 5.0 * std,
                    chain_flattened[:, i] < mean - 5.0 * std,
                )
                chain_temp[idx, i] = np.nan
        else:
            raise ValueError()

        return chain_temp[
            ~np.any(np.isnan(chain_temp), axis=1)
        ]

    # NOTE:
    chain_flattened = sanitize(chain_flattened)

    # NOTE:
    figure, axes = emcee_plot_utils.plot_chain(
        chain=chain,
        ncols=2,
        figsize=(20, 10),
        truths=None
    )
    # plt.show()
    # fig = corner.corner(
    #     chain_flattened#, labels=labels
    # )
    # plt.show()
    # exit()

    return figure, axes

def get_keys(
    fixed_centre=True,
    harmonics=None # NOTE: THIS IS NOT CURRENTLY BEING USED
):

    if fixed_centre:
        keys = [
            "angle",
            "ellipticity",

        ]
    else:
        keys = [
            "x0",
            "y0",
            "angle",
            "ellipticity",
        ]

    # NOTE:
    # keys_harmonics = [
    #     "a_1",
    #     "b_1",
    #     "a_3",
    #     "b_3",
    #     "a_4",
    #     "b_4",
    # ]
    if harmonics is None:
        pass
    else:
        keys_harmonics = []
        if "m1" in harmonics:
            keys_harmonics.append("a_1")
            keys_harmonics.append("b_1")
        if "m3" in harmonics:
            keys_harmonics.append("a_3")
            keys_harmonics.append("b_3")
        if "m4" in harmonics:
            keys_harmonics.append("a_4")
            keys_harmonics.append("b_4")
        keys.extend(keys_harmonics)

    return keys

def get_mappings(fixed_centre=True, harmonics=None):

    keys = get_keys(fixed_centre=fixed_centre, harmonics=harmonics)

    # NOTE:
    mappings = {}
    for i, key in enumerate(keys):
        mappings[key] = i

    return mappings, keys

def get_limits(
    fixed_centre=True,
    centre=None,
    harmonics=None,
    dx=5.0,
    dy=5.0,
    dx_min=5.0,
    dy_min=5.0,
):
    """
    if fixed_centre:
        limits = np.array([
            #[0.0, np.pi],
            #[-np.pi, 0.0],
            [-np.pi / 2.0, np.pi / 2.0],
            [0.0, 1.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
        ])
    else:
        limits = np.array([
            [centre[0] - 5.0, centre[0] + 5.0],
            [centre[1] - 5.0, centre[1] + 5.0],
            #[0.0, np.pi],
            #[-np.pi, 0.0],
            [-np.pi / 2.0, np.pi / 2.0],
            [0.0, 1.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
        ])
    """

    # NOTE: Try this prior range for:
    #v = 10.0 # NGC2274
    #v = 15.0 # NGC0741, NGC6482
    v = 25.0 # NGC2274_version_2
    if fixed_centre:
        limits = np.array([
            #[0.0, np.pi],
            #[-np.pi, 0.0],
            [-np.pi / 2.0, np.pi / 2.0],
            [0.0, 1.0],
        ])
        if harmonics is None:
            pass
        else:
            if "m1" in harmonics:
                limits = np.vstack((
                    limits,
                    np.array([
                        [-v, v],
                        [-v, v],
                    ])
                ))
            if "m3" in harmonics:
                limits = np.vstack((
                    limits,
                    np.array([
                        [-v, v],
                        [-v, v],
                    ])
                ))
            if "m4" in harmonics:
                limits = np.vstack((
                    limits,
                    np.array([
                        [-v, v],
                        [-v, v],
                    ])
                ))
            # limits = np.array([
            #     #[0.0, np.pi],
            #     #[-np.pi, 0.0],
            #     [-np.pi / 2.0, np.pi / 2.0],
            #     [0.0, 1.0],
            #     [-v, v],
            #     [-v, v],
            #     [-v, v],
            #     [-v, v],
            #     [-v, v],
            #     [-v, v],
            # ])
    else:

        if dx is None:
            dx = dx_min
        elif dx < dx_min:
            dx = dx_min
        else:
            pass

        if dy is None:
            dy = dx_min
        elif dy < dy_min:
            dy = dy_min
        else:
            pass

        limits = np.array([
            [centre[0] - dx, centre[0] + dx],
            [centre[1] - dy, centre[1] + dy],
            #[0.0, np.pi],
            #[-np.pi, 0.0],
            [-np.pi / 2.0, np.pi / 2.0],
            [0.0, 1.0],
        ])
        if harmonics is None:
            pass
        else:
            if "m1" in harmonics:
                limits = np.vstack((
                    limits,
                    np.array([
                        [-v, v],
                        [-v, v],
                    ])
                ))
            if "m3" in harmonics:
                limits = np.vstack((
                    limits,
                    np.array([
                        [-v, v],
                        [-v, v],
                    ])
                ))
            if "m4" in harmonics:
                limits = np.vstack((
                    limits,
                    np.array([
                        [-v, v],
                        [-v, v],
                    ])
                ))
            # limits = np.array([
            #     [centre[0] - 5.0, centre[0] + 5.0],
            #     [centre[1] - 5.0, centre[1] + 5.0],
            #     #[0.0, np.pi],
            #     #[-np.pi, 0.0],
            #     [-np.pi / 2.0, np.pi / 2.0],
            #     [0.0, 1.0],
            #     [-v, v],
            #     [-v, v],
            #     [-v, v],
            #     [-v, v],
            #     [-v, v],
            #     [-v, v],
            # ])

    return limits


if __name__ == "__main__":

    keys = [
        "x0",
        "y0",
        "angle",
        "ellipticity",
        "a_1",
        "b_1",
        "a_3",
        "b_3",
        "a_4",
        "b_4",
    ]
    mapping = {}
    for i, key in enumerate(keys):
        mapping[key] = i
