import os, sys
import numpy as np
import matplotlib.pyplot as plt

# NOTE: astropy
from astropy import (
    units,
    constants,
)
from astropy.io import fits

# NOTE:
from matplotlib.colors import (
    LogNorm,
    TwoSlopeNorm,
)

# NOTE:
import ellipse_fitting_utils as ellipse_fitting_utils
import ellipse_emcee_fitting_utils as ellipse_emcee_fitting_utils
import emcee_wrapper as emcee_wrapper

if __name__ == "__main__":

    base_directory = "/Volumes/MyPassport_red/slacs_james/"

    # NOTE:
    pixel_scale = 0.05

    # NOTE:
    #name = "slacs0252+0039"
    #name = "slacs0946+1006"
    name = "slacs1023+4230"
    #name = "slacs1143-0144"
    #name = "slacs1205+4910"
    #name = "slacs1525+3327"
    #name = "slacs2303+1422"

    # NOTE: This is an initial guess for lmfit (only for x0, y0, angle, ellipticity). It does not need to be very accurate, expect for the centre.
    if name is None:
        raise NotImplementedError()
    elif name == "slacs0252+0039":
        parameters_0 = {
            "x0":140.0,
            "y0":140.0,
            "angle":0.0,
            "ellipticity":0.2,
            "a_1":None,
            "b_1":None,
            "a_3":None,
            "b_3":None,
            "a_4":None,
            "b_4":None,
        }
    elif name == "slacs0946+1006":
        parameters_0 = {
            "x0":140.0,
            "y0":140.0,
            "angle":0.8,
            "ellipticity":0.4,
            "a_1":None,
            "b_1":None,
            "a_3":None,
            "b_3":None,
            "a_4":None,
            "b_4":None,
        }
    elif name == "slacs1023+4230":
        parameters_0 = {
            "x0":140.0,
            "y0":140.0,
            "angle":0.0,
            "ellipticity":0.5,
            "a_1":None,
            "b_1":None,
            "a_3":None,
            "b_3":None,
            "a_4":None,
            "b_4":None,
        }
    elif name == "slacs1143-0144":
        parameters_0 = {
            "x0":140.0,
            "y0":140.0,
            "angle":-0.8,
            "ellipticity":0.6,
            "a_1":None,
            "b_1":None,
            "a_3":None,
            "b_3":None,
            "a_4":None,
            "b_4":None,
        }
    elif name == "slacs1205+4910":
        parameters_0 = {
            "x0":140.0,
            "y0":140.0,
            "angle":-0.4,
            "ellipticity":0.5,
            "a_1":None,
            "b_1":None,
            "a_3":None,
            "b_3":None,
            "a_4":None,
            "b_4":None,
        }
    elif name == "slacs1525+3327":
        parameters_0 = {
            "x0":140.0,
            "y0":140.0,
            "angle":-0.8,
            "ellipticity":0.6,
            "a_1":None,
            "b_1":None,
            "a_3":None,
            "b_3":None,
            "a_4":None,
            "b_4":None,
        }
    elif name == "slacs2303+1422":
        raise NotImplementedError()
        parameters_0 = {
            "x0":140.0,
            "y0":140.0,
            "angle":-0.8,
            "ellipticity":0.6,
            "a_1":None,
            "b_1":None,
            "a_3":None,
            "b_3":None,
            "a_4":None,
            "b_4":None,
        }

    # NOTE: The range of radii where we apply the ellipse fitting (in units of pixels)
    if name == "slacs0252+0039":
        a_min = 5
        a_max = 50
        a_n = 10
    elif name == "slacs0946+1006":
        a_min = 5
        a_max = 80
        a_n = 10
    elif name == "slacs1023+4230":
        a_min = 5
        a_max = 100
        a_n = 15
    elif name == "slacs1143-0144":
        a_min = 5
        a_max = 100
        a_n = 15
    elif name == "slacs1205+4910":
        a_min = 5
        a_max = 100
        a_n = 15
    elif name == "slacs1525+3327":
        a_min = 5
        a_max = 80
        a_n = 10
    elif name == "slacs2303+1422":
        a_min = 5
        a_max = 100
        a_n = 15


    # NOTE:
    directory = "{}/{}".format(base_directory, name)

    # NOTE: Load data
    image = fits.getdata(
        filename="{}/data.fits".format(directory)
    )
    source_light_model_image = fits.getdata(
        filename="{}/model_source_light.fits".format(directory)
    )
    error = fits.getdata(
        filename="{}/noise_map.fits".format(directory)
    )

    # NOTE:
    image_source_light_subtracted = image - source_light_model_image
    vmin = np.nanmin(image_source_light_subtracted)
    vmax = np.nanmax(image_source_light_subtracted)

    # plt.figure()
    # plt.imshow(
    #     image_source_light_subtracted,
    #     cmap="jet",
    #     norm=TwoSlopeNorm(vcenter=vmax / 50.0, vmin=vmin, vmax=vmax)
    # )
    # levels = np.logspace(
    #     np.log10(vmin * 5.0),
    #     np.log10(vmax / 5.0),
    #     15
    # )
    # plt.contour(
    #     image_source_light_subtracted,
    #     levels=levels,
    #     colors="black"
    # )
    # plt.show()

    # ========== #

    # NOTE: (Bad coding) Which parameters to optimize with lmfit
    fixed_centre = True
    if fixed_centre:
        keys = [
            "angle",
            "ellipticity",
            "a_1",
            "b_1",
            "a_3",
            "b_3",
            "a_4",
            "b_4",
        ]
    else:
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
    mappings = {}
    for i, key in enumerate(keys):
        mappings[key] = i

    # NOTE: Initilize the main IsophoteFitter object
    obj = ellipse_fitting_utils.main(
        image=image_source_light_subtracted,
        error=error,
        mask=None,
        a_min=a_min,
        a_max=a_max,
        a_n=a_n,
        parameters_0=parameters_0,
        extract_condition=True # NOTE: Turn-on automasking
    )

    # NOTE: Perform ellipse fitting with lmfit. This gives an initial set of parameters for each ellipse which we can later feed to emcee.
    harmonics = []
    #harmonics = ["m3", "m4"]
    #harmonics = ["m1", "m3", "m4"]
    list_of_parameters, list_of_parameters_errors = obj.fit_image(
        fixed_centre=False,
        harmonics=harmonics
    )


    # NOTE:
    visualize = True
    if visualize:
        if name == "slacs0252+0039":
            vmin = 0.02
            vmax = 5.0
        elif name == "slacs0946+1006":
            vmin = 0.01
            vmax = 5.0
        elif name == "slacs1023+4230":
            vmin = 0.02
            vmax = 5.0
        elif name == "slacs1143-0144":
            vmin = 0.05
            vmax = 5.0
        elif name == "slacs1205+4910":
            vmin = 0.02
            vmax = 5.0
        elif name == "slacs1525+3327":
            vmin = 0.02
            vmax = 5.0
        elif name == "slacs2303+1422":
            vmin = 0.05
            vmax = 5.0
        figure, axes, figure_stats, axes_stats = ellipse_fitting_utils.visualize(
            array=obj.array,
            sample=obj.sample,
            list_of_parameters=list_of_parameters,
            a_max=obj.a_max,
            extract_condition=obj.extract_condition,
            vmin=vmin,
            vmax=vmax,
        )
        ellipse_fitting_utils.plot_list_of_parameters(
            list_of_parameters=list_of_parameters,
            list_of_parameters_errors=None,
        )
        if harmonics:
            k1, k1_errors, k3, k3_errors, k4, k4_errors = ellipse_fitting_utils.plot_multipole_amplitudes_from_list_of_parameters_lite(
                array=obj.array,
                list_of_parameters=list_of_parameters,
                list_of_parameters_errors=None,
                #ylim=(2e-4 * 100, 2e-1 * 100),
                pixel_scale=1.0,
                name=None,
            )
        plt.show()
        exit()


    # NOTE: Perform ellipse fitting for different values of the major-axis, a.
    list_of_parameters_emcee = []
    list_of_parameters_errors_emcee = []
    harmonics = []
    harmonics = ["m3", "m4"]
    for i, a in enumerate(obj.array):
        print("a =", a)
        if True:
            if i == 0: # NOTE: For the innermost ellipse allow the centre to be a free parameter, regardless of whether you keep it fixed for the rest of the ellipses.
                centre = (
                    parameters_0["x0"],
                    parameters_0["y0"],
                )
                limits = ellipse_emcee_fitting_utils.get_limits(# NOTE: Get limits for the emcee
                    fixed_centre=False,
                    centre=centre,
                    #harmonics=harmonics,
                    harmonics=[],
                )
                mappings, keys = ellipse_emcee_fitting_utils.get_mappings(# NOTE: Keys and mappings
                    fixed_centre=False,
                    #harmonics=harmonics
                    harmonics=[],
                )
            else:
                limits = ellipse_emcee_fitting_utils.get_limits(# NOTE: Get limits for the emcee
                    fixed_centre=True,
                    centre=centre,
                    harmonics=harmonics,
                )
                mappings, keys = ellipse_emcee_fitting_utils.get_mappings(# NOTE: Keys and mappings
                    fixed_centre=True,
                    harmonics=harmonics
                )

            # NOTE: Filename to save the chain from emcee
            backend_filename = "{}/{}_ellipse_fitting_{}".format(
                base_directory, name, a
            )
            if np.logical_and(
                "a_1" in keys,
                "b_1" in keys,
            ):
                backend_filename += "_m1"
            if np.logical_and(
                "a_3" in keys,
                "b_3" in keys,
            ):
                backend_filename += "_m3"
            if np.logical_and(
                "a_4" in keys,
                "b_4" in keys,
            ):
                backend_filename += "_m4"
            backend_filename += ".h5"

            # NOTE:
            if True:

                # NOTE: For the innermost ellipse use the parameters from lmfit. We can either do this for each ellipse, or, use the best-fit parameter of each ellipse as a starting point for the next (this is currently what is done)
                if i == 0:
                    parameters = list_of_parameters[i]
                else:
                    pass # NOTE: use the parameters from the previous fit ...
                #print(parameters);exit()

                # NOTE: emcee things
                p0 = []
                for j, key in enumerate(keys):
                    p_j = parameters[key]
                    if p_j is None:
                        p_j = 0.0
                    p0.append(p_j)

                # NOTE: Initialize a "dataset" object
                dataset = ellipse_emcee_fitting_utils.Dataset(
                    image=image,
                    error=error,
                    a=a,
                    mappings=mappings,
                    centre=centre,
                    extract_condition=True # NOTE: THIS PERFORMS AUTOMASKING
                )

                # NOTE: Fit ellipse. Choose emcee settings
                sampler = ellipse_emcee_fitting_utils.fit_emcee(
                    dataset=dataset,
                    p0=p0,
                    limits=limits,
                    backend_filename=backend_filename,
                    nwalkers=100,
                    #nsteps=5000,
                    nsteps=1000,
                    debug=False
                )
                # if True:
                #     figure_sampler_subplots, axes_sampler_subplots = ellipse_emcee_fitting_utils.sampler_subplots(
                #         sampler=sampler,
                #         discard=500,
                #         thin=1
                #     )
                #     plt.show()
                #     #exit()

                # NOTE: Get parameters from chain
                discard = 500
                samples_flattened = sampler.get_chain(
                    discard=discard, thin=1, flat=True
                )
                p_emcee, p_errors_emcee = emcee_wrapper.results(
                    samples_flattened=samples_flattened
                )
                if i == 0:
                    centre = (p_emcee[0], p_emcee[1])

                # NOTE: Update list of parameters
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
                parameters_errors = {
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
                for key, index in mappings.items():
                    parameters[key] = p_emcee[index]
                    parameters_errors[key] = p_errors_emcee[index]
                #print("appending ...")
                list_of_parameters_emcee.append(parameters)
                list_of_parameters_errors_emcee.append(parameters_errors)
                #exit()

    # NOTE: If fixed_centre = True, then update the best-fit parameters
    list_of_parameters_emcee_updated = []
    for i, parameters in enumerate(list_of_parameters_emcee):

        if i == 0: # NOTE: The centre is only optimized in the 1st fit...
            centre = (
                parameters["x0"],
                parameters["y0"],
            )
            parameters["a_1"] = 0.0
            parameters["b_1"] = 0.0
            parameters["a_3"] = 0.0
            parameters["b_3"] = 0.0
            parameters["a_4"] = 0.0
            parameters["b_4"] = 0.0
        else:
            pass

        # NOTE:
        if i > 0:
            parameters["x0"] = centre[0]
            parameters["y0"] = centre[1]
        list_of_parameters_emcee_updated.append(parameters)

    # NOTE: Visualization
    ellipse_fitting_utils.visualize(
        array=obj.array[:len(list_of_parameters_emcee_updated)],
        sample=obj.sample,
        list_of_parameters=list_of_parameters_emcee_updated,
        a_max=obj.a_max,
        #extract_condition=obj.extract_condition
        extract_condition=False,
    )
    # plt.show()
    # exit()

    # NOTE: Visualization
    ellipse_fitting_utils.plot_list_of_parameters(
        list_of_parameters=[
            p for p in list_of_parameters_emcee_updated[1:] if p is not None
        ],
        list_of_parameters_errors=[
            perr for perr in list_of_parameters_errors_emcee[1:] if perr is not None
        ],
        x=obj.array[1:] * pixel_scale,
    )

    # NOTE:
    plt.show()
    exit()
