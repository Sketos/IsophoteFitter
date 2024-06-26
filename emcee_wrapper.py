import os, sys, time#, random
import numpy as np
import matplotlib.pyplot as plt
import emcee
from multiprocessing import Pool
import h5py
#from schwimmbad import MPIPool # NOTE: This is not curerntly implemented and it's not working on COSMA

# # NOTE: Not sure if this is actually needed ...
# os.environ["OMP_NUM_THREADS"] = "1"

#import cosma_utils as cosma_utils
import emcee_plot_utils as emcee_plot_utils


# class emcee_wrapper:
#
#
#     def __init__(self, data, log_likelihood_func):
#         self.data = data
#         self.log_likelihood_func = log_likelihood_func
#
#         ...
#         ...
#         ...
#
#
#     def log_likelihood(self, theta):
#
#         # NOTE: I am passing the obj itself, so that the log_likelihood func has the data
#         _log_likelihood = self.log_likelihood_func(
#             self, theta
#         )
#
#         return _log_likelihood
#
#
#     def log_probability(self, theta):
#
#         lp = self.log_prior(theta)
#
#         if not np.isfinite(lp):
#             return -np.inf
#         else:
#             return lp + self.log_likelihood(theta=theta)
#
#
#     def run(self, nsteps, parallel=False):
#
#         def run_func(nsteps, pool=None):
#             sampler = emcee.EnsembleSampler(
#                 nwalkers=self.nwalkers,
#                 ndim=self.ndim,
#                 log_prob_fn=self.log_probability,
#                 backend=self.backend,
#                 pool=pool
#             )
#
#             sampler.run_mcmc(
#                 initial_state=self.initial_state,
#                 nsteps=nsteps - self.previous_nsteps,
#                 progress=True
#             )
#
#             return sampler
#
#         if parallel:
#             with MPIPool() as pool:
#                 if not pool.is_master():
#                     pool.wait()
#                     sys.exit(0)
#
#                 sampler = run_func(nsteps=nsteps, pool=pool)

# NOTE:
def results(samples_flattened):

    a = np.zeros(
        shape=(samples_flattened.shape[1], 3)
    )
    for i in range(samples_flattened.shape[1]):
        value_lower, value, value_upper = np.percentile(
            samples_flattened[:, i], [16, 50, 84]
        )
        a[i, 0] = value
        a[i, 1] = abs(value - value_lower)
        a[i, 2] = abs(value - value_upper)

    # NOTE:
    a_errors_lower = a[:, 1]
    a_errors_upper = a[:, 2]
    a_errors = np.array([
        max(x, y) for x, y in zip(a_errors_lower, a_errors_upper)
    ])

    return a[:, 0], a_errors


class emcee_wrapper:

    # NOTE: move "limits", "nwalkers", "backend_filename" in the run_mcmc function of this class
    def __init__(
        self,
        dataset,
        log_likelihood_func,
        limits,
        p0,
        nwalkers=500,
        backend_filename="backend.h5",
        parallization=False,
        eps=1e-3
    ):

        # NOTE: This helper object
        self.dataset = dataset

        # NOTE:
        self.log_likelihood_func = log_likelihood_func

        # ...
        if limits is None:
            raise ValueError("...")
        else:
            self.limits = limits

        self.par_min = limits[:, 0]
        self.par_max = limits[:, 1]

        self.ndim = limits.shape[0]

        self.nwalkers = nwalkers

        # NOTE: The backend is not working properly when a run is reset ...
        self.backend = emcee.backends.HDFBackend(
            filename=backend_filename
        )

        self.previous_nsteps = 0


        # NOTE: There is an issue with "backend.get_last_sample" when using MPI
        self.parallization = parallization
        if not self.parallization:
            try:
                self.initial_state = self.backend.get_last_sample()
            except:
                # NOTE: Previous version of the "initialize_state" function.
                # self.initial_state = self.initialize_state(
                #     par_min=self.par_min,
                #     par_max=self.par_max,
                #     ndim=self.ndim,
                #     nwalkers=self.nwalkers
                # )

                # NOTE: THE CODE WORKS SIGNIFICANTLY BETTER IF ALL WALKERS ARE INITIALLIZED IN A BALL AROUND p0. Using flat priors results in some walkers getting infinitelly stuck which ...
                if p0 is None:
                    self.initial_state = self.initialize_state(
                        p0=p0,
                        ndim=self.ndim,
                        nwalkers=self.nwalkers,
                        limits=self.limits,
                        eps=eps
                    )

                    # ---------- #
                    # NOTE:
                    # ---------- #
                    values = np.zeros(shape=(self.nwalkers, ))
                    for i in range(self.nwalkers):
                        print("i =", i)#;exit()
                        values[i] = self.log_likelihood(theta=self.initial_state[i, :])

                    p0 = self.initial_state[np.argmax(values), :]

                    self.initial_state = self.initialize_state_from_p0(
                        p0=p0,
                        ndim=self.ndim,
                        nwalkers=self.nwalkers,
                        limits=self.limits,
                        eps=eps
                    )
                    # ---------- #
                else:
                    self.initial_state = self.initialize_state_from_p0(
                        p0=p0,
                        ndim=self.ndim,
                        nwalkers=self.nwalkers,
                        limits=self.limits,
                        eps=eps
                    )

                self.check_state(
                    state=self.initial_state,
                    par_min=self.par_min,
                    par_max=self.par_max
                )

                self.backend.reset(
                    self.nwalkers, self.ndim
                )
        elif self.parallization in ["MPI", "OpenMP"]:
            raise NotImplementedError()

            """
            # NOTE: Previous version of the "initialize_state" function.
            # self.initial_state = self.initialize_state(
            #     par_min=self.par_min,
            #     par_max=self.par_max,
            #     ndim=self.ndim,
            #     nwalkers=self.nwalkers
            # )

            self.initial_state = self.initialize_state(
                p0=p0,
                ndim=self.ndim,
                nwalkers=self.nwalkers,
                limits=self.limits,
                eps=eps
            )
            self.check_state(
                state=self.initial_state,
                par_min=self.par_min,
                par_max=self.par_max
            )

            self.backend.reset(
                self.nwalkers, self.ndim
            )
            """
        else:
            raise ValueError("...")



        self.previous_nsteps += self.backend.iteration


    # @staticmethod
    # def initialize_state(par_min, par_max, ndim, nwalkers):
    #
    #     return np.array([
    #         par_min + (par_max - par_min) * np.random.rand(ndim)
    #         for i in range(nwalkers)
    #     ])

    @staticmethod
    def initialize_state_from_p0(
        p0,
        ndim,
        nwalkers,
        limits,
        eps=1e-5
    ):

        initial_state = p0 + eps * np.random.randn(
            nwalkers, ndim
        )

        # # NOTE: IS THIS A GOOD THING TO DO?
        # eps = np.log10(abs(p0 / 10))#.astype(int)
        # initial_state = np.zeros(shape=(nwalkers, ndim))
        # for i in range(ndim):
        #     initial_state[:, i] = p0[i] + 10**eps[i] * np.random.randn(
        #         nwalkers
        #     )

        for i in range(ndim):
            par_min, par_max = limits[i, :]
            for j in range(nwalkers):
                p = initial_state[j, i]
                # print(
                #     "i = {}: p = {} | min = {}, max = {}".format(i, p, par_min, par_max)
                # )

                # NOTE:
                # if p < par_min:
                #     initial_state[j, i] = par_min
                # if p > par_max:
                #     initial_state[j, i] = par_max

                # NOTE:
                if np.logical_or(p < par_min, p > par_max):
                    initial_state[j, i] = p0[i]

        return initial_state


    @staticmethod
    def initialize_state(
        ndim,
        nwalkers,
        limits,
        p0=None,
        eps=1e-5
    ):

        # np.random.seed(
        #     int(time.time())
        # )

        # NOTE: This creates a ball around a given set of parameters
        # initial_state = p0 + eps * np.random.randn(
        #     nwalkers,
        #     ndim
        # )

        # NOTE:
        # --- #
        if p0 is None:
            initial_state = np.zeros(
                shape=(nwalkers, ndim)
            )
            for i in range(ndim):
                initial_state[:, i] = np.random.uniform(
                    low=limits[i, 0], high=limits[i, 1], size=nwalkers
                )
        else:
            initial_state = p0 + eps * np.random.randn(
                nwalkers,
                ndim
            )
        # --- #

        return initial_state


    def check_state(
        self,
        state,
        par_min,
        par_max,
    ):

        # # NOTE: Figure
        # for i in range(state.shape[1]):
        #     plt.figure()
        #     plt.plot(
        #         state[:, i],
        #         linestyle="None",
        #         marker="o",
        #         color="b"
        #     )
        #     plt.axhline(par_min[i], linestyle="--", color="black")
        #     plt.axhline(par_max[i], linestyle="--", color="black")
        # plt.show()
        # exit()

        # TODO:
        # state_mean = np.mean(state, axis=0)
        # print(state_mean);exit()

        for i in range(state.shape[0]):

            conditions = self.log_prior_conditions(
                values=state[i, :],
                values_min=par_min,
                values_max=par_max
            )

            if np.all(conditions):
                pass
            else:
                # TODO: if any values are outside the range (min, max) set them equal to the state_mean
                raise ValueError("Some values in the initial state are outside the limits")


    def log_prior(
        self,
        theta
    ):

        conditions = self.log_prior_conditions(
            values=theta,
            values_min=self.par_min,
            values_max=self.par_max
        )

        if np.all(conditions):
            return 0.0
        else:
            return -np.inf


    def log_prior_conditions(
        self,
        values,
        values_min,
        values_max,
    ):

        conditions = np.zeros(
            shape=values.shape, dtype=bool
        )

        for i, value in enumerate(values):
            if values_min[i] < value < values_max[i]:
                conditions[i] = True

        return conditions


    def log_likelihood(self, theta):

        # NOTE: pass the object to have flexibility on what to do on the log_likelihood function.
        return self.log_likelihood_func(
            self.dataset, theta
        )

    def log_probability(self, theta):

        lp = self.log_prior(theta)

        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp + self.log_likelihood(theta=theta)


    # NOTE: This function makes the "log_probability" pickleable. I dont think so ...
    #def __call__(self, theta):
    #    return self.log_probability(theta)


    def run(self, nsteps, parallel=False):

        def run_func(nsteps, pool=None):
            sampler = emcee.EnsembleSampler(
                nwalkers=self.nwalkers,
                ndim=self.ndim,
                log_prob_fn=self.log_probability,
                backend=self.backend,
                pool=pool
            )

            sampler.run_mcmc(
                initial_state=self.initial_state,
                nsteps=nsteps - self.previous_nsteps,
                progress=True
            )

            return sampler

        # NOTE: I need to able to check if MPI is possible
        # NOTE: Does it make sense to return the sampler when in parallel mode?
        # NOTE: MPI is not working
        if parallel:
            if self.parallization == "MPI":

                raise ValueError("This is not currently implemented")
                # with MPIPool() as pool:
                #     if not pool.is_master():
                #         pool.wait()
                #         sys.exit(0)
                #
                #     sampler = run_func(nsteps=nsteps, pool=pool)
            else:
                with Pool() as pool:
                    sampler = run_func(nsteps=nsteps, pool=pool)
        else:
            sampler = run_func(nsteps=nsteps)

        return sampler


# NOTE:
def model(x, theta):

    a, b, c = theta

    return a * x**2.0 + b * x + c

# def model(x, theta):
#
#     a, b, c, d, e = theta
#
#     return a * x**2.0 + b * x + c + np.exp(-d * x**2.0) + e * x**3.0



# NOTE:
def log_likelihood_func(dataset, theta):

    def noise_normalization(y_error):
        return np.sum(
            np.log(2.0 * np.pi * y_error**2.0)
        )

    y_model = model(dataset.x, theta)

    chi_square = np.sum(
        (dataset.y - y_model)**2.0 / dataset.y_error**2.0
    )

    log_likelihood = -0.5 * (
        chi_square + noise_normalization(y_error=dataset.y_error)
    )

    print("theta =", theta, "-> log_likelihood =", log_likelihood)

    return log_likelihood


class Dataset:

    def __init__(self, x, y, y_error):
        self.x = x
        self.y = y
        self.y_error = y_error


def get_random_state_from_limits(limits):

    # np.random.seed(
    #     int(time.time())
    # )

    random_state = np.array(
        limits[:, 0] + (limits[:, 1] - limits[:, 0]) * np.random.rand(limits.shape[0])
    )
    #print("random_state =", random_state);exit()

    return random_state


def check_values(values, limits):

    if limits.shape[0] == len(values):
        conditions = np.zeros(
            shape=limits.shape[0],
            dtype=bool
        )

        for i, value in enumerate(values):
            if limits[i, 0] < value < limits[i, 1]:
                conditions[i] = True

        if np.all(conditions):
            print("OK")
        else:
            raise ValueError("...")
    else:
        raise ValueError("...")

# def get_random_state_from_limits(limits, types=None):
#
#     # np.random.seed(
#     #     int(time.time())
#     # )
#
#     lower_limits = limits[:, 0]
#     upper_limits = limits[:, 1]
#
#     random_state = np.zeros(
#         shape=limits.shape[0],
#         dtype=np.float
#     )
#
#     for i in range(limits.shape[0]):
#         if types[i] == "Uniform":
#             random_state[i] =
#     exit()
#
#     # if types is None:
#     #     types =
#     #
#     # 10.0 ** (
#     #             np.log10(self.lower_limit)
#     #             + unit * (np.log10(self.upper_limit) - np.log10(self.lower_limit))
#     #     )
#     #
#     # return np.array(
#     #     limits[:, 0] + (limits[:, 1] - limits[:, 0]) * np.random.rand(limits.shape[0])
#     # )


import corner

if __name__ == "__main__":
    os.system("rm backend.h5")

    #eps = 1e-3
    eps = 1e1

    xmin = -1.0
    xmax = 2.0
    xnum = 20
    x = np.linspace(xmin, xmax, xnum)

    a_true = 100.0
    b_true = -2.5
    c_true = 0.5
    #d_true = -0.5
    #e_true = 1.5
    theta = [a_true, b_true, c_true]
    #theta = [a_true, b_true, c_true, d_true, e_true]
    y = model(
        x=x, theta=theta
    )

    y_error = np.random.normal(0.0, 0.2, size=len(x))
    y += y_error

    # plt.errorbar(x, y, y_error=y_error)
    # plt.show()
    # exit()

    dataset = Dataset(x=x, y=y, y_error=y_error)


    limits = np.array([
        # [-50.0, 50.0],
        # [-50.0, 50.0],
        # [-50.0, 50.0],
        [-500.0, 500.0],
        [-500.0, 500.0],
        [-500.0, 500.0],
    ])
    # limits = np.array([
    #     [-5.0, 5.0],
    #     [-5.0, 5.0],
    #     [-5.0, 5.0],
    #     [-5.0, 5.0],
    #     [0.0, 5.0],
    # ])

    #p0 = [-25.0, 25.0, 10.0]
    #p0 = [-2.5, 2.5, 1.0, 0.0, 2.0]
    p0 = get_random_state_from_limits(
        limits=limits
    )


    # NOTE:
    # backend_directory = os.path.dirname(
    #     os.path.realpath(__file__)
    # )
    # backend_filename = "backend.h5"
    # os.system(
    #     "rm {}/{}".format(
    #         backend_directory, backend_filename
    #     )
    # )
    obj = emcee_wrapper(
        #helper=helper_obj,
        dataset=dataset,
        log_likelihood_func=log_likelihood_func,
        limits=limits,
        p0=p0,
        nwalkers=100,
        # backend_filename="{}/{}".format(
        #     backend_directory, backend_filename
        # ),
        parallization=False,
        eps=eps,
    )

    # NOTE:
    sampler = obj.run(
        nsteps=200,
        parallel=False
    )

    chain = sampler.get_chain(
        discard=0,
        thin=1,
        flat=False
    )


    # emcee_plot_utils.plot_chain(
    #     chain=chain,
    #     ncols=2,
    #     figsize=(20, 6),
    #     truths=theta
    # )
    chain_indexed = chain[100:, :, :]
    samples = chain_indexed.reshape(int(chain_indexed.shape[0] * chain_indexed.shape[1]), chain_indexed.shape[2])



    fig = corner.corner(samples, truths=theta)
    plt.show()

    # # # plt.figure()
    # # # plt.errorbar(x, y, y_error=y_error, linestyle="None", marker="o")
    # # # plt.show()
    # # # exit()
    #
    # import corner
    #
    # fig = corner.corner(
    #     flat_samples, truths=theta
    # )
    # plt.show()
    #
    # # NOTE: If no errors are added then the true value is recovered exactly, otherwise the best-fit value is within the 1sigma
