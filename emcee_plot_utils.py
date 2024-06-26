import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner


def plot_chain(
    chain,
    log_prob=None,
    ncols=5,
    figsize=None,
    walkers=None,
    truths=None,
    limits=None,
    title=None,
    ylabels=None,
    show=False
):

    # ...
    if chain.shape[-1] % ncols == 0:
        nrows = int(chain.shape[-1] / ncols)
    else:
        nrows = int(chain.shape[-1] / ncols) + 1

    chain_averaged = np.average(chain, axis=1)

    figure, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize
    )
    axes_flattened = np.ndarray.flatten(axes)

    if log_prob is not None:
        log_prob_max = np.max(log_prob)
    else:
        log_prob_max = 0.0

    k = 0
    for i in range(len(axes_flattened)):
        if k < chain.shape[-1]:

            # for n in range(chain.shape[0]):
            #     axes[i, j].scatter(np.tile(n, chain.shape[1]), chain[n, :, k], c=log_prob[n, :]/log_prob_max, cmap="jet")

            for n in range(chain.shape[1]):
                axes_flattened[i].plot(chain[:, n, k], color="black", alpha=0.25)

            axes_flattened[i].plot(chain_averaged[:, k], linewidth=2, color="r", alpha=1.00)

            if truths is not None:
                axes_flattened[i].axhline(truths[k], linestyle="--", color="b")

            if limits is not None:
                axes_flattened[i].set_ylim((limits[k, 0], limits[k, 1]))

                axes_flattened[i].set_yticks(np.linspace(limits[k, 0], limits[k, 1], 3))

            if ylabels is not None:
                axes_flattened[i].set_ylabel(
                    r"{}".format(ylabels[k])
                )
            else:
                axes_flattened[i].set_ylabel("param_{}".format(k))



            k += 1
        else:
            axes_flattened[i].axis("off")


    if walkers is None:
        pass
    else:
        k = 0
        for i in range(len(axes_flattened)):
            if k < chain.shape[-1]:
                axes_flattened[i].plot(chain[:, walkers, k], color="b", alpha=0.75)
                k += 1


    if title is not None:
        figure.suptitle(title)


    plt.subplots_adjust(wspace=0.25, left=0.05, right=0.995)

    if show:
        plt.show()

    return figure, axes


# NOTE: Move this function to the main "plot_utils.py"
def plot_corner(chain, c=None, truths=None, labels=None, s=10, figsize=(10, 9)):

    #print(chain.shape)

    N = int(chain.shape[-1] - 1)

    figure, axes = plt.subplots(nrows=N, ncols=N, figsize=figsize)

    for i in range(N):
        for j in range(i + 1, N):
            axes[i, j].axis("off")


    for i in range(N):

        for j in range(0, i + 1):
            print(i, j)

            axes[i, j].plot(
                chain[:, j],
                chain[:, i+1],
                linewidth=1,
                color="black",
                alpha=0.5
            )

            sc = axes[i, j].scatter(
                chain[:, j],
                chain[:, i+1],
                cmap="jet",
                c=c,
                s=s,
                alpha=0.5
            )



            if truths:
                axes[i, j].axvline(truths[j], linestyle="--", color="black")
                axes[i, j].axhline(truths[i+1], linestyle="--", color="black")
                axes[i, j].plot([truths[j]],[truths[i+1]], linestyle="None", marker="o", markersize=10, color="black")

            if i != N-1:
                axes[i, j].set_xticks([])

            if j != 0:
                axes[i, j].set_yticks([])


    if labels:
        for i in range(N):
            axes[i, 0].set_ylabel(labels[i+1], fontsize=15)
            axes[N-1, i].set_xlabel(labels[i], fontsize=15)

    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    cbar_ax = figure.add_axes([0.85, 0.30, 0.05, 0.6])
    figure.colorbar(sc, cax=cbar_ax)

    plt.show()


def filter_chain(chain, parameter_indexes, values_min, values_max):

    idx = np.full(
        shape=chain.shape[1], fill_value=True
    )
    for i, parameter_idx in enumerate(parameter_indexes):
        for j in range(chain.shape[0]):
            idx_temp = np.logical_and(
                chain[j, :, parameter_idx] > values_min[i],
                chain[j, :, parameter_idx] < values_max[i]
            )

            idx[~idx_temp] = False

    return idx


def plot_log_prob(log_prob, truth=None, xlabel="# of steps", ylabel="-logL", xlim=None, ylim=None):

    figure = plt.figure(
        figsize=(10, 5)
    )


    for i in range(log_prob.shape[1]):
        plt.plot(
            np.arange(log_prob.shape[0]),
            -log_prob[:, i],
            color="black",
            alpha=0.5
        )

    if truth:
        plt.axhline(
            -truth,
            linestyle="--",
            color="b"
        )

    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.yscale("log")
    plt.show()

def plot_list_of_log_probs(list_of_log_probs, truth=None, xlabel="# of steps", ylabel="-Likelihood", xlim=None, ylim=None, legends=None):

    figure = plt.figure(
        figsize=(10, 5)
    )

    # TODO: initialize random colors
    colors = ["b", "r"]

    legend_conditions = np.full(
        shape=(len(legends), ), fill_value=True
    )

    for j, log_prob in enumerate(list_of_log_probs):
        for i in range(log_prob.shape[1]):

            plt.plot(
                np.arange(log_prob.shape[0]),
                -log_prob[:, i],
                color=colors[j],
                alpha=0.5,
                label=legends[j] if legends is not None and legend_conditions[j] else None
            )
            if legend_conditions[j]:
                legend_conditions[j] = False

    if truth:
        plt.axhline(
            -truth,
            linestyle="--",
            color="black"
        )

    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.yscale("log")
    if legends is not None:
        plt.legend(fontsize=15)
    plt.show()


def get_best_fit_parameters_from_chain_as_50th_percentile(chain):


    if len(chain.shape) == 2:
        pass
    elif len(chain.shape) == 3:
        chain = chain.reshape(-1, chain.shape[-1])
    else:
        raise ValueError

    best_fit_parameters = np.zeros(
        shape=chain.shape[-1],
        dtype=np.float
    )
    for i in range(chain.shape[-1]):
        best_fit_parameters[i] = np.percentile(
            a=chain[:, i], q=50.0
        )

    return best_fit_parameters

if __name__ =="__main__":

    pass
