import numpy as np
import matplotlib.pyplot as plt

# NOTE:
import matplotlib.cm as cm
import matplotlib as mpl

# NOTE:
from mpl_toolkits.axes_grid1 import make_axes_locatable



# def figure(figsize):
#
#     figure = plt.figure()
#
#     return figure
#
# def subplots(nrows, ncols, figsize):
#
#     figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
#
#     return figure, axes

def get_common_xlim_from_axes(axes):

    xlim_min = np.min([
        ax.get_xlim()[0]
        for ax in np.ndarray.flatten(axes)
    ])
    xlim_max = np.max([
        ax.get_xlim()[1]
        for ax in np.ndarray.flatten(axes)
    ])

    return (xlim_min, xlim_max)

def get_common_ylim_from_axes(axes):

    ylim_min = np.min([
        ax.get_ylim()[0]
        for ax in np.ndarray.flatten(axes)
    ])
    ylim_max = np.max([
        ax.get_ylim()[1]
        for ax in np.ndarray.flatten(axes)
    ])

    return (ylim_min, ylim_max)

def set_common_xlim_from_axes(axes):

    xlim = get_common_xlim_from_axes(axes=axes)
    for ax in np.ndarray.flatten(axes):
        ax.set_xlim(xlim)

def set_common_ylim_from_axes(axes):

    ylim = get_common_ylim_from_axes(axes=axes)
    for ax in np.ndarray.flatten(axes):
        ax.set_ylim(ylim)


def reset_xlim_and_ylim_from_axes(axes):

    xlim_min = np.min([
        ax.get_xlim()[0]
        for ax in axes
    ])
    xlim_max = np.max([
        ax.get_xlim()[1]
        for ax in axes
    ])

    ylim_min = np.min([
        ax.get_ylim()[0]
        for ax in axes
    ])
    ylim_max = np.max([
        ax.get_ylim()[1]
        for ax in axes
    ])
    #print(xlim_min, xlim_max);exit()
    xlim = (xlim_min, xlim_max)
    ylim = (ylim_min, ylim_max)

    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_xlim(ylim)

def plot_with_twinx(
    y0,
    y1,
    x=None,
):

    figure, axes = plt.subplots()

    axes_twinx = axes.twinx()

    axes.plot(
        x if x is not None else np.arange(len(y0)),
        y0,
        marker="o",
        color="b",
    )
    axes_twinx.plot(
        x if x is not None else np.arange(len(y1)),
        y1,
        marker="o",
        color="r",
    )

    plt.show()

def add_colorbar_to_axes(
    figure,
    im,
    axes,
    ticks=None,
    loc="right",
    ticks_position="right",
    cmap="jet",
    label=None,
    labelpad=None,
    rotation=0,
    pad=0.05,
    format=None,
    return_colorbar=True
):

    # TODO: Make an example to demonstrate how to use this function

    cax = make_axes_locatable(
        axes
    ).append_axes(loc, size='5%', pad=pad)

    if loc in ["top"]:
        orientation = "horizontal"
    else:
        orientation = "vertical"

    # NOTE:
    colorbar = figure.colorbar(
        im,
        cax=cax,
        orientation=orientation,
        format=format,
        cmap=cmap,
        ticks=ticks,
    )

    # NOTE:
    if loc in ["top", "bottom"]:
        colorbar.ax.xaxis.set_ticks_position("top")

    if label is not None:
        colorbar.set_label(
            label,
            weight="bold",
            labelpad=labelpad,
            rotation=rotation
        )

    if return_colorbar:
        return colorbar


def add_colorbar_for_pixelization_to_axes(
    axes,
    figure,
    value_min,
    value_max,
    loc="right",
    label=None,
    pad=0.05,
    format=None,
    cmap="jet",
):

    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([value_min, value_max])

    cax = make_axes_locatable(
        axes
    ).append_axes(loc, size='5%', pad=pad)

    if loc in ["top"]:
        orientation = "horizontal"
    else:
        orientation = "vertical"

    colorbar = figure.colorbar(
        mappable,
        cax=cax,
        orientation=orientation,
        format=format,
        cmap=cmap
    )

    if loc == "top":
        colorbar.ax.xaxis.set_ticks_position("top")

    return colorbar

def colors_from(n, cmap="jet"):

    cmap_mappable = plt.get_cmap(cmap)

    return [
        cmap_mappable(i) for i in np.linspace(
            0.0, 1.0, n
        )
    ]

def colorbar_from():
    pass


def mappable_from(a, cmap):
    pass


def axes_iterable(axes):

    shape = axes.shape

    list_of = []
    if len(shape) == 1:
        for i in range(shape[0]):
            list_of.append(axes[i])
    elif len(shape) == 2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                list_of.append(axes[i, j])
    else:
        raise ValueError(
            "This has not been implemented yet"
        )

    return list_of


def colorbar_with_bounds():

    # NOTE: https://matplotlib.org/stable/tutorials/colors/colorbar_only.html

    cmap = mpl.cm.viridis
    bounds = [-1, 2, 5, 7, 12, 15]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')


def colorbar_custom(
    figure,
    xmin,
    ymin,
    dx,
    dy,
    array=[0, 1],
    ticks=None,
    orientation="horizontal",
    cmap="jet"
):

    cax = figure.add_axes([xmin, ymin, dx, dy])

    mappable = cm.ScalarMappable(cmap=plt.get_cmap(cmap))
    mappable.set_array(array)

    colorbar = plt.colorbar(
        mappable=mappable,
        cax=cax,
        orientation=orientation,
        ticks=ticks,
    )

    return colorbar

def example_1():

    figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    n = 100

    im0 = np.random.normal(0.0, 1.0, size=(n, n))
    im1 = np.random.normal(0.0, 1.0, size=(n, n))
    im2 = np.random.normal(0.0, 1.0, size=(n, n))

    vmin = np.min([np.min(im) for im in [im0, im1, im2]])
    vmax = np.max([np.max(im) for im in [im0, im1, im2]])

    cmap = "jet"

    _im0 = axes[0].imshow(im0, cmap=cmap, vmin=vmin, vmax=vmax)
    _im1 = axes[1].imshow(im1, cmap=cmap, vmin=vmin, vmax=vmax)
    _im2 = axes[2].imshow(im2, cmap=cmap, vmin=vmin, vmax=vmax)

    for i in range(axes.shape[0]):
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    dx = 0.3
    cax_axes_0 = figure.add_axes([0.05 + 1.0 / 2.0 * dx, 0.825, 0.125, 0.05])
    cax_axes_1 = figure.add_axes([0.05 + 3.0 / 2.0 * dx, 0.825, 0.125, 0.05])
    cax_axes_2 = figure.add_axes([0.05 + 5.0 / 2.0 * dx, 0.825, 0.125, 0.05])

    _cmap = plt.get_cmap(cmap)

    mappable_for_im0 = cm.ScalarMappable(cmap=_cmap)
    mappable_for_im0.set_array(im0.flatten())
    mappable_for_im1 = cm.ScalarMappable(cmap=_cmap)
    mappable_for_im1.set_array(im1.flatten())
    mappable_for_im2 = cm.ScalarMappable(cmap=_cmap)
    mappable_for_im2.set_array(im2.flatten())

    colorbar_axes_0 = plt.colorbar(
        mappable=mappable_for_im0, cax=cax_axes_0, orientation="horizontal"
    )
    colorbar_axes_1 = plt.colorbar(
        mappable=mappable_for_im1, cax=cax_axes_1, orientation="horizontal"
    )
    colorbar_axes_2 = plt.colorbar(
        mappable=mappable_for_im2, cax=cax_axes_2, orientation="horizontal"
    )

    # NOTE: This line must follow after the "plt.colorbar" is called
    cax_axes_2.xaxis.set_ticks_position('top')

    plt.subplots_adjust(wspace=0.0, hspace=0.0, left=0.05, right=0.95, bottom=0.05, top=0.95)
    plt.show()


def imshow(a, cmap="jet", show=True):

    plt.figure()
    plt.imshow(a, origin="lower", cmap=cmap)
    plt.xticks([])
    plt.yticks([])

    if show:
        plt.show()


def add_circle_to_axes(
    axes,
    centre,
    radius,
    facecolor="None",
    edgecolor="black",
    linewidth=5,
):

    for ax in np.ndarray.flatten(axes):

        circle = plt.Circle(
            centre,
            radius,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )

        ax.add_patch(circle)


def get_colors_from_cmap(n, cmap="jet"):

    # NOTE:
    cmap = plt.get_cmap(cmap)

    # NOTE:
    values = np.linspace(0, 1, n)

    # NOTE: ...
    return [
        cmap(value) for value in values
    ]

if __name__ == "__main__":

    import matplotlib.gridspec as gridspec


    # NOTE:
    # --- #
    # figure = plt.figure(
    #     tight_layout=True,
    #     figsize=(12, 6)
    # )
    # gs = gridspec.GridSpec(2, 3)
    # axes_0 = figure.add_subplot(gs[0, 0:2])
    # axes_1 = figure.add_subplot(gs[1, 0:2])
    # axes_2 = figure.add_subplot(gs[0, 2])
    # axes_3 = figure.add_subplot(gs[1, 2])
    #
    # figure.subplots_adjust(wspace=0.0, hspace=0.0)
    # plt.show()
    # --- #

    # --- #
    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        #sharex=True,
        sharey=True,
        gridspec_kw={
            'width_ratios': [2, 1],
            'height_ratios': [1, 1],
        },
        figsize=(12, 6)
    )
    figure.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.show()
    # --- #

    # nrows = 2
    # ncols = 10
    # figure, axes = plt.subplots(nrows=nrows, ncols=ncols)
    #
    # for i, ax in enumerate(
    #     axes_iterable(axes=axes)
    # ):
    #
    #     print(i)

    #example_1()

    # def example_2():
    #
    #     x = np.random.normal(0.0, 1.0, 100)
    #     y = np.random.normal(0.0, 1.0, 100)
    #
    #     figure, axes = plt.subplots(nrows=1, ncols=2)
    #     axes[0].scatter(x, y)
    #     axes[1].scatter(x, y)
    #     axes[0].minorticks_on()
    #     axes[0].set_xticks([-2.0, 0.0, 2.0])
    #     axes[0].set_yticks([-2.0, 0.0, 2.0])
    #     axes[0].set_xticklabels([-2.0, 0.0, 2.0])
    #     axes[0].tick_params(
    #         axis='both',
    #         which="major",
    #         length=5,
    #         right=True,
    #         top=True,
    #         colors='black',
    #         direction="in"
    #     )
    #     axes[0].tick_params(
    #         axis='both',
    #         which="minor",
    #         length=2,
    #         right=True,
    #         top=True,
    #         colors='black',
    #         direction="in"
    #     )
    #     plt.show()
    # example_2()
