"""
    This modules provides some custom plots to show experiment designs.
"""

import numpy as np
import matplotlib.pyplot as plt


def scatterplot_matrix(data, names: str, **kwargs):
    """
    Plots a scatterplot matrix of subplots.  Each row of "data" is
    plotted against other rows, resulting in a nrows by nrows grid of
    subplots with the diagonal subplots labeled with "names".

    Arguments
    ---------
    data
        matrix of data points
    names : str
        Labels for subplot names
    **kwargs
        Additional keyword arguments are passed on to matplotlib's
        "plot" command.

    Returns
    -------
    Returns the matplotlib figure object containg the
    subplot grid.
    """
    n_points, n_columns = data.shape
    fig, axes = plt.subplots(nrows=n_columns, ncols=n_columns, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...

        # if ax.is_first_col():
        #     ax.yaxis.set_ticks_position('left')
        # if ax.is_last_col():
        #     ax.yaxis.set_ticks_position('right')
        # if ax.is_first_row():
        #     ax.xaxis.set_ticks_position('top')
        # if ax.is_last_row():
        #     ax.xaxis.set_ticks_position('bottom')

    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i, j), (j, i)]:
            x_d = [
                (xp if xp is not None and not np.isnan(xp) else -0.25)
                for xp in data[:, x]
            ]
            y_d = [
                (xp if xp is not None and not np.isnan(xp) else -0.25)
                for xp in data[:, y]
            ]
            axes[x, y].scatter(y_d, x_d, **kwargs)
            axes[x, y].spines["top"].set_visible(True)
            axes[x, y].spines["bottom"].set_visible(True)
            axes[x, y].spines["left"].set_visible(True)
            axes[x, y].spines["right"].set_visible(True)

    # Turn on the proper x or y axes ticks.
    for i in range(n_columns):
        axes[n_columns - 1, i].xaxis.set_visible(True)
        axes[i, 0].yaxis.set_visible(True)
    # for i, j in zip(range(n_columns), itertools.cycle((-1, 0))):
    #    axes[j, i].xaxis.set_visible(True)
    #    axes[i, j].yaxis.set_visible(True)

    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i, i].annotate(
            label,
            (0.5, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
        )
        axes[i, i].spines["top"].set_visible(False)
        axes[i, i].spines["bottom"].set_visible(False)
        axes[i, i].spines["left"].set_visible(False)
        axes[i, i].spines["right"].set_visible(False)

    return fig
