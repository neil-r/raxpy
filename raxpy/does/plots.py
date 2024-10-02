"""
    This modules provides some custom plots to show experiment designs.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
    fig, axes = plt.subplots(nrows=n_columns, ncols=n_columns, figsize=(8, 13))
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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_with_seaborn(data, names, title="Pairplot"):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Convert data to DataFrame
    df = pd.DataFrame(data, columns=names)

    # Check for NaN values
    has_nan = df.isna().any().any()

    # Fill NaNs with -0.2 if any NaNs are present
    if has_nan:
        df_filled = df.fillna(-0.2)
    else:
        df_filled = df.copy()

    num_vars = len(df_filled.columns)

    # Create an n x n grid of subplots
    fig, axes = plt.subplots(
        nrows=num_vars, ncols=num_vars, figsize=(4 * num_vars, 4 * num_vars)
    )
    fig.suptitle(title, y=0.95, fontsize=18)

    # Ensure axes is a 2D array even if num_vars is 1
    if num_vars == 1:
        axes = np.array([[axes]])

    # Loop over the grid to create plots
    for i in range(num_vars):
        for j in range(num_vars):
            ax = axes[i, j]
            x_col = df_filled.columns[j]
            y_col = df_filled.columns[i]

            if i == j:
                # Diagonal: histogram
                data_col = df_filled[x_col]
                counts, bins, patches = ax.hist(
                    data_col, bins=8, color="gray", edgecolor="black"
                )

                # Adjust y-axis limits
                y_max = counts.max()
                y_min = -0.2 if has_nan else 0
                if y_min == y_max:
                    y_min -= 0.5
                    y_max += 0.5

                # Snap y_max to nearest 0.1 if close
                y_max_snapped = snap_to_nearest_point_one(y_max)

                ax.set_ylim(y_min, y_max_snapped)

                # Adjust x-axis limits
                x_min = data_col.min()
                x_max = data_col.max()
                if has_nan:
                    x_min = min(-0.2, x_min)
                if x_min == x_max:
                    x_min -= 0.5
                    x_max += 0.5

                # Snap x_max to nearest 0.1 if close
                x_max_snapped = snap_to_nearest_point_one(x_max)

                ax.set_xlim(x_min, x_max_snapped)

                # Remove axis labels and ticks
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_yticks([])
                ax.set_yticklabels([])

                # Set tick label font size
                ax.tick_params(axis="both", which="major", labelsize=14)

                # Expand xlim and ylim by 5% after setting ticks
                x_range = x_max_snapped - x_min
                x_padding = x_range * 0.05
                ax.set_xlim(x_min - x_padding, x_max_snapped + x_padding)

                y_range = y_max_snapped - y_min
                y_padding = y_range * 0.05
                ax.set_ylim(y_min - y_padding, y_max_snapped + y_padding)

                # Add x-axis label to bottom-most subpanels
                if i == num_vars - 1:
                    ax.set_xlabel(names[j], fontsize=16)

                # Add y-axis label to leftmost subpanels
                if j == 0:
                    ax.set_ylabel(names[i], fontsize=16)

            else:
                # Off-diagonal: scatter plot
                x_data = df_filled[x_col]
                y_data = df_filled[y_col]
                ax.scatter(x_data, y_data, color="black", edgecolor="k", s=40)

                # Determine x and y limits
                x_min, x_max = x_data.min(), x_data.max()
                y_min, y_max = y_data.min(), y_data.max()
                if has_nan:
                    x_min = min(-0.2, x_min)
                    y_min = min(-0.2, y_min)

                # Handle case where min == max
                if x_min == x_max:
                    x_min -= 0.5
                    x_max += 0.5
                if y_min == y_max:
                    y_min -= 0.5
                    y_max += 0.5

                # Snap x_max and y_max to nearest 0.1 if close
                x_max_snapped = snap_to_nearest_point_one(x_max)
                y_max_snapped = snap_to_nearest_point_one(y_max)

                ax.set_xlim(x_min, x_max_snapped)
                ax.set_ylim(y_min, y_max_snapped)

                # Expand xlim and ylim by 5% after setting ticks
                x_range = x_max_snapped - x_min
                x_padding = x_range * 0.05
                ax.set_xlim(x_min - x_padding, x_max_snapped + x_padding)

                y_range = y_max_snapped - y_min
                y_padding = y_range * 0.05
                ax.set_ylim(y_min - y_padding, y_max_snapped + y_padding)

                # Set ticks and tick labels
                if has_nan:
                    ticks = [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
                    ax.set_xticks(ticks)
                    ax.set_yticks(ticks)
                else:
                    ax.set_xticks(
                        np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 6)
                    )
                    ax.set_yticks(
                        np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 6)
                    )

                # Replace ticks at -0.2 with 'null'
                x_tick_labels = [
                    (
                        "null"
                        if np.isclose(tick, -0.2, atol=0.1)
                        else f"{tick:.1f}"
                    )
                    for tick in ax.get_xticks()
                ]
                y_tick_labels = [
                    (
                        "null"
                        if np.isclose(tick, -0.2, atol=0.1)
                        else f"{tick:.1f}"
                    )
                    for tick in ax.get_yticks()
                ]

                ax.set_xticklabels(x_tick_labels)
                ax.set_yticklabels(y_tick_labels)

                # Set tick label font size
                ax.tick_params(axis="both", which="major", labelsize=14)

                # Add gridlines
                ax.grid(True)

                # Add x-axis label to bottom-most subpanels
                if i == num_vars - 1:
                    ax.set_xlabel(names[j], fontsize=16)

                # Add y-axis label to leftmost subpanels
                if j == 0:
                    ax.set_ylabel(names[i], fontsize=16)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def snap_to_nearest_point_one(value, tolerance=0.05):
    """
    If the value is within 'tolerance' of the next multiple of 0.1,
    snap it to that multiple.
    """
    remainder = value % 0.1
    if remainder >= 0.1 - tolerance:
        value = value + (0.1 - remainder)
        value = round(value, 1)
    return value
