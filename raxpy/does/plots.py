"""
    This modules provides some custom plots to show experiment designs.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_scatterplot_matrix(data, names, title="Pairplot"):
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
    title : str
        Additional keyword arguments are passed on to matplotlib's
        "plot" command.

    Returns
    -------
    Returns the matplotlib figure object containg the
    subplot grid.
    """

    # Check for NaN values
    has_nan = np.isnan(data).any()

    # Fill NaNs with -0.2 if any NaNs are present
    if has_nan:
        np_a_filled = np.nan_to_num(data, nan=-0.2)
    else:
        np_a_filled = np.copy(data)

    num_vars = np_a_filled.shape[1]

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
            x_col = np_a_filled[:, j]
            y_col = np_a_filled[:, i]

            if i == j:
                # Diagonal: histogram
                data_col = x_col
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
                y_max_snapped = _snap_to_nearest_point_one(y_max)

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
                x_max_snapped = _snap_to_nearest_point_one(x_max)

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
                x_data = x_col
                y_data = y_col
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
                x_max_snapped = _snap_to_nearest_point_one(x_max)
                y_max_snapped = _snap_to_nearest_point_one(y_max)

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


def _snap_to_nearest_point_one(value, tolerance=0.05):
    """
    If the value is within 'tolerance' of the next multiple of 0.1,
    snap it to that multiple.
    """
    remainder = value % 0.1
    if remainder >= 0.1 - tolerance:
        value = value + (0.1 - remainder)
        value = round(value, 1)
    return value
