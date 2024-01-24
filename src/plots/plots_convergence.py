"""
.py file for plots that deal with SM convergence. For GP, this means plotting the hyperparameters
"""

from src.plots.plots_config import *


# GPs
def plot_hp_ls(list_sm: list, n_dim: int, output_names: list, n_loc: int, label_list=None, fig_title=''):
    """Plot length scales for a given number of TP.
    All surrogates being compared must have the same input dimension.

    Args:
        list_sm (list): List with the different surrogates to plot the convergence for
        n_dim (int): number of parameters considered
        output_names (list): with the names of each output type (corresponds to each column in the figure)
        n_loc (int): number of output locations for each output type
        label_list (list, optional): List with labels, needed in len(list_sm)>1. Defaults to None.
        fig_title (str, optional): Figure title. Defaults to ''.
    """
    colormap = plt.cm.tab20
    # Create an array of equally spaced values to sample colors
    color_indices = np.linspace(0, 1, n_loc)
    # Sample the colors from the colormap
    colors = [colormap(color_index) for color_index in color_indices]

    fig, axs = plt.subplots(len(list_sm), len(output_names), dpi=150)

    if len(list_sm) == 1 and len(output_names) == 1:
        x_ = np.arange(1, n_dim + 1)
        for i, sm_ in enumerate(list_sm):
            for k in range(n_loc):
                axs.plot(x_, sm_.gp_list[k]['cl_hp'], color=colors[k], label=f'{k + 1}')

        axs.set_xlabel('Parameter')
        axs.set_ylabel('Length scale')
        handles, labels = axs.get_legend_handles_labels()
        fig.legend(handles=handles, labels=labels, loc="center right", ncol=1)
    else:
        ax = axs.reshape(-1)
        counter = 0
        for i, sm_ in enumerate(list_sm):
            c = 0
            for ot in output_names:
                for k in range(n_loc):
                    x_ = np.arange(1, sm_.gp_list[k + c]['cl_hp'].shape[0] + 1)
                    ax[counter].plot(x_, sm_.gp_list[k + c]['cl_hp'], color=colors[k], label=f'{k + 1}')
                ax[counter].set_title(f'{label_list[i]} - {ot}', loc='left', fontsize=8)
                counter = counter + 1
                c = c + n_loc

        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles=handles, labels=labels, loc="center right", ncol=1)

        fig.text(0.5, 0.08, 'Parameter number', va='center', ha='center')
        fig.text(0.04, 0.5, 'GP length scale', va='center', ha='center', rotation='vertical')

    fig.suptitle(f'{fig_title}')
    plt.subplots_adjust(top=0.9, bottom=0.15, wspace=0.5, hspace=0.8)


def plot_hp_noise(list_sm, output_names, n_loc, label_list=None, fig_title=''):
    """
    Plots the noise hyperparameter for GPs. There is one column for each output type, and in each subplot, there is one
    line for each surrogate in list_sm, to compare the results.
    Args:
        list_sm (list): List with the different surrogates to plot the convergence for
        output_names (list): with the names of each output type (corresponds to each column in the figure)
        n_loc (int): number of output locations for each output type
        label_list (list, optional): List with labels, needed in len(list_sm)>1. Defaults to None.
        fig_title (str, optional): Figure title. Defaults to ''.

    Returns: --

    """
    fig, ax = plt.subplots(1, len(output_names), dpi=150)
    if len(output_names) == 1:
        ax = np.array([ax]).T

    if label_list is None:
        label_list = np.zeros(len(list_sm))

    x_ = np.arange(1, n_loc + 1)
    for i, sm_ in enumerate(list_sm):
        y_c = []
        y_n = []

        c = 0
        for o, ot in enumerate(output_names):
            for j in range(n_loc):
                y_n.append(sm_.gp_list[j + c]['noise_hp'])
                # y_c.append(sm_.gp_list[j]['c_hp'])

            ax[o].plot(x_, y_n, color=colors_scen[i], label=label_list[i], marker=markers_scen[i])

            # ax[o].plot(x_, y_c, color=colors_scen[i], label=label_list[i])

            ax[o].set_xlabel('Output location')
            ax[o].set_ylabel('Noise HP')

            ax[o].xaxis.set_ticks(x_)
            ax[o].grid()

        # ax[1].set_xlabel('Output location')
        # ax[1].set_ylabel('Variance HP')

    if len(list_sm) > 1:
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles=handles, labels=labels, loc="center right", ncol=1)

    fig.suptitle(f'{fig_title}')
    plt.subplots_adjust(top=0.9, bottom=0.15, wspace=0.5, hspace=0.8)
