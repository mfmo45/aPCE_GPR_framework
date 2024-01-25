from src.plots.plots_config import *

best_vals = {'rmse': 0,
             'mse': None,
             'nse': 1,
             'r2': 1,
             'mean_error': 0,
             'std_error': 0,
             'norm_error': 1,
             'P95': 1
             }


def plot_correlation(sm_out, valid_eval, output_names, label_list, n_loc_=1, fig_title=''):
    """Function plots the scatter plots for the outputs, comparing the validation output (x axis) and the
    surrogate outputs (y axis)

    Args:
        sm_out (np.array): surrogate outputs, of size [mc_size, n_obs].
        valid_eval (np.array): array [mc_size, n_obs], with the validation output
        output_names (list): with names of the different output types
        label_list (list): size same as output names. It contains the R2 information to add to each subplot label
        n_loc_ (int, optional): with the number of locations, where each output_names is read. Defaults to 1.
        fig_title (str, optional): Title of the plot_description_. Defaults to ''.
    """
    colormap = plt.cm.tab20
    color_indices = np.linspace(0, 1, n_loc_)
    colors_obs = [colormap(color_index) for color_index in color_indices]

    fig, axs = plt.subplots(1, len(output_names))
    if len(output_names) == 1:
        for i in range(n_loc_):
            axs.scatter(valid_eval[:, i], sm_out[:, i], color=colors_obs[i], label=f'{i + 1}')

        axs.plot([np.min(valid_eval), np.max(valid_eval)], [np.min(valid_eval), np.max(valid_eval)], color='black')
        axs.set_xlabel(f'Full complexity model outputs')
        axs.set_ylabel(f'Simulator outputs')
        if label_list is not None:
            axs.set_title(f'{output_names[0]} - R2:{label_list[0]}', loc='left')
        fig.suptitle(f'{fig_title}')
        handles, labels = axs.get_legend_handles_labels()
    else:
        c = 0
        for o, ot in enumerate(output_names):
            for i in range(n_loc_):
                axs[o].scatter(valid_eval[:, i + c], sm_out[:, i + c], color=colors_obs[i], label=f'{i + 1}')

            mn = np.min(np.hstack((valid_eval[:, c:n_loc_ + c], sm_out[:, c:n_loc_ + c])))
            mx = np.max(np.hstack((valid_eval[:, c:n_loc_ + c], sm_out[:, c:n_loc_ + c])))

            axs[o].plot([mn, mx], [mn, mx], color='black', linestyle='--')
            axs[o].set_title(f'{ot} - R2:{label_list[o]}', loc='left')
            axs[o].set_xlabel(f'Full complexity model outputs')

            if o == 0:
                axs[o].set_ylabel(f'Simulator outputs')
            c = c + n_loc_

        fig.suptitle(f'{fig_title}')
        handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(handles=handles, labels=labels, loc="center right", ncol=1)
    plt.subplots_adjust(top=0.9, bottom=0.15, wspace=0.2, hspace=0.5)


def plot_validation_loc(eval_dict_list, label_list, output_names, n_loc, criteria=None, fig_title=''):
    """
    Plots the evaluation criteria for each output location, for different surrogate scenarios. Min 1 surrogate scenario
    evaluation criteria must be passed.
    Args:
        eval_dict_list: list with evaluation criteria (as dictionaries)
        label_list: (list) of strings, with the label for each eval_dict in the previous list
        output_names: (list) of strings with output type names
        n_loc: (int) with the number of output locations pero output type
        criteria: (list) of strings with the criteria to plot. Default is None, so all criteria are plotted
        fig_title: (string) with the title of the figure

    Returns:

    """
    # Plot criteria values for each output location, for increasing number of KLD coefficients

    if criteria is None:
        criteria_list = eval_dict_list[0].keys()
    else:
        criteria_list = criteria

    # fig, ax = plt.subplots(len(criteria_list), len(output_names), dpi=150, sharex=True)
    if len(criteria_list) < 2:
        plt_sz = plot_size(fraction=1, height_fraction=0.4)
    else:
        plt_sz = plot_size(fraction=1, height_fraction=1.0)
    fig, ax = plt.subplots(len(criteria_list), len(output_names), figsize=plt_sz, sharex=True)

    if len(output_names) == 1:
        ax = np.array([ax]).T

    x_ = np.arange(1, n_loc + 1)  # y axis is each output location

    # fig, ax = plt.subplots(len(eval_full), 1, dpi=250, sharex=True)
    for c, crit in enumerate(criteria_list):  # loop through each criteria (row)
        for o, ot in enumerate(output_names):  # loop through each output type (column)
            for k, scenario in enumerate(eval_dict_list):
                ax[c, o].plot(x_, scenario[crit][ot], color=colors_scen[k], label=label_list[k], marker=markers_scen[k])

            ax[c, o].set_ylabel(crit)
            ax[c, o].plot(x_, np.full(n_loc, best_vals[crit]), color='black', linestyle='--')

            ax[c, o].xaxis.set_ticks(x_)
            ax[c, o].grid()

        if c == len(criteria_list) - 1:
            ax[c, o].set_xlabel('Output location')
        elif c == 0:
            ax[c, o].set_title(f'{ot}', loc='left')

    if len(eval_dict_list) > 1:
        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles=handles, labels=labels, loc="center right", ncol=1)
    fig.suptitle(f'{fig_title}')
    plt.subplots_adjust(top=0.9, bottom=0.15, wspace=0.2, hspace=0.2)


def plot_validation_tp(eval_dict_list, label_list, output_names, n_loc,
                       criteria=None, plot_loc=False,
                       x_label=None, fig_title=''):
    """Function plots the evaluation criteria with increasing number of training points, for different scenarios that
    are to be compared.

    Args:
        eval_dict_list (list): of dictionaries, with each dictionary corresponding to the evaluation criteria for a
        given surrogate iteration. The dictionary must be filled with the "save_valid_criteria" function
        label_list (list): of strings, with labels for each scenario in eval_dict_list
        output_names (list): of strings with output type names
        n_loc (int): with the number of output locations pero output type
        plot_loc (bool): True to plot (lighter) lines for each output location, False to only print mean of output locs
        criteria (list, optional): of strings with the criteria to plot. Default is None, so all criteria are plotted.
        Defaults to None.
        fig_title (str, optional): With figure title. Defaults to ''.
    """

    # Set criteria to plot
    if criteria is None:
        criteria_list = eval_dict_list[0].keys()
    else:
        criteria_list = criteria

    # fig, ax = plt.subplots(len(criteria_list), len(output_names), dpi=150, sharex=True)
    if len(criteria_list) < 2:
        plt_sz = plot_size(fraction=1, height_fraction=0.4)
    else:
        plt_sz = plot_size(fraction=1, height_fraction=1.0)
    fig, ax = plt.subplots(len(criteria_list), len(output_names), figsize=plt_sz, sharex=True)

    if len(output_names) == 1:
        ax = np.array([ax]).T

    for o, ot in enumerate(output_names):
        # fig, ax = plt.subplots(len(eval_full), 1, dpi=250, sharex=True)
        for c, crit in enumerate(criteria_list):
            for i, scenario in enumerate(eval_dict_list):  # For each eval_dict in the list (each line in a figure)
                ax[c, o].plot(scenario['N_tp'], np.mean(scenario[crit][ot], axis=1), color=colors_scen[i],
                              label=label_list[i])
                box = ax[c, o].boxplot(scenario[crit][ot].T, positions=scenario['N_tp'], patch_artist=True,
                                       showmeans=False, showfliers=False)
                for key in box.keys():
                    if key != 'means':  # Exclude modifying mean lines
                        for element in box[key]:
                            if key == 'boxes':
                                element.set(facecolor='none', linewidth=1,
                                            edgecolor=colors_scen[i])  # Set box edge color, no fill
                            elif key == 'whiskers' or key == 'caps':
                                element.set(color=colors_scen[i],
                                            linewidth=1)  # Set line color and width for whiskers and caps
                            else:
                                element.set(color=colors_scen[i], linewidth=1)
                if plot_loc:
                    for i in range(n_loc):
                        ax[c, o].plot(scenario['N_tp'], scenario[crit][ot][:, i], color='gray', alpha=0.2)

            ax[c, o].plot(scenario['N_tp'], np.full(len(scenario['N_tp']), best_vals[crit]), color='red',
                          linestyle='--', label='best value')
            ax[c, o].grid()

            ax[c, 0].set_ylabel(crit)
            if x_label is None:
                ax[-1, o].set_xlabel('Number of training points')
            else:
                ax[-1, o].set_xlabel(f'{x_label}')
            ax[0, o].set_title(f'{ot}', loc='left')

        if len(eval_dict_list) > 1:
            handles, labels = ax[0, 0].get_legend_handles_labels()
            fig.legend(handles=handles, labels=labels, loc="center right", ncol=1)
        fig.suptitle(f'{fig_title}')
        plt.subplots_adjust(top=0.9, bottom=0.15, wspace=0.2, hspace=0.2)