"""
Plots for 1D or 2D surrogates, where we can visually plot the results.
"""

import math

from src.plots.plots_config import *


def plot_1d_gpe_bal(gpr_x, gpr_y, gpr_std, bal_x, bal_y, tp_x, tp_y, it, true_y, true_x=None,
                    analytical=None):
    """
        Plots the GPE run + BAL step, including GPE predictions, training points and confidence intervals.Additionally, it
        plots the RE for each parameter output explored and the new training point location
        posterior output space of 2 points, from a Gaussian distribution.
        Parameters
        ----------
        :param gpr_x: np.array [mc_size, 1], parameter values where GPE was evaluated (X)
        :param gpr_y: np.array [mc_size, n_obs], GPE prediction values (Y)
        :param gpr_std: np.array [mc_size, n_obs], GPE standard deviation values
        :param bal_x: np.array [d_size, 1], parameter values explored in BAL
        :param bal_y: np.array [d_size, 1], BAL criteria values obtained for each exploration point in bal_x
        :param tp_x: np.array [#tp+1, 1], parameter values from training points
        :param tp_y: np.array [#tp+1, N.Obs], forward model outputs at tp_x
        :param true_y: np.array [1, N.Obs] Y value of the observation value
        :param true_x: np.array [1, 1] true parameter value or None (if not available)
        :param analytical: np.array [2, mc_size], x and y values evaluated in forward model or None (to not plot it)
        :param it: int , number of iteration, for plot title
        """
    y_items = gpr_y.items()
    # bal_y_mod = np.nan_to_num(bal_y, nan=0)
    bal_y_mod = bal_y

    for key, y_gp in y_items:
        n_o = y_gp.shape[1]

        if n_o == 10:
            rows = int(n_o/2)
            cols = 2
        elif n_o == 1:
            rows = 1
            cols = 1
        else:
            rows = n_o
            cols = 1

        fig, ax = plt.subplots(rows, cols, sharex=True)

        if n_o > 1:
            loop_item = ax.reshape(-1)
        else:
            loop_item = np.array([ax])

        # get confidence intervals:
        conf_int_upper = y_gp + (1.96 * gpr_std[key])
        conf_int_lower = y_gp - (1.96 * gpr_std[key])

        for o, ax_i in enumerate(loop_item):
            # Get overall limits:
            # GPR limits:
            lims_gpr = np.array([math.floor(np.min(conf_int_lower)), math.ceil(np.max(conf_int_upper))])
            # Get limits
            lims_x = [math.floor(np.min(gpr_x)), math.ceil(np.max(gpr_x))]

            if analytical is not None:
                # order analytical evaluations
                a_data = analytical[:, analytical[0, :].argsort()]
                # Analytical limits
                lims_an = np.array([math.floor(np.min(analytical[1, :])), math.ceil(np.max(analytical[1, :]))])
                # Max limit is defined by gpr and analytical
                lims_y = [np.min(np.array([lims_gpr[0], lims_an[0]])), np.max(np.array([lims_gpr[1], lims_an[1]]))]
            else:
                lims_y = lims_gpr

            # GPE data ---------------------------------------------------------------
            data = np.vstack((gpr_x[:, 0], y_gp[:, o], conf_int_lower[:, o], conf_int_upper[:, o]))
            gpr_data = data[:, data[0, :].argsort()]

            ax_i.plot(gpr_data[0, :], gpr_data[1, :], label="GPE mean", linewidth=1, color='b', zorder=1)
            ax_i.fill_between(gpr_data[0, :].ravel(), gpr_data[2, :], gpr_data[3, :], alpha=0.5,
                              label=r"95% confidence interval")

            # TP ---------------------------------------------------------------------
            # Find GPR result for max RE
            # max_score = np.argmax(bal_y)
            # ind = np.where(gpr_x == tp_x[-1])[0][0]
            # newp = np.array([gpr_x[ind], gpr_y[o, ind]], dtype=object)

            ax_i.scatter(tp_x[:-1], tp_y[key][:-1, o], label="TP", color="b", zorder=2)  # s=500
            ax_i.scatter(tp_x[-1], tp_y[key][-1, o], color="r", marker="x", zorder=2, linewidths=3)  # s=800
            # ax_i.scatter(newp[0], newp[1], color="r", marker="x", zorder=2, linewidths=3)  # s=800

            # Plot observation ----------------------------------------------------------------
            if true_x is not None:
                ax_i.scatter(true_x[0, 0], true_y[key][0, o], marker="*", color="g", zorder=2)  # s=500
            else:
                ax_i.plot([np.min(gpr_x), np.max(y_gp)], [true_y[key][0, o], true_y[key][0, o]], color="g", linestyle="dotted")

            # Plot analytical data
            if analytical is not None:
                ax.plot(a_data[0, :], a_data[1, :], color='k', linewidth=1, linestyle="dashed", label="f(x)", zorder=1)

            # Legends
            handles, labels = ax_i.get_legend_handles_labels()

            # BAL criteria -------------------------------------------------------------------------------
            # Order Criteria (RE) results

            data = np.vstack((bal_x.T, bal_y_mod.T))
            re_data = data[:, data[0, :].argsort()]

            ax2 = ax_i.twinx()
            ax2.plot(re_data[0, :], re_data[1, :], color="orange", linewidth=2)
            # ax2.fill_between(re_data[1, :]-1, re_data[1, :], alpha=0.4, color="orange", label="SD criteria")

            h, lab = ax2.get_legend_handles_labels()
            handles = handles + h
            labels = labels + lab

            # Set limits --------------------------------------------------------------------------------
            ax_i.set_xlim(lims_x)
            ax_i.set_ylim([lims_y[0] - (0.5 * (lims_y[1] - lims_y[0])), lims_y[1] * 1.1])

            non_nan_mask = ~np.isnan(bal_y)
            if np.all(bal_y[non_nan_mask] <= 0):
                x = np.nanmin(bal_y)
                ax2.set_ylim([np.nanmin(bal_y), np.nanmin(bal_y) - np.nanmin(bal_y)* 1])
            else:
                ax2.set_ylim([0, np.nanmax(bal_y) * 3])
            # ax.set_xlim([0, 10])
            # ax2.set_ylim([0, 8])
            # ax.set_ylim([-15, np.max(gpr_data)+10])

            # set labels ------------------------------------------------------------------------------------
            # ax_i.set_xlabel("Parameter value")
            # ax_i.set_ylabel("Y")
            # ax2.set_ylabel("BAL Criteria")
            # ax.axes.xaxis.set_visible(False)
            # ax.axes.yaxis.set_visible(False)
            # ax2.axes.yaxis.set_visible(False)

        if n_o > 4:
            fig.text(0.5, 0.1, "Parameter value", ha='center')
            fig.text(0.04, 0.5, "Model output", ha='center', rotation='vertical')
            fig.text(0.96, 0.5, "BAL Criteria", ha='center', rotation='vertical')
        elif n_o == 1:
            ax_i.set_xlabel("Parameter value")
            ax_i.set_ylabel("Y")
            ax2.set_ylabel("BAL Criteria")

        fig.suptitle(f'Iteration={it + 1}')
        fig.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.18, wspace=0.3, hspace=0.3)
        fig.legend(handles, labels, loc='lower center', ncol=5)
        plt.show(block=False)


def plot_1d_gpe_final(gpr_x, gpr_y, gpr_std, tp_x, tp_y, tp_init, it, true_y, true_x=None,
                      analytical=None, save_name=None):
    """

    Args:
        gpr_x: np.array [mc_size, 1]
            parameter values where surrogate was evaluated (X)
        gpr_y: [mc_size, n_obs]
            GPE prediction values (Y) when evaluated at gpr_x
        gpr_std: [mc_size, n_obs]
            GPE standard deviation values
        tp_x: np.array [n_tp, 1]
            training point values
        tp_y: array [n_tp, n_obs]
            Simulator outputs for each tp_x
        tp_init: int
            Number of initial training points
        it: int
            iteration number
        true_y: array [1, n_obs]
            Measured output values.
        true_x: array [1,1]
            true parameter values. default is None, so it is not included in the plot.
        analytical: array [2, n_mc]
            x and y values evaluated in the simulator. Default is None, so it is not included
        save_name: Path
            Full file name with which to save the resulting figure

    Returns:

    """

    y_items = gpr_y.items()

    for key, y_gp in y_items:

        n_o = y_gp.shape[1]
        del_last = False
        if n_o == 1:
            rows = 1
            cols = 1
        elif (n_o % 2) == 0:
            rows = int(n_o/2)
            cols = 2
        elif (n_o % 3) == 0:
            rows = int(n_o / 3)
            cols = 3
        else:
            rows = math.ceil(n_o / 2)
            cols = 2
            del_last = True

        # get confidence intervals:
        conf_int_upper = y_gp + (1.96 * gpr_std[key])
        conf_int_lower = y_gp - (1.96 * gpr_std[key])

        fig, ax = plt.subplots(rows, cols, sharex=True)
        if del_last:
            ax[rows-1, cols-1].remove()

        if n_o > 1:
            loop_item = ax.reshape(-1)
        else:
            loop_item = np.array([ax])

        for o, ax_i in enumerate(loop_item):
            # Get overall limits:
            # GPR limits:
            lims_gpr = np.array([math.floor(np.min(conf_int_lower[:, o])), math.ceil(np.max(conf_int_upper[:, o]))])
            # Get limits
            lims_x = [math.floor(np.min(gpr_x)), math.ceil(np.max(gpr_x))]

            if analytical is not None:
                # order analytical evaluations
                a_data = analytical[key][:, analytical[key][0, :].argsort()]
                # Analytical limits
                lims_an = np.array([math.floor(np.min(analytical[1, :])), math.ceil(np.max(analytical[1, :]))])
                # Max limit is defined by gpr and analytical
                lims_y = [np.min(np.array([lims_gpr[0], lims_an[0]])), np.max(np.array([lims_gpr[1], lims_an[1]]))]
            else:
                lims_y = lims_gpr

            # GPE data ---------------------------------------------------------------
            try:
                data = np.vstack((gpr_x[:, 0], y_gp[:, o], conf_int_lower[:, o], conf_int_upper[:, o]))
                gpr_data = data[:, data[0, :].argsort()]

                ax_i.plot(gpr_data[0, :], gpr_data[1, :], label="GPE mean", linewidth=1, color='b', zorder=1)
                ax_i.fill_between(gpr_data[0, :].ravel(), gpr_data[2, :], gpr_data[3, :], alpha=0.5,
                                  label=r"95% confidence interval")
            except:
                break

            # TP ---------------------------------------------------------------------
            # Color map:
            cm = plt.cm.get_cmap('YlGn')
            cls = []
            tot = tp_x.shape[0]-tp_init+1
            for i in range(1, tp_x.shape[0]-tp_init+1):
                cls.append(cm(i/tot))

            ax_i.scatter(tp_x[0:tp_init], tp_y[key][0:tp_init, o], label="TP", color="k", zorder=2)  # s=500
            ax_i.scatter(tp_x[tp_init:], tp_y[key][tp_init:, o], color=cls, marker="x", zorder=2, linewidths=3, label="BAL TP")  # s=800

            # Plot observation ----------------------------------------------------------------
            if true_x is not None:
                ax_i.scatter(true_x[0, 0], true_y[0, o], marker="*", color="g", zorder=2)  # s=500
            else:
                ax_i.plot([np.min(gpr_x), np.max(y_gp)], [true_y[key][0, o], true_y[key][0, o]], color="g", linestyle="dotted")

            # Plot analytical data
            if analytical is not None:
                ax.plot(a_data[0, :], a_data[1, :], color='k', linewidth=1, linestyle="dashed", label="f(x)", zorder=1)

            # Legends
            handles, labels = ax_i.get_legend_handles_labels()

            # Set limits --------------------------------------------------------------------------------
            ax_i.set_xlim(lims_x)
            ax_i.set_ylim([lims_y[0] - (0.5 * (lims_y[1] - lims_y[0])), lims_y[1] * 1.1])

            # set labels ------------------------------------------------------------------------------------
            # ax_i.set_xlabel("Parameter value")
            # ax_i.set_ylabel("Y")
            # ax.axes.xaxis.set_visible(False)
            # ax.axes.yaxis.set_visible(False)

        if n_o > 4:
            fig.text(0.5, 0.1, "Parameter value", ha='center')
            fig.text(0.04, 0.5, "Model output", ha='center', rotation='vertical')
        elif n_o == 1:
            ax_i.set_xlabel("Parameter value")
            ax_i.set_ylabel("Y")

        fig.suptitle(f'Iteration={it + 1}')
        fig.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.18, wspace=0.3, hspace=0.3)
        fig.legend(handles, labels, loc='lower center', ncol=5)
        if save_name is not None:
            plt.savefig(save_name)
        plt.show(block=False)


def plot_likelihoods(gpe_prior, gpe_likelihood, ref_prior, ref_likelihood, n_iter=None, save_name=None):
    """
    Plots 1D and 2D likelihoods for the GPE and the reference solution
    :param gpe_prior: np.array [# param sets, # parameters] with parameter sets used to evaluate final gpe
    :param gpe_likelihood: np.array[# param sets, ], with likelihood values for each parameter set evaluated
    :param ref_prior: np.array [# param sets, # parameters] with parameter sets used to evaluate the reference FM
    :param ref_likelihood: np.array[# param sets, ], with likelihood values for each parameter set evaluated in FM
    :param n_iter: int
        Iteration number
    :param save_name: Path object
        file path and name with which to save resulting figures
    :return: None

    Note: For 1D case, plots the likelihood in y, parameter value in x (line plot). In 2D, 3D scatter plot
    """
    if gpe_prior.shape[1] == 1:
        # order
        data = np.vstack((gpe_prior.T, gpe_likelihood.reshape(1, gpe_likelihood.shape[0])))
        gpe_data = data[:, data[0, :].argsort()]

        data2 = np.vstack((ref_prior[:, 0].T, ref_likelihood.reshape(1, ref_likelihood.shape[0])))
        ref_data = data2[:, data[0, :].argsort()]

        fig, ax = plt.subplots(figsize=plot_size())
        ax.plot(gpe_data[0, :], gpe_data[1, :], label="GPE likelihood", color='blue')
        ax.plot(ref_data[0, :], ref_data[1, :], label="Ref likelihood", color='black', linestyle="--")

        if n_iter is not None:
            fig.suptitle(f'Iteration <{n_iter}>')
        fig.legend(loc='lower center', ncol=2)
        plt.subplots_adjust(top=0.92, bottom=0.2, wspace=0.1, hspace=0.4)
        if save_name is not None:
            plt.savefig(f'{save_name}_iter{n_iter}_1D.png')
        plt.show(block=False)

        ind_2 = 0
    else:
        ind_2 = 1

    # --------------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=plot_size())

    # Set axis limits:
    max_gpe, min_gpe = np.max(gpe_likelihood), np.min(gpe_likelihood)
    max_ref, min_ref = np.max(ref_likelihood), np.min(ref_likelihood)
    vmax, vmin = np.max(np.array(max_gpe, max_ref)), np.min(np.array(min_gpe, min_ref))

    ax[0].scatter(gpe_prior[:, 0], gpe_prior[:, ind_2], c=gpe_likelihood, s=5, vmax=vmax, vmin=vmin, cmap='coolwarm',
                  marker='o')
    im = ax[1].scatter(ref_prior[:, 0], ref_prior[:, ind_2], c=ref_likelihood, s=5, vmax=vmax, vmin=vmin,
                       cmap='coolwarm', marker='o')

    # Labels
    ax[0].set_xlabel('$\omega_1$')
    ax[0].set_ylabel('$\omega_2$')
    ax[1].set_xlabel('$\omega_1$')
    ax[1].set_ylabel('$\omega_2$')
    ax[0].set_xlim(np.min(gpe_prior) - 1, np.max(gpe_prior) + 1)
    ax[0].set_ylim(np.min(gpe_prior) - 1, np.max(gpe_prior) + 1)
    ax[1].set_xlim(np.min(ref_prior) - 1, np.max(ref_prior) + 1)
    ax[1].set_ylim(np.min(ref_prior) - 1, np.max(ref_prior) + 1)

    ax[0].set_title("GPE")
    ax[1].set_title("Reference")

    if n_iter is not None:
        fig.suptitle(f'Iteration <{n_iter}>')

    cbar = fig.colorbar(im, ax=ax[1])
    cbar.set_label("Likelihood", labelpad=10)

    plt.subplots_adjust(top=0.89, bottom=0.1, wspace=0.25, hspace=0.55)
    plt.margins(y=1, tight=True)
    if save_name is not None:
        plt.savefig(f'{save_name}_iter{n_iter}_2D.png')
    plt.show(block=False)


def plot_combined_bal(collocation_points, n_init_tp, bayesian_dict, save_name=None):
    """
    Plots the initial training point and which points were selected using DKL and which ones using BME, when the
    combined utility function is chosen.
    Args:
        collocation_points: np.array[n_tp, n_param]
            Array with all collocation points, in order in which they were selected.
        n_init_tp: int
            Number of TP selected initially
        bayesian_dict: dictionary
            With keys 'util_function', which details which utility function was used in each iteration.
        save_name: Path file
            File name with which to save results. Default is None, so no file is saved.

    Returns:

    """

    if collocation_points.shape[1] == 1:
        collocation_points = np.hstack((collocation_points, collocation_points))

    fig, ax = plt.subplots()

    # initial TP:
    ax.scatter(collocation_points[0:n_init_tp, 0], collocation_points[0:n_init_tp, 1], label='InitialTP', c='black',
               s=100)
    selected_tp = collocation_points[n_init_tp:, :]

    # Get indexes for 'dkl'
    dkl_ind = np.where(bayesian_dict['util_func'] == 'dkl')
    ax.scatter(selected_tp[dkl_ind, 0], selected_tp[dkl_ind, 1], label='DKL', c='gold', s=200, alpha=0.5)

    # Get indexes for 'bme'
    bme_ind = np.where(bayesian_dict['util_func'] == 'bme')
    ax.scatter(selected_tp[bme_ind, 0], selected_tp[bme_ind, 1], label='BME', c='blue', s=200, alpha=0.5)

    # Get indexes for 'ie'
    ie_ind = np.where(bayesian_dict['util_func'] == 'ie')
    ax.scatter(selected_tp[ie_ind, 0], selected_tp[ie_ind, 1], label='BME', c='green', s=200, alpha=0.5)

    # Global MC
    ie_ind = np.where(bayesian_dict['util_func'] == 'global_mc')
    ax.scatter(selected_tp[ie_ind, 0], selected_tp[ie_ind, 1], label='MC', c='red', s=200, alpha=0.5)

    ax.set_xlabel('$\omega_1$')
    ax.set_ylabel('$\omega_2$')

    fig.legend(loc='lower center', ncol=5)
    plt.subplots_adjust(top=0.95, bottom=0.15, wspace=0.25, hspace=0.55)
    if save_name is not None:
        plt.savefig(save_name)
    plt.show(block=False)


def plot_1d_bal_tradeoff(gpr_x, gpr_y, conf_int_lower, conf_int_upper, bal_x, sd_object, tp_x, tp_y, it, true_y, true_x=None,
                    analytical=None):
    """
        Plots the GPE run + BAL step, including GPE predictions, training points and confidence intervals.Additionally,
        it plots the sequential design criteria for each parameter output explored and the new training point location
        posterior output space of 2 points, from a Gaussian distribution. This case is when a tradeoff between BAL and
        SF methods is used, so the contribution of each score is also plotted.
        Parameters
        ----------
        :param gpr_x: np.array [mc_size, 1], parameter values where GPE was evaluated (X)
        :param gpr_y: np.array [mc_size, n_obs], GPE prediction values (Y)
        :param conf_int_lower: np.array [mc_size, n_obs], 95% lower confidence intervals for each gpr_y from GPE
        :param conf_int_upper: np.array [mc_size, n_obs], 95% upper confidence intervals for each gpr_y from GPE
        :param bal_x: np.array [d_size, 1], parameter values explored in BAL
        :param sd_object: object, SequentialDesign object
        :param tp_x: np.array [#tp+1, 1], parameter values from training points
        :param tp_y: np.array [#tp+1, N.Obs], forward model outputs at tp_x
        :param true_y: np.array [1, N.Obs] Y value of the observation value
        :param true_x: np.array [1, 1] true parameter value or None (if not available)
        :param analytical: np.array [2, mc_size], x and y values evaluated in forward model or None (to not plot it)
        :param it: int , number of iteration, for plot title
        """
    n_o = gpr_y.shape[1]

    if n_o == 10:
        rows = int(n_o/2)
        cols = 2
    elif n_o == 1:
        rows = 1
        cols = 1
    else:
        rows = 3
        cols = 1

    fig, ax = plt.subplots(rows, cols, sharex=True)

    if n_o > 1:
        loop_item = ax.reshape(-1)
    else:
        loop_item = np.array([ax])

    for o, ax_i in enumerate(loop_item):
        # Get overall limits:
        # GPR limits:
        lims_gpr = np.array([math.floor(np.min(conf_int_lower)), math.ceil(np.max(conf_int_upper))])
        # Get limits
        lims_x = [math.floor(np.min(gpr_x)), math.ceil(np.max(gpr_x))]

        if analytical is not None:
            # order analytical evaluations
            a_data = analytical[:, analytical[0, :].argsort()]
            # Analytical limits
            lims_an = np.array([math.floor(np.min(analytical[1, :])), math.ceil(np.max(analytical[1, :]))])
            # Max limit is defined by gpr and analytical
            lims_y = [np.min(np.array([lims_gpr[0], lims_an[0]])), np.max(np.array([lims_gpr[1], lims_an[1]]))]
        else:
            lims_y = lims_gpr

        # GPE data ---------------------------------------------------------------
        data = np.vstack((gpr_x[:, 0], gpr_y[:, o], conf_int_lower[:, o], conf_int_upper[:, o]))
        gpr_data = data[:, data[0, :].argsort()]

        ax_i.plot(gpr_data[0, :], gpr_data[1, :], label="GPE mean", linewidth=1, color='b', zorder=1)
        ax_i.fill_between(gpr_data[0, :].ravel(), gpr_data[2, :], gpr_data[3, :], alpha=0.5,
                          label=r"95% confidence interval")

        # TP ---------------------------------------------------------------------
        # Find GPR result for max RE
        # max_score = np.argmax(bal_y)
        # ind = np.where(gpr_x == tp_x[-1])[0][0]
        # newp = np.array([gpr_x[ind], gpr_y[o, ind]], dtype=object)

        ax_i.scatter(tp_x[:-1], tp_y[:-1, o], label="TP", color="b", zorder=2)  # s=500
        ax_i.scatter(tp_x[-1], tp_y[-1, o], color="r", marker="x", zorder=2, linewidths=3)  # s=800
        # ax_i.scatter(newp[0], newp[1], color="r", marker="x", zorder=2, linewidths=3)  # s=800

        # Plot observation ----------------------------------------------------------------
        if true_x is not None:
            ax_i.scatter(true_x[0, 0], true_y[0, o], marker="*", color="g", zorder=2)  # s=500
        else:
            ax_i.plot([np.min(gpr_x), np.max(gpr_y)], [true_y[0, o], true_y[0, o]], color="g", linestyle="dotted")

        # Plot analytical data
        if analytical is not None:
            ax.plot(a_data[0, :], a_data[1, :], color='k', linewidth=1, linestyle="dashed", label="f(x)", zorder=1)

        # Legends
        handles, labels = ax_i.get_legend_handles_labels()

        # BAL criteria -------------------------------------------------------------------------------
        # Order Criteria (RE) results
        data = np.vstack((bal_x.T, sd_object.total_score.T, sd_object.exploit_score_norm.T, sd_object.explore_score.T))
        re_data = data[:, data[0, :].argsort()]

        ax2 = ax_i.twinx()
        ax2.plot(re_data[0, :], re_data[1, :], color="orange", linewidth=2)
        ax2.fill_between(re_data[0, :], re_data[1, :], alpha=0.4, color="orange", label="SD criteria")

        ax2.fill_between(re_data[0, :], 0.5*re_data[2, :], alpha=0.4, color="blue", label="BAL")
        ax2.fill_between(re_data[0, :], 0.5*re_data[3, :], alpha=0.4, color="green", label="SF")

        h, lab = ax2.get_legend_handles_labels()
        handles = handles + h
        labels = labels + lab

        # Set limits --------------------------------------------------------------------------------
        ax_i.set_xlim(lims_x)
        ax_i.set_ylim([lims_y[0] - (0.5 * (lims_y[1] - lims_y[0])), lims_y[1] * 1.1])
        ax2.set_ylim([0, np.max(sd_object.total_score) * 3])
        # ax.set_xlim([0, 10])
        # ax2.set_ylim([0, 8])
        # ax.set_ylim([-15, np.max(gpr_data)+10])

        # set labels ------------------------------------------------------------------------------------
        # ax_i.set_xlabel("Parameter value")
        # ax_i.set_ylabel("Y")
        # ax2.set_ylabel("BAL Criteria")
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        # ax2.axes.yaxis.set_visible(False)

    if n_o > 4:
        fig.text(0.5, 0.1, "Parameter value", ha='center')
        fig.text(0.04, 0.5, "Model output", ha='center', rotation='vertical')
        fig.text(0.96, 0.5, "BAL Criteria", ha='center', rotation='vertical')
    elif n_o == 1:
        ax_i.set_xlabel("Parameter value")
        ax_i.set_ylabel("Y")
        ax2.set_ylabel("BAL Criteria")

    fig.suptitle(f'Iteration={it + 1}')
    fig.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.18, wspace=0.3, hspace=0.3)
    fig.legend(handles, labels, loc='lower center', ncol=5)
    plt.show(block=False)
