"""
Program trains an arbitraryPCE (aPCE) for the synthetic non-linear model.

Can use normal training (once) or sequential training (BAL, SF, Sobol). It can use a GP as an error model, or note.

TODO: Bayesian inference is missing in evaluation
"""

from pathlib import Path
import sys
import pickle
import warnings

par_path = Path.cwd().parent.parent  # Main directory
sys.path.append(str(par_path))
sys.path.append(str(par_path / 'src'))

sys.path.append(str(par_path / 'src/surrogate_modelling'))
sys.path.append(str(par_path / 'src/utils'))

from analytical_model import nonlinear_model

from src.surrogate_modelling.bal_functions import BayesianInference, SequentialDesign
from src.surrogate_modelling.apce_tests import PCEConfig, aPCE, validation_error, save_valid_criteria
from src.surrogate_modelling.gpe_skl import SklTraining, RBF
from src.surrogate_modelling.inputs import Input
from src.surrogate_modelling.exp_design_ import ExpDesign

from src.plots.plots_convergence import *
from src.plots.plots_1d_2d import *
from src.plots.plots_validation import *

from src.utils.log import *

warnings.filterwarnings("ignore")
matplotlib.use("Qt5Agg")

if __name__ == '__main__':

    # =====================================================
    # =============   INPUT DATA  ================
    # =====================================================


    # paths ..........................................................................
    results_path = Path('Results/apce/solvers/')

    # surrogate data .................................................................
    parallelize = False  # to parallelize training, BAL

    pce_optimizer = 'FastARD'  # optimizer for pce coefficients:
    degree = 5  # maximum pce order
    use_gpe = False  # True to add a GPE as an error model

    # =====================================================
    # =============   COMPUTATIONAL MODEL  ================
    # =====================================================
    # Full-complexity model data. In this section, everything related to the full-complexity  model,
    # including everything that is **constant** should be initialized here

    input_data_path = None

    n_loc = 2  # number of output locations
    ndim = 3  # number of parameters
    output_names = ['Z']  # output names

    pt_loc = np.arange(0, n_loc, 1.) / (n_loc - 1)  # location of ouput locations  (could be read from a )

    # True data: Data to be used as observed or synthetic truth, for Bayesian inference purposes
    obs = np.full((1, n_loc), [2.])  # observation values, for each loc
    error_pp = np.repeat([2], n_loc)

    # Reference data:
    """Normally the reference data would be read from a file and imported here"""

    # =====================================================
    # =============   EXPERIMENT DESIGN  ================
    # =====================================================
    # Probabilistic model input: ....................................................
    # Define the uncertain parameters with their mean and standard deviation
    Inputs = Input()

    for i in range(ndim):
        Inputs.add_marginals()
        Inputs.Marginals[i].name = "$\\theta_{" + str(i + 1) + "}$"
        Inputs.Marginals[i].dist_type = 'uniform'
        Inputs.Marginals[i].parameters = [-5, 5]

    # Experimental design: ....................................................................
    # Everything related to the experimental design, including what type of *surrogate model* to use,
    # the training method, initial and maximum number of training point, etc. (see below for mode detail)

    exp_design = ExpDesign(input_object=Inputs,
                           exploit_method='space_filling',  # bal, space_filling, sobol
                           explore_method='random',  # method to sample from parameter set for active learning
                           training_step=1,  # No. of training points to sample in each iteration
                           sampling_method='sobol',  # how to sample the initial training points
                           main_meta_model='apce',  # main surrogate method: 'gpr' or 'apce'
                           n_initial_tp=50,  # Number of initial training points (min = n_trunc*2)
                           n_max_tp=50,  # max number of tp to use
                           training_method='normal',  # normal (train only once) or sequential (Active Learning)
                           util_func='global_mc',  # criteria for bal (dkl, bme, ie, dkl_bme) or SF (default: global_mc)
                           eval_step=1,  # every how many iterations to evaluate the surrogate
                           secondary_meta_model=False  # only gpr is available as a secondary model
                           )

    exp_design.setup_ED_()

    print(f'<<< Will run <{exp_design.n_iter + 1}> GP training iterations and <{exp_design.n_evals}> GP evaluations. >>>')

    # Setup surrogate model

    if exp_design.main_meta_model == 'apce':
        raw_data = exp_design.generate_samples(n_samples=100_000, sampling_method='random')
        pce_config = PCEConfig(max_degree=degree, prior_samples=raw_data, q_norm=0.9)
        pce_config.setup_apce()

    # Create folder for specific case ....................................................................
    results_folder = results_path / f'ndim_{ndim}_nout_{n_loc}'
    if not results_folder.exists():
        logger.info(f'Creating folder {results_folder}')
        Path.mkdir(results_folder)

    # Create a folder for the exploration method:
    # results_folder = results_folder / f'{exp_design.exploit_method}_{exp_design.util_func}'    # name for diff BAL
    results_folder = results_folder / f'solver_{pce_optimizer}'  # name for diff solvers
    if not results_folder.exists():
        logger.info(f'Creating folder {results_folder}')
        Path.mkdir(results_folder)

    # =====================================================
    # =============   COLLOCATION POINTS  ================
    # =====================================================
    collocation_points = exp_design.generate_samples(n_samples=exp_design.n_init_tp,
                                                     sampling_method=exp_design.sampling_method)
    model_evaluations = nonlinear_model(params=collocation_points, loc=pt_loc)

    # =====================================================
    # =============   Validation points  ================
    # =====================================================

    # Reference data .......................................................
    prior = exp_design.generate_samples(n_samples=10_000)
    ref_output = nonlinear_model(params=prior, loc=pt_loc)

    prior_logpdf = np.log(exp_design.JDist.pdf(prior.T)).reshape(-1)
    ref_scores = BayesianInference(model_predictions=ref_output,
                                   observations=obs, error=error_pp ** 2,
                                   sampling_method='rejection_sampling',
                                   prior=prior,
                                   prior_log_pdf=prior_logpdf)
    ref_scores.estimate_bme()

    # Validation ....................................................................
    valid_samples = exp_design.generate_samples(n_samples=100)
    model_valid_output = nonlinear_model(params=valid_samples, loc=pt_loc)

    exp_design.val_x = valid_samples
    exp_design.val_y = model_valid_output

    # --------------------------------------------------------------------------------------------------------------- #

    # =====================================================
    # =============   INITIALIZATION  ================
    # =====================================================

    bayesian_dict = {'N_tp': np.zeros(exp_design.n_iter + 1),
                     'BME': np.zeros(exp_design.n_iter + 1),  # To save BME value for each GPE, after training
                     'ELPD': np.zeros(exp_design.n_iter + 1),
                     'RE': np.zeros(exp_design.n_iter + 1),  # To save RE value for each GPE, after training
                     'IE': np.zeros(exp_design.n_iter + 1),
                     'post_size': np.zeros(exp_design.n_iter + 1),
                     f'{exp_design.exploit_method}_{exp_design.util_func}': np.zeros(exp_design.n_iter),
                     'util_func': np.empty(exp_design.n_iter, dtype=object)}

    eval_dict = {}
    gp_dict = {}

    # =====================================================
    # =============   SURROGATE MODEL TRAINING  ===========
    # =====================================================

    for it in range(0, exp_design.n_iter + 1):  # Train the aPCE a maximum of "iteration_limit" times
        # 1. Setup and train
        sm = aPCE(collocation_points=collocation_points, model_evaluations=model_evaluations,
                  pce_config=pce_config,
                  sparsity=True,
                  variance_cutoff=0,
                  pce_reg_method=pce_optimizer)

        sm.train_(initial_reg_method='BRR')
        # sm.train_with_retrain_(initial_reg_method='BRR')

        # 1.1 Error model:
        if exp_design.secondary_model:
            diff = np.subtract(model_evaluations, sm.surrogate_output)
            kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            # 1.2. Setup a PCE
            em = SklTraining(collocation_points=collocation_points, model_evaluations=diff,
                             noise=True,
                             kernel=kernel,
                             alpha=0.0002,
                             n_restarts=10,
                             parallelize=parallelize)
            em.train_()

        # 2. Validate GPR
        if it % exp_design.eval_step == 0 or it == exp_design.n_iter:

            valid_sm = sm.predict_(input_sets=exp_design.val_x, get_conf_int=True)
            if exp_design.secondary_model:
                valid_em = em.predict_(exp_design.val_x, get_conf_int=False)
                valid_sm['output'] = valid_sm['output'] + valid_em['output']
                valid_sm['std'] = valid_sm['std'] + valid_em['std']

            rmse_sm, run_valid = validation_error(true_y=exp_design.val_y, sim_y=valid_sm,
                                                  n_per_type=n_loc, output_names=output_names)

            eval_dict = save_valid_criteria(new_dict=run_valid, old_dict=eval_dict,
                                            n_tp=collocation_points.shape[0])

            # Save GP after evaluation ...................
            if exp_design.secondary_model:
                sname = 'apce-error'
            else:
                sname = 'apce'

            save_name = results_folder / f'{sname}_{sm.pce_reg_method}_TP{collocation_points.shape[0]:02d}_q{pce_config.q_norm}_varcut{sm.var_cutoff}_{exp_design.exploit_method}.pickle'
            sm.Exp_Design = exp_design
            pickle.dump(sm, open(save_name, "wb"))

            # Save eval_criteria dictionaries .........................
            save_name = results_folder / f'Validation_{sname}_{sm.pce_reg_method}_TP{collocation_points.shape[0]:02d}_q{pce_config.q_norm}_varcut{sm.var_cutoff}_{exp_design.exploit_method}.pickle.pickle'
            pickle.dump(eval_dict, open(save_name, "wb"))
            print('saving')

        # 3. Compute Bayesian scores in parameter space ----------------------------------------------------------
        surrogate_output = sm.predict_(input_samples=prior, get_conf_int=True)
        if exp_design.secondary_model:
            error_output = em.predict_(input_sets=prior, get_conf_int=False)
            surrogate_output['output'] = surrogate_output['output'] + error_output['output']
            surrogate_output['std'] = surrogate_output['std'] + error_output['std']

        total_error = (error_pp ** 2)
        bi_gpe = BayesianInference(model_predictions=surrogate_output['output'], observations=obs, error=total_error,
                                   sampling_method='rejection_sampling', prior=prior, prior_log_pdf=prior_logpdf)
        bi_gpe.estimate_bme()
        bayesian_dict['N_tp'][it] = collocation_points.shape[0]
        bayesian_dict['BME'][it], bayesian_dict['RE'][it] = bi_gpe.BME, bi_gpe.RE
        bayesian_dict['ELPD'][it], bayesian_dict['IE'][it] = bi_gpe.ELPD, bi_gpe.IE
        bayesian_dict['post_size'][it] = bi_gpe.post_likelihood.shape[0]

        # 4. Sequential Design --------------------------------------------------------------------------------------
        if it < exp_design.n_iter:
            # logger.info(f'Selecting {exp_design.training_step} additional TP using {exp_design.exploit_method}')
            SD = SequentialDesign(exp_design=exp_design, sm_object=sm, obs=obs, errors=error_pp ** 2,
                                  do_tradeoff=False, multiprocessing=parallelize)

            # SD.run_sequential_design(prior_samples=prior_samples)
            new_tp, util_fun = SD.run_sequential_design()

            bayesian_dict[f'{exp_design.exploit_method}_{exp_design.util_func}'][it] = SD.selected_criteria[0]
            bayesian_dict['util_func'][it] = util_fun

            # Evaluate model in new TP .............................................................................
            new_output = nonlinear_model(params=new_tp, loc=pt_loc)

            # Update collocation points:
            if exp_design.exploit_method == 'sobol':
                collocation_points = new_tp
                model_evaluations = new_output
            else:
                collocation_points = np.vstack((collocation_points, new_tp))
                model_evaluations = np.vstack((model_evaluations, new_output))

            # Plot process
            # if it % exp_design.eval_step == 0:
            #     if ndim == 1:
            #         if SD.do_tradeoff:
            #             plot_1d_bal_tradeoff(prior, surrogate_output['output'], surrogate_output['lower_ci'],
            #                                     surrogate_output['upper_ci'], SD.candidates,
            #                                     SD, collocation_points, model_evaluations, it, obs)
            #         else:
            #             plot_1d_gpe_bal(prior, surrogate_output['output'], surrogate_output['lower_ci'],
            #                             surrogate_output['upper_ci'], SD.candidates,
            #                             SD.total_score, collocation_points, model_evaluations, it, obs)
            #     if ndim < 3:
            #         plot_likelihoods(prior, bi_gpe.likelihood, prior, ref_scores.likelihood,
            #                             n_iter=it + 1)

            #     stop=1
        logger.info(f'------------ Finished iteration {it + 1}/{exp_design.n_iter} -------------------')

    # ## Evaluations ..............................................................................................

    # Plots for 1D - 2D examples
    if ndim == 1:
        # sn = save_folder / f'FinalSurrogate_ndim{ndim}_nout{n_loc}_{exp_design.gpr_lib}_{exp_design.exploit_method}_{exp_design.util_func}.png'
        plot_1d_gpe_final(prior, surrogate_output['output'],
                          surrogate_output['lower_ci'], surrogate_output['upper_ci'],
                          collocation_points, model_evaluations,
                          exp_design.n_init_tp, exp_design.n_iter, obs)

    if ndim < 3:
        # sn = results_folder / f'Likelihood_ndim{ndim}_nout{n_loc}_{exp_design.gpr_lib}_{exp_design.exploit_method}_{exp_design.util_func}'
        plot_likelihoods(prior, bi_gpe.likelihood, prior, ref_scores.likelihood,
                         n_iter=exp_design.n_iter, save_name=None)
        if exp_design.exploit_method == 'bal' or exp_design.exploit_method == 'space_filling':
            # sn = results_folder / f'SelectedTP_ndim{ndim}_nout{n_loc}_{exp_design.gpr_lib}_{exp_design.exploit_method}_{exp_design.util_func}.png'
            plot_combined_bal(collocation_points=collocation_points, n_init_tp=exp_design.n_init_tp,
                              bayesian_dict=bayesian_dict, save_name=None)

    # ### Plot convergence ..........................................................................
    # Plots PCE coefficient information

    # %%
    fig, ax = plt.subplots(1, len(output_names))
    if len(output_names) == 1:
        ax = np.array([ax]).T

    c = 0
    for o, ot in enumerate(output_names):
        for i in range(n_loc):
            descending_idx = (-(sm.pce_list[i + c]['full_coeffs'] ** 2)).argsort()  # index in descending order
            var_ordered = sm.pce_list[i + c]['full_coeffs'][descending_idx] ** 2  # arrange coefficients in desc order
            var_sum = np.cumsum(var_ordered) / np.sum(var_ordered)

            ax[o].plot(np.arange(1, sm.pce_list[i + c]['coeffs'].shape[0] + 1), var_sum, label=f'loc{i + 1}', marker='x')

        ax[o].xaxis.set_ticks(np.arange(1, sm.pce_list[i + c]['coeffs'].shape[0] + 1))
        ax[o].grid()
        ax[o].set_ylabel('Variance')
        ax[o].set_xlabel('Number of Coefficients')

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="center right", ncol=1)
    fig.suptitle(f'')
    plt.subplots_adjust(top=0.9, bottom=0.15, wspace=0.2, hspace=0.2)

    # %% [markdown]
    # ### Plot validation results ...........................................................................
    # Validation for final trained surrogate
    surrogate_output = sm.predict_(input_samples=prior, get_conf_int=True)
    plot_correlation(sm_out=valid_sm['output'], valid_eval=exp_design.val_y,
                     output_names=output_names,
                     label_list=[f"{np.mean(eval_dict['r2']['Z'][-1, :]):0.3f}"],
                     fig_title=f'Outputs: TP{sm.training_points.shape[0]}', n_loc_=n_loc)

    # validation criteria at each output location ("run_valid" is the evaluation criteria for the final
    # trained surrogate)
    plot_validation_loc(eval_dict_list=[run_valid], label_list=[''],
                        output_names=output_names, n_loc=n_loc,
                        criteria=['mse', 'nse', 'r2', 'mean_error', 'std_error'],
                        fig_title=f'Validation criteria: TP: {exp_design.n_max_tp}')

    plot_validation_loc(eval_dict_list=[run_valid], label_list=[''],
                        output_names=output_names, n_loc=n_loc,
                        criteria=['norm_error', 'P95'],
                        fig_title=f'Validation criteria with error consideration: TP: {exp_design.n_max_tp}')

    # %%
    plot_validation_tp(eval_dict_list=[eval_dict], label_list=[''],
                       output_names=output_names, n_loc=n_loc,
                       criteria=['mse', 'nse', 'mean_error', 'std_error'], plot_loc=True,
                       fig_title='Validation criteria for increasing number of TP')

    # ## Plot Bayesian results ....................................................................................

    # Parameter distributions for final run
    fig, ax = plt.subplots(2, ndim, sharex=True)

    for i in range(ndim):
        ax[0, i].hist(prior[:, i], alpha=1.0)
        ax[1, i].hist(bi_gpe.posterior[:, i], color='blue', alpha=0.5)
        ax[1, i].hist(ref_scores.posterior[:, i], color='red', alpha=0.8)

    stop = 1
