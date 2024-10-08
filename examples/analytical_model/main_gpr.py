"""
Program trains a Gaussian process emulator (GPE) for the synthetic non-linear model. (Does not work for aPCE)

Can use normal training (once) or sequential training (BAL, SF, Sobol)
"""
import time
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
from src.plots.plots_1d_2d import plot_likelihoods, plot_combined_bal, plot_1d_gpe_bal, plot_1d_gpe_final

from src.surrogate_modelling.bal_functions import BayesianInference, SequentialDesign
from src.surrogate_modelling.gpe_skl import *
from src.surrogate_modelling.gpe_gpytorch import *
# from src.surrogate_modelling.apce import *
from src.surrogate_modelling.inputs import Input
from src.surrogate_modelling.exp_design_ import ExpDesign

from src.plots.plots_convergence import *
# from src.plots.plots_1d_2d import *
from src.plots.plots_validation import *

from src.utils.log import *

warnings.filterwarnings("ignore")
matplotlib.use("Qt5Agg")

if __name__ == '__main__':

    # =====================================================
    # =============   INPUT DATA  ================
    # =====================================================
    # paths ..........................................................................
    results_path = Path('Results')  # Folder where to save results

    # surrogate data .................................................................
    parallelize = False   # to parallelize surrogate training, BAL
    # gpr_data
    gp_library = 'skl'    # gpy: GPyTorch or skl: Scikit-Learn

    # ===============================================
    # ============= COMPUTATIONAL MODEL: ============
    # ===============================================
    input_data_path = None

    n_loc = 10   # number of output locations (Number of surrogates to train)
    ndim = 2     # number of parameters
    output_names = ['Z']    # Name for the different type of output types (here only 1)

    pt_loc = np.arange(0, n_loc, 1.) / (n_loc - 1)
    # pt_loc = np.array([0.1111])

    # Synthetic true data .......................................
    obs = {'Z': np.full((1, n_loc), [2.])}  # observation values, for each loc
    error_pp = {'Z': np.repeat([2.**2], n_loc)}

    # Reference data:
    """Normally the reference data would be read from a file and imported here"""

    # =====================================================
    # =============   EXPERIMENT DESIGN  ================
    # =====================================================

    # Probabilistic model input: ....................................................
    # Define the uncertain parameters with their mean and standard deviation
    Inputs = Input()

    # One "Marginal" for each parameter.
    for i in range(ndim):
        Inputs.add_marginals()   # Create marginal for parameter "i"
        Inputs.Marginals[i].name = "$\\theta_{" + str(i + 1) + "}$"    # Parameter name
        Inputs.Marginals[i].dist_type = 'uniform'    # Parameter distribution (see exp_design.py --> build_dist()
        Inputs.Marginals[i].parameters = [-3, 3]                       # Inputs needed for districution

    # Experimental design: ....................................................................

    exp_design = ExpDesign(input_object=Inputs,
                           exploit_method='bal',  # bal, space_filling, sobol
                           explore_method='random',  # method to sample from parameter set for active learning
                           training_step=1,  # No. of training points to sample in each iteration
                           sampling_method='sobol',  # how to sample the initial training points
                           main_meta_model=gp_library,  # main surrogate method: 'gpr' or 'apce'
                           n_initial_tp=50,  # Number of initial training points (min = n_trunc*2)
                           n_max_tp=50,  # max number of tp to use
                           training_method='sequential',  # normal (train only once) or sequential (Active Learning)
                           util_func='dkl',  # criteria for bal (dkl, bme, ie, dkl_bme) or SF (default: global_mc)
                           eval_step=1,  # every how many iterations to evaluate the surrogate
                           secondary_meta_model=False   # only gpr is available
                           )

    exp_design.setup_ED_()

    print(
        f'<<< Will run <{exp_design.n_iter + 1}> GP training iterations and <{exp_design.n_evals}> GP evaluations. >>>')

    # =====================================================
    # =============   COLLOCATION POINTS  ================
    # =====================================================

    # Collocation points ...................................................
    # This part is specific to the problem: here you add the functions to create input sets and evaluate the model/read
    # already-run model runs.
    collocation_points = exp_design.generate_samples(n_samples=exp_design.n_init_tp,
                                                     sampling_method=exp_design.sampling_method)
    model_evaluations = nonlinear_model(params=collocation_points, loc=pt_loc, as_dict=True)
    """Note: The collocation points and model evaluations should be in numpy-array form, and in the order needed by 
    the gpe_skl.py or gpe_gpytorch.py classes"""

    # =====================================================
    # =============   Validation points  ================
    # =====================================================

    # Reference data .......................................................
    prior = exp_design.generate_samples(n_samples=10_000)
    ref_output = nonlinear_model(params=prior, loc=pt_loc, as_dict=True)

    prior_logpdf = np.log(exp_design.JDist.pdf(prior.T)).reshape(-1)
    ref_scores = BayesianInference(model_predictions=ref_output,
                                   observations=obs,
                                   error=error_pp,
                                   sampling_method='rejection_sampling',
                                   prior=prior,
                                   prior_log_pdf=prior_logpdf)

    ref_scores.do_inference()

    # Validation_data .....................................................
    valid_samples = exp_design.generate_samples(n_samples=100)
    model_valid_output = nonlinear_model(params=valid_samples, loc=pt_loc, as_dict=True)

    exp_design.val_x = valid_samples
    exp_design.val_y = model_valid_output

    # --------------------------------------------------------------------------------------------------------------- #

    # =====================================================
    # =============   INITIALIZATION  ================
    # =====================================================
    """This part can be ommited, if the files can be saved directly in the results_path folder"""
    # Create folder for specific case ....................................................................
    results_folder = results_path / f'ndim_{ndim}_nout_{n_loc}'
    if not results_folder.exists():
        logger.info(f'Creating folder {results_folder}')
        Path.mkdir(results_folder)

    # Create a folder for the exploration method: needed only if BAL is to be used
    results_folder = results_folder / f'{exp_design.exploit_method}_{exp_design.util_func}'
    if not results_folder.exists():
        logger.info(f'Creating folder {results_folder}')
        Path.mkdir(results_folder)

    # Arrays to save results ---------------------------------------------------------------------------- #

    bayesian_dict = {'N_tp': np.zeros(exp_design.n_iter + 1),
                     'BME': np.zeros(exp_design.n_iter + 1),  # To save BME value for each GPE, after training
                     'ELPD': np.zeros(exp_design.n_iter + 1),
                     'RE': np.zeros(exp_design.n_iter + 1),  # To save RE value for each GPE, after training
                     'IE': np.zeros(exp_design.n_iter + 1),
                     'post_size': np.zeros(exp_design.n_iter + 1),
                     f'{exp_design.exploit_method}_{exp_design.util_func}': np.zeros(exp_design.n_iter),
                     'util_func': np.empty(exp_design.n_iter, dtype=object)}

    eval_dict = {}

    # ========================================================================================================= #
    exploration_set = exp_design.generate_samples(n_samples=2000)  # for BAL
    # =====================================================
    # =============   SURROGATE MODEL TRAINING  ===========
    # =====================================================

    # --------------------------------
    t_start = time.time()
    # --------------------------------

    for it in range(0, exp_design.n_iter + 1):  # Train the GPE a maximum of "iteration_limit" times

        # 1. Train surrogate
        if gp_library == 'skl':
            sm = SklTraining(train_x=collocation_points, train_y=model_evaluations,
                             noise=False,
                             kernel_type='RBF', kernel_isotropy=True,
                             alpha=1e-6,
                             n_restarts=10,
                             parallelize=False)

        elif gp_library == 'gpy':
            # 1.3. Train a GPE, which consists of a gpe for each location being evaluated
            sm = GPyTraining(train_X=collocation_points, train_y=model_evaluations,
                             kernel_type='Matern', kernel_isotropy=True, noise=True,
                             training_iter=100, n_restarts=1, y_normalization=True,
                             optimizer="Adam",
                             verbose=False)

        # Train the GPR
        sm.train_()

        # 2. Validate GPR
        if it % exp_design.eval_step == 0:
            # Evaluate surrogates
            valid_pred, valid_std = sm.predict_(x_=exp_design.val_x) #, get_conf_int=True)

            # Get validation criteria
            run_valid = sm.validation_error(true_y=exp_design.val_y, sim_y=valid_pred, sim_std=valid_std)

            # Save the results to a dictionary
            eval_dict = save_valid_criteria(new_dict=run_valid, old_dict=eval_dict,
                                            n_tp=collocation_points.shape[0])

            # Save GP after evaluation ...................
            save_name = results_folder / f'gpr_{exp_design.main_meta_model}_TP{collocation_points.shape[0]:02d}_{exp_design.exploit_method}.pickle'
            sm.Exp_Design = exp_design
            pickle.dump(sm, open(save_name, "wb"))

            # Save eval_criteria dictionaries .........................
            save_name = results_folder / f'Validation_{exp_design.main_meta_model}_{exp_design.exploit_method}.pickle'
            pickle.dump(eval_dict, open(save_name, "wb"))

        # 3. Compute Bayesian scores in parameter space ----------------------------------------------------------
        surrogate_output, surrogate_std = sm.predict_(x_=prior)

        # total_error = (error_pp ** 2)
        bi_gpe = BayesianInference(model_predictions=surrogate_output, observations=obs, error=error_pp,
                                   sampling_method='rejection_sampling', prior=prior, prior_log_pdf=prior_logpdf)
        bi_gpe.do_inference()
        bayesian_dict['N_tp'][it] = collocation_points.shape[0]
        bayesian_dict['BME'][it], bayesian_dict['RE'][it] = bi_gpe.BME, bi_gpe.RE
        bayesian_dict['ELPD'][it], bayesian_dict['IE'][it] = bi_gpe.ELPD, bi_gpe.IE
        bayesian_dict['post_size'][it] = bi_gpe.posterior.shape[0]

        # 4. Sequential Design --------------------------------------------------------------------------------------
        if it < exp_design.n_iter:
            logger.info(f'Selecting {exp_design.training_step} additional TP using {exp_design.exploit_method}')
            SD = SequentialDesign(exp_design=exp_design, sm_object=sm, obs=obs, errors=error_pp,
                                  parallel=False)

            # new_tp, util_fun = SD.run_sequential_design(prior_samples=prior_samples)
            SD.gaussian_assumption = True
            new_tp, util_fun = SD.run_sequential_design(prior_samples=exploration_set)

            bayesian_dict[f'{exp_design.exploit_method}_{exp_design.util_func}'][it] = SD.selected_criteria[0]
            bayesian_dict['util_func'][it] = util_fun

            # Evaluate model in new TP: This is specific to each problem --------
            new_output = nonlinear_model(params=new_tp, loc=pt_loc, as_dict=True)
            # -------------------------------------------------------------------

            # Update collocation points:
            if exp_design.exploit_method == 'sobol':
                collocation_points = new_tp
                model_evaluations = new_output
            else:
                collocation_points = np.vstack((collocation_points, new_tp))
                for key in model_evaluations:
                    model_evaluations[key] = np.vstack((model_evaluations[key], new_output[key]))

            # Plot process
            if it % exp_design.eval_step == 0:
                if ndim == 1:

                    plot_1d_gpe_bal(gpr_x=prior, gpr_y=surrogate_output, gpr_std=surrogate_std, bal_x=SD.candidates,
                                    bal_y=SD.total_score, tp_x=collocation_points, tp_y=model_evaluations, it=it,
                                    true_y=obs)
                # if ndim < 3:
                #     plot_likelihoods(prior, bi_gpe.likelihood, prior, ref_scores.likelihood,
                #                      n_iter=it + 1)

                stop=1

        logger.info(f'------------ Finished iteration {it + 1}/{exp_design.n_iter} -------------------')

    # Plot all TP:
    if ndim == 1:
        plot_1d_gpe_final(gpr_x=prior, gpr_y=surrogate_output, gpr_std=surrogate_std,
                          tp_x=collocation_points, tp_y=model_evaluations,
                          tp_init=exp_design.n_init_tp, it=exp_design.n_iter, true_y=obs)

    if ndim < 3:
        plot_likelihoods(prior, bi_gpe.likelihood, prior, ref_scores.likelihood,
                         n_iter=exp_design.n_iter)
        if exp_design.exploit_method == 'bal' or exp_design.exploit_method == 'space_filling':
            plot_combined_bal(collocation_points=collocation_points, n_init_tp=exp_design.n_init_tp,
                              bayesian_dict=bayesian_dict)

    # Plot evaluation criteria:
    plot_correlation(sm_out=valid_pred, valid_eval=exp_design.val_y,
                     r2_dict=eval_dict['r2'],
                     fig_title=f'Outputs')

    # plot_validation_loc(eval_dict_list=[run_valid], label_list=[''],
    #                     output_names=output_names, n_loc=n_loc,
    #                     criteria=['mse', 'nse', 'r2', 'mean_error', 'std_error'],
    #                     fig_title=f'Validation criteria: TP: {exp_design.n_max_tp}')
    #
    # plot_validation_loc(eval_dict_list=[run_valid], label_list=[''],
    #                     output_names=output_names, n_loc=n_loc,
    #                     criteria=['norm_error', 'P95'],
    #                     fig_title=f'Validation criteria with error consideration: TP: {exp_design.n_max_tp}')
    #
    # plot_validation_tp(eval_dict_list=[eval_dict], label_list=[''],
    #                    output_names=output_names, n_loc=n_loc,
    #                    criteria=['mse', 'nse', 'r2', 'mean_error', 'std_error'], plot_loc=True,
    #                    fig_title='Validation criteria for increasing number of TP')

    # # Plot Bayesian criteria
    # plot_gpe_scores(bayesian_dict['BME'], bayesian_dict['RE'], exp_design.n_init_tp,
    #                 ref_bme=ref_scores.BME, ref_re=ref_scores.RE)
    #
    # if exp_design.exploit_method == 'BAL':
    #     plot_bal_criteria(bayesian_dict[f'{exp_design.exploit_method}_{exp_design.util_func}'], exp_design.util_func)
    # # Plot training points
    # plot_parameters(prior_samples, collocation_points, post=bi_gpe.posterior)

    end = 0

"""
TODO:
- FOR 1D-2D plots: 
# 1) To plot GPE + BAL for single and multiple observation points (and only one N_p), do separate functions (CONTINUE)
# 2) Plot GPE and Ref likelihoods

- For general case:
# 1) Plot RE, ELPD and BME for GPE and reference model with increasing number of iterations 
# 2) Plot BAl criteria with increasing number of iterations 
# 3) Plot validation criteria for each output type
3) Save time for each run, changing number of data points and parameters (one at a time) to plot later
4) Plot 1 and 2 for case 3
5) Add loop for number of parameters? 

"""
