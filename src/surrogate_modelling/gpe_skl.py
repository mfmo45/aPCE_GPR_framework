"""

A new class is generated, which inherits all attributes from the GaussianProcessRegressor class from Scikit learn. This
is done to manually set the "max_iter" and "gtol" values for the optimization of hyperparameters in the GPR kernel.

ToDo: Check GPyTorch+lbfgs to see if results can be improved by changing initial values or with Adam ?
ToDo: Save each gp (for each loc) in a list, to call it later to do BAL+MCMC methods with them.
"""
import logging

import numpy as np
import sys
import math
import sklearn
import scipy
from sklearn.utils.optimize import _check_optimize_result
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sklearn.gaussian_process.kernels
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
import copy
from joblib import Parallel, delayed
from pathlib import Path


# General GPR class

# Scikit-Learn -----------------------------------------------------------------------------------------------------
class MySklGPR(GaussianProcessRegressor):
    def __init__(self, max_iter=2e05, gtol=1e-06, **kwargs):
        super().__init__(**kwargs)
        self.max_iter = max_iter
        self.gtol = gtol

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds,
                                              options={'maxiter':self.max_iter, 'gtol': self.gtol})
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min


class SklTraining():
    """
    Uses the 'sklearn' library to generate a GPE for a given forward model, based on collocation points generated by
    said forward model.

    Parameters:
        collocation_points = np.array[number of TP, number of parameters per TP]
            with training points (parameter sets)
        model_evaluations = np.array[number of TP, number of points where the model is evaluated],
            with full-complexity model outputs in each location where the fcm was evaluated/in the locations
            being considered

        tp_normalization: bool, False (default) to use training points as they are, True to normalize TP parameter
        values before training GPE

    Parameters needed by sklearn for GPR:
    (more info: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor)
        kernel = object or list of objects
            instance from "sklearn.gaussian_process.kernels" class to be used to train GPE. Will be turned to a list of
            objects
        alpha: float or list of objects,
            value added to diagonal to avoid numerical errors. If single variable, will be changed to list of floats.
        n_restarts: int,
            number of times the optimizer is run to find the optimal kernel hyper-parameters (to avoid local minima)
        optimizer: str,
            with name of optimizer to use (default is the default in GPR)
        parallelize : bool
            True to parallelize surrogate training, False to train in a sequential loop
        noise : bool
            True to add a WhiteNoiseKernel to the input kernel, False to ignore noise

    ToDo: Give evaluation location as input and then, add a function receives the observation point location and extracts
     the gpe predictions from it. These are the ones that will be used in BAL.
    """
    def __init__(self, train_x, train_y,
                 kernel_type, kernel_isotropy,
                 alpha, n_restarts, noise=True,
                 y_normalization=True, y_log=False,
                 tp_normalization=True,
                 optimizer="fmin_l_bfgs_b", parallelize=False, n_jobs=-2):

        self.X_train = train_x
        self.y_train = train_y
        self.ndim = train_x.shape[1]

        # Input for GPR library in sklearn:
        self.n_restart = n_restarts
        self.y_normalization_ = y_normalization
        self.y_log = y_log
        self.optimizer_ = optimizer
        self.noise = noise
        self.alpha = alpha

        self.parallel = parallelize
        self.n_jobs = n_jobs

        # Kernel
        self.kernel_type = kernel_type
        self.kernel_isotropy = kernel_isotropy

        # Options for GPR library:
        self.tp_norm = tp_normalization

        # self._id_vectors(alpha, kernel)

        # Initialize variables needed
        self.x_scaler = None
        self.gp_dict = {}
        self.gp_score = {}

    def _id_vectors(self, alpha, kernel):
        """
        Function checks if the inputs for alpha and kernel are a single variable or a list. If they are a single value,
        the function generates a list filled with the same value/object, so it can be properly read in the train_
        function.
        ToDO: Fix this to be able to send an error value associated to training point as alpha.
        Args:
            alpha: <float> or <list of floats [n_obs]>
                with input alpha value(s). If list, there should be one value per observation.
            kernel: <object> or <list of objects [n_obs]>
                Scikit learn kernel objects, to send to the GPR training

        Returns:
        """
        if isinstance(alpha, list):
            self.alpha = np.array(alpha)
        elif isinstance(alpha, float):
            self.alpha = np.full((self.X_train.shape[0], 1), alpha)
        elif isinstance(alpha, np.ndarray):
            if alpha.shape != (self.X_train.shape[0], 1):
                print('Using an alpha of 0')
                self.alpha = np.full((self.X_train.shape[0], 1), 0.0000001)
            else:
                self.alpha = alpha
        else:
            self.alpha = np.full((self.training_points.shape[0], self.n_obs), 0.0000001)

        # if isinstance(kernel, list):
        #     self.kernel = kernel
        # else:
        #     self.kernel = np.full(self.n_obs, kernel)

    def build_kernel(self):
        """
        Build the GP kernel based on user input
        ToDo: Add more Kernel options.
        Returns:
            Kernel object
        """
        # Bound values:
        value = np.empty((), dtype=object)
        value[()] = (1e-5, 1e3)

        if self.kernel_isotropy:
            # Isotropic kernel
            ls_initial = 1
            ls_bounds = list(np.full(1, value, dtype=object))
        else:
            # Anisotorpic kernel
            ls_initial = list(np.full(self.ndim, 1))
            ls_bounds = list(np.full(self.ndim, value, dtype=object))

        # Create Kernel
        if self.kernel_type.lower() == 'rbf':
            kernel = 1 * RBF(length_scale=ls_initial,
                             length_scale_bounds=ls_bounds)
        elif self.kernel_type.lower == 'matern':
            kernel = 1*Matern(length_scale=ls_initial,
                              length_scale_bounds=ls_bounds,
                              nu=1.5)
        else:
            logging.info(f'There is no available {self.kernel_type}. The RBF kernel will be used instead')
            kernel = 1 * RBF(length_scale=ls_initial,
                             length_scale_bounds=ls_bounds)
            self.kernel_type = 'RBF'

        return kernel

    def train_(self):
        """
        Trains a GP using the ScikitLearn library.
        ToDO: See how to save the trained kernel hyperparameters (for plotting)
        Returns:

        """
        # Transform the inputs
        if self.tp_norm:
            # Normalize the training points
            scaler_x_train = MinMaxScaler()
            scaler_x_train.fit(self.X_train)
            X_scaled = scaler_x_train.transform(self.X_train)
            self.x_scaler = scaler_x_train
        else:
            X_scaled = self.X_train

        items = self.y_train.items()

        for key, output in items:
            n_obs = output.shape[1]
            if self.parallel and n_obs > 1:
                out_list = Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(self._fit)(X_scaled,
                                                                                                output[:, i])
                                                                             for i in range(n_obs))
            else:
                out_list = []
                for i, model in enumerate(output.T):
                    out = self._fit(X_scaled, model)
                    out_list.append(out)

            self.gp_dict[key] = {}
            self.gp_score[key] = {}
            for i in range(n_obs):
                self.gp_dict[key][f'y_{i}'] = out_list[i]['gp']
                self.gp_score[key][f'y_{i}'] = out_list[i]['R2']

    def _fit(self, X_, model_y):
        """
        Function trains the Scikit-Learn surrogate model for each training location
        Args:
            X_: array[n_tp, n_param]
                with training parameter sets
            model_y: array[n_tp,]
                with simulator outputs in training points

        Returns: dict
            with trained gp object, hyperparameters and normalization parameters (if needed)

        """
        # Set Kernel:
        kernel = self.build_kernel()

        if self.noise:
            kernel = kernel + sklearn.gaussian_process.kernels.WhiteKernel(noise_level=np.std(model_y)/np.sqrt(2))

        # 1.Initialize instance of sklearn GPR:
        gp = MySklGPR(kernel=kernel, alpha=self.alpha, normalize_y=self.y_normalization_,
                      n_restarts_optimizer=self.n_restart, optimizer=self.optimizer_)

        gp.fit(X_, model_y)
        score = gp.score(X_, model_y)

        return_out_dic = dict()
        return_out_dic['gp'] = gp

        hp = np.exp(gp.kernel_.theta)

        return_out_dic['c_hp'] = hp[0]
        if hp.shape[0] < self.ndim:
            return_out_dic['cl_hp'] = hp[1]
        else:
            return_out_dic['cl_hp'] = hp[1:self.ndim + 1]

        if self.noise:
            return_out_dic['noise_hp'] = hp[-1]
        return_out_dic['R2'] = score

        return return_out_dic

    def predict_(self, x_):  # , get_conf_int=False):
        """
        Evaluates the surrogate models (for each loc) in all input_sets
        Args:
            x_: array[MC, n_params]
                with parameter sets to evaluate the surrogate models in
        Returns: dict
            with surrogate model mean (output) and the standard deviation (std) for each loc, size [n_obs, MC]

        """
        if self.tp_norm:
            x_scaled = self.x_scaler.transform(x_)
        else:
            x_scaled = x_

        items = self.gp_dict.items()

        # Loop over output types:
        y_pred = {}
        y_std = {}

        for key, gp_list in items:
            n_obs = len(gp_list)

            surrogate_prediction = np.zeros((x_.shape[0], n_obs))  # GPE mean, for each obs
            surrogate_std = np.zeros((x_.shape[0], n_obs))  # GPE mean, for each obs
            # if get_conf_int:
            #     upper_ci = np.zeros((x_.shape[0], n_obs))  # GPE mean, for each obs
            #     lower_ci = np.zeros((x_.shape[0], n_obs))  # GPE mean, for each obs

            for i in range(n_obs):
                gp = gp_list[f'y_{i}']
                surrogate_prediction[:, i], surrogate_std[:, i] = gp.predict(x_scaled, return_std=True)

                # y_pred[key] = {'output': surrogate_prediction,
                #                'std': surrogate_std}
                #
                # if get_conf_int:
                #     lower_ci[:, i] = surrogate_prediction[:, i] - (1.96 * surrogate_std[:, i])
                #     upper_ci[:, i] = surrogate_prediction[:, i] + (1.96 * surrogate_std[:, i])
                #
                #     y_pred[key]['upper_ci'] = upper_ci
                #     y_pred[key]['lower_ci'] = lower_ci
            y_pred[key] = surrogate_prediction
            y_std[key] = surrogate_std

        return y_pred, y_std

    @staticmethod
    def validation_error(true_y, sim_y, sim_std=None, std_metrics=False):
        """
        Estimates different evaluation (validation) criteria for a surrogate model, for each output location. Results for
        each output type are saved under different keys in a dictionary.
        Args:
            true_y: array [mc_valid, n_obs]
                simulator outputs for valid_samples
            sim_y: dict, with an array [mc_valid, n_obs] for each output type.
                surrogate/emulator's outputs for valid_samples.
            sim_std: dict, with an array [mc_valid, n_obs] for each output type.
                Surrogate/emulator standar deviation
            std_metrics: bool
                True to estimate error-based validation criteria. Default is False

        Returns: dict
            with validation criteria, a key for each criteria, and a subkey for each output type

        ToDo: add as part of MyGeneralGPR class, and the outputs are a dictionary, with output type as a key.
        """
        criteria_dict = {'rmse': dict(),
                         'mse': dict(),
                         'nse': dict(),
                         'r2': dict(),
                         'mean_error': dict(),
                         'std_error': dict()}

        if std_metrics:
            criteria_dict['norm_error'] = dict()
            criteria_dict['P95'] = dict()
            criteria_dict['DS'] = dict()

        for i, key in enumerate(sim_y):
            sm_out = sim_y[key]
            sm_std = sim_std[key]

            # RMSE
            criteria_dict['rmse'][key] = sklearn.metrics.mean_squared_error(y_true=true_y[key],
                                                                            y_pred=sm_out,
                                                                            multioutput='raw_values', squared=False)
            # MSE
            criteria_dict['mse'][key] = sklearn.metrics.mean_squared_error(y_true=true_y[key],
                                                                           y_pred=sm_out,
                                                                           multioutput='raw_values', squared=True)
            # NSE
            criteria_dict['nse'][key] = sklearn.metrics.r2_score(y_true=true_y[key],
                                                                 y_pred=sm_out,
                                                                 multioutput='raw_values')
            # Mean errors
            criteria_dict['mean_error'][key] = np.abs(
                np.mean(true_y[key], axis=0) - np.mean(sm_out, axis=0)) / np.mean(true_y[key], axis=0)

            criteria_dict['std_error'][key] = np.abs(
                np.std(true_y[key], axis=0) - np.std(sm_out, axis=0)) / np.std(true_y[key], axis=0)

            # R2 correlation
            criteria_dict['r2'][key] = np.zeros(sm_out.shape[1])
            for j in range(sm_out.shape[1]):
                criteria_dict['r2'][key][j] = np.corrcoef(true_y[key][:, j], sm_out[:, j])[0, 1]

            # Norm error
            if std_metrics and sim_std is not None:
                upper_ci = sm_out[:, i] + (1.96 * sm_std[:, i])
                lower_ci = sm_out[:, i] - (1.96 * sm_std[:, i])

                # Normalized error
                ind_val = np.divide(np.subtract(sm_out, true_y[key]), sm_std[key])
                criteria_dict['norm_error'][key] = np.mean(ind_val ** 2, axis=0)

                # P95
                p95 = np.where((true_y[key] <= upper_ci) & (
                        true_y[key] >= lower_ci), 1, 0)
                criteria_dict['P95'][key] = np.mean(p95, axis=0)

                # Dawid Score (https://www.jstor.org/stable/120118)
                criteria_dict['DS'][key] = np.mean(((np.subtract(sm_out, true_y[key])) / (sm_std ** 2)) + np.log(sm_std ** 2), axis=0)

        return criteria_dict

# -----------------------------------------------------------------------------------------------------------------


# def validation_error(true_y, sim_y, output_names, n_per_type):
#     """
#     Estimates different evaluation (validation) criteria for a surrogate model, for each output location. Results for
#     each output type are saved under different keys in a dictionary.
#     Args:
#         true_y: array [mc_valid, n_obs]
#             simulator outputs for valid_samples
#         sim_y: array [mc_valid, n_obs] or dict{}
#             surrogate/emulator's outputs for valid_samples. If a dict is given, it has output and std keys.
#         output_names: array [n_types,]
#             with strings, with name of each output
#         n_per_type: int
#             Number of observation per output type
#
#     Returns: float, float or array[n_obs], float or array[n_obs]
#         with validation criteria for each output locaiton, and each output type
#
#     ToDo: Like in BayesValidRox, estimate surrogate predictions here, by giving a surrogate object as input (maybe)
#     ToDo: add as part of MyGeneralGPR class, and the outputs are a dictionary, with output type as a key.
#     """
#     criteria_dict = {'rmse': dict(),
#                      'mse': dict(),
#                      'nse': dict(),
#                      'r2': dict(),
#                      'mean_error': dict(),
#                      'std_error': dict()}
#
#     # criteria_dict = {'rmse': dict(),
#     #                  'valid_error': dict(),
#     #                  'nse': dict()}
#
#     if isinstance(sim_y, dict):
#         sm_out = sim_y['output']
#         sm_std = sim_y['std']
#         upper_ci = sim_y['upper_ci']
#         lower_ci = sim_y['lower_ci']
#
#         criteria_dict['norm_error'] = dict()
#         criteria_dict['P95'] = dict()
#         criteria_dict['DS'] = dict()
#     else:
#         sm_out = sim_y
#
#     # RMSE for each output location: not a dictionary (yet). [n_obs, ]
#     rmse = sklearn.metrics.mean_squared_error(y_true=true_y, y_pred=sm_out, multioutput='raw_values',
#                                               squared=False)
#
#     c = 0
#     for i, key in enumerate(output_names):
#         # RMSE
#         criteria_dict['rmse'][key] = sklearn.metrics.mean_squared_error(y_true=true_y[:, c:c + n_per_type],
#                                                                         y_pred=sm_out[:, c:c + n_per_type],
#                                                                         multioutput='raw_values', squared=False)
#         criteria_dict['mse'][key] = sklearn.metrics.mean_squared_error(y_true=true_y[:, c:c + n_per_type],
#                                                                        y_pred=sm_out[:, c:c + n_per_type],
#                                                                        multioutput='raw_values', squared=True)
#
#         # # NSE
#         criteria_dict['nse'][key] = sklearn.metrics.r2_score(y_true=true_y[:, c:c+n_per_type],
#                                                              y_pred=sm_out[:, c:c+n_per_type],
#                                                              multioutput='raw_values')
#         # # Validation error:
#         # criteria_dict['valid_error'][key] = criteria_dict['rmse'][key] ** 2 / np.var(true_y[:, c:c+n_per_type],
#         #                                                                              ddof=1, axis=0)
#
#         # NSE
#         criteria_dict['nse'][key] = sklearn.metrics.r2_score(y_true=true_y[:, c:c + n_per_type],
#                                                              y_pred=sm_out[:, c:c + n_per_type],
#                                                              multioutput='raw_values')
#         # Mean errors
#         criteria_dict['mean_error'][key] = np.abs(
#             np.mean(true_y[:, c:c + n_per_type], axis=0) - np.mean(sm_out[:, c:c + n_per_type], axis=0)) / np.mean(
#             true_y[:, c:c + n_per_type], axis=0)
#
#         criteria_dict['std_error'][key] = np.abs(
#             np.std(true_y[:, c:c + n_per_type], axis=0) - np.std(sm_out[:, c:c + n_per_type], axis=0)) / np.std(
#             true_y[:, c:c + n_per_type], axis=0)
#
#         # Norm error
#         if isinstance(sim_y, dict):
#             # Normalized error
#             ind_val = np.divide(np.subtract(sm_out[:, c:c + n_per_type], true_y[:, c:c + n_per_type]),
#                                 sm_std[:, c:c + n_per_type])
#             criteria_dict['norm_error'][key] = np.mean(ind_val ** 2, axis=0)
#
#             # P95
#             p95 = np.where((true_y[:, c:c + n_per_type] <= upper_ci[:, c:c + n_per_type]) & (
#                         true_y[:, c:c + n_per_type] >= lower_ci[:, c:c + n_per_type]), 1, 0)
#             criteria_dict['P95'][key] = np.mean(p95, axis=0)
#
#             # Dawid Score (https://www.jstor.org/stable/120118)
#             criteria_dict['DS'][key] = np.mean(((np.subtract(sm_out[:, c:c + n_per_type],
#                                                              true_y[:, c:c + n_per_type])) / (
#                                                             sm_std[:, c:c + n_per_type] ** 2)) + np.log(
#                 sm_std[:, c:c + n_per_type] ** 2), axis=0)
#
#         criteria_dict['r2'][key] = np.zeros(n_per_type)
#         for j in range(n_per_type):
#             criteria_dict['r2'][key][j] = np.corrcoef(true_y[:, j+c], sm_out[:, j+c])[0, 1]
#
#         c = c + n_per_type
#
#     return rmse, criteria_dict


def save_valid_criteria(new_dict, old_dict, n_tp):
    """
    Saves the validation criteria for the current iteration (n_tp) to an existing dictionary, so we can have the
    results for all iterations in the same file. Each dictionary has a dictionary for each validation criteria.
    Each validation criteria has a key for each output type, which corresponds to a vector with n_loc, one value for
    each output value.
    Args:
        new_dict: Dict
            with the validation criteria for the current iteration
        old_dict: Dict
            With the validation criteria for all the previous iterations, including a key for N_tp, which saves
            the number of iteration.
        n_tp: int
            number of training points for the current BAL iteration.

    Returns: dict, with the old dictionary, with the
    """

    if len(old_dict) == 0:
        old_dict = dict(new_dict)
        old_dict['N_tp'] = [n_tp]
    else:
        for key in old_dict:
            if key == 'N_tp':
                old_dict[key].append(n_tp)
            else:
                for out_type in old_dict[key]:
                    old_dict[key][out_type] = np.vstack((old_dict[key][out_type], new_dict[key][out_type]))

    return old_dict
