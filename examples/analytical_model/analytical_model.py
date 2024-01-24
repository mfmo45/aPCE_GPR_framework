"""
Module contains the class for the analytical model, comporsed of a non-linear equation, obtained from:

    Oladyshkin, S., Mohammadi, F., Kroeker I. and Nowak, W. Bayesian active learning for Gaussian process emulator using
        information theory. Entropy, X(X), X, 2020,
    Oladyshkin, S. and Nowak, W. The Connection between Bayesian Inference and Information Theory for Model Selection,
        Information Gain and Experimental Design. Entropy, 21(11), 1081, 2019,

For function details and reference information, see:
https://doi.org/10.3390/e21111081
"""
import numpy as np


class AnalyticalModel:
    """
    :parameter loc: np.array [number of observations, 1] with the location where the non-linear function will be
    evaluated
    :parameter params: int, np.array [number of parameters, ] or [number of parameters, number of sample sets], with
    the values of the parameters to use in the equation

    Attributes:
        self.loc: np.array [number of observations, 1] with the location where the non-linear function will be
    evaluated
        self.params: np. array[number of parameters, number of sample sets], with parameters to input in the function
        self.output: np. array[number of sample sets, number of observations], function output evaluated in each
        parameter set, and in each location (t or x)
    """
    def __init__(self, func='non-linear', loc=None, observations=None, error=None):

        self.func = func
        self.loc = loc

        self.observations = observations

        self.n_obs = None
        self.n_params = None

        self.var = None

        self.check_input()

    def check_input(self):
        """
        Checks class input for correct format: a) if a parameter vector is given, converts is to a matrix b) at least
        2 parameters, and if only one is given, uses the same for both needed parameters
        """
        pass

    def nonlinear_model(self, params):
        """
        Function evaluates the parameters (self.params) in the non-linear equation for each location (self.loc)
        """
        # If vector with parameters is passed, reshape it to matrix
        if params.ndim == 1:
            params = np.reshape(params, (1, params.shape[0]))
        # If only one parameter is sent
        if params.shape[1] == 1:
            params = np.hstack((params, params))

        if self.n_params is None:
            self.n_params = params.shape[1]  # number of parameters

        param_sets = params.shape[0]  # number of parameter sets

        term1 = (params[:, 0] ** 2 + params[:, 1] - 1) ** 2
        term2 = params[:, 0] ** 2
        term3 = 0.1 * params[:, 0] * np.exp(params[:, 1])

        # Term that all models have in common:
        term5 = 0
        if self.n_params > 2:
            for i in range(2, self.n_params):
                term5 = term5 + np.power(params[:, i], 3) / (i + 1)

        # Sum all non-time-related terms: gives one value per row, and one row for each parameter set
        const_per_set = term1 + term2 + term3 + term5 + 1  # All non-time-related terms

        # Calculate time term: gives one value per row for each time interval
        term4 = np.full((param_sets, self.loc.shape[0]), 0.0)
        for i in range(0, param_sets):
            term4[i, :] = -2 * params[i, 0] * np.sqrt(0.5 * self.loc)

        output = term4 + np.repeat(const_per_set[:, None], self.loc.shape[0], axis=1)

        return output

    def sin_func(self):
        self.n_params = 1

        output = self.params * np.sin(self.params)

        return output

    def evaluate_model(self, params):
        """

        :param params: <array [n_params, n_mc]
                        parameters to input in the function
        :return: <array[n_mc, n_obs]>
                model evaluations in each parameter set i = {1...n_mc} for each location in "loc"
        """
        if self.func == 'non_linear':
            output = self.nonlinear_model(params)

        return output


def nonlinear_model(params, loc, as_dict=False):
    """
    Function evaluates a set of parameters (params) in the non-linear, non-Gaussian equation for each location
    (t).
    Source:
    Oladyshkin, S., Mohammadi, F., Kroeker I. and Nowak, W. Bayesian active learning for Gaussian process emulator using
        information theory. Entropy, X(X), X, 2020,
    Oladyshkin, S. and Nowak, W. The Connection between Bayesian Inference and Information Theory for Model Selection,
        Information Gain and Experimental Design. Entropy, 21(11), 1081, 2019,

    For function details and reference information, see:
    https://doi.org/10.3390/e21111081
    Args:
        params: np.array [n_mc, n_dim]
            parameter sets to evaluate the function in
        loc: np.array [1, n_loc]
            locations at which to evaluate the model in.
        as_dict: bool, True to return outpus as a dictionary, False to return them as an np.array.

    Returns: np.array [n_mc, n_loc] or a dict, with 1 entry ['Z'], which has the np.array in it. The no.array contains
    the model outputs at each location and each parameter set.

    """

    # If vector with parameters is passed, reshape it to matrix
    if params.ndim == 1:
        params = np.reshape(params, (1, params.shape[0]))
    # If only one parameter is sent
    if params.shape[1] == 1:
        params = np.hstack((params, params))

    n_params = params.shape[1]  # number of parameters
    param_sets = params.shape[0]  # number of parameter sets

    term1 = (params[:, 0] ** 2 + params[:, 1] - 1) ** 2
    term2 = params[:, 0] ** 2
    term3 = 0.1 * params[:, 0] * np.exp(params[:, 1])

    # Term that all models have in common:
    term5 = 0
    if n_params > 2:
        for i in range(2, n_params):
            term5 = term5 + np.power(params[:, i], 3) / (i + 1)

    # Sum all non-time-related terms: gives one value per row, and one row for each parameter set
    const_per_set = term1 + term2 + term3 + term5 + 1  # All non-time-related terms

    # Calculate time term: gives one value per row for each time interval
    term4 = np.full((param_sets, loc.shape[0]), 0.0)
    for i in range(0, param_sets):
        term4[i, :] = -2 * params[i, 0] * np.sqrt(0.5 * loc)

    output = term4 + np.repeat(const_per_set[:, None], loc.shape[0], axis=1)

    if as_dict:
        return {'Z': output}
    else:
        return output
