"""

"""

import numpy as np
import scipy.stats as stats
import math
from joblib import Parallel, delayed
from tqdm import tqdm

from src.surrogate_modelling.exploration import Exploration


class BayesianInference:
    """
    parameters:
        model_predictions: np.array [MC_size, No.Observations],
            with model predictions
        observations: np.array [1, No.Observations]
            with true observations
        error = np.array [No.Observations,],
            with error + noise to be inserted as is in covariance matrix
        prior: np.array [MC_size, number of parameters],
            array with prior parameter sets. If None, no posterior parameter set is saved.

        prior_log_pdf: np.array [MC_size, ]
            array with the prior log probabilities of each parameter sample in "prior". Default is None, in which case
            the IE will not be estimated. It can be sent with or without the 'prior' variable.

        sampling_method : string
            Method to sample from the posterior distribution.
            Options:

            * rejection_sampling (default)
            * bayesian_weighting

    Attributes:
        self.observations = observations
        self.error = error
        self.model_predictions = model_predictions
        self. prior = prior
        self.sampling_method = sampling_method

        self.likelihood : array [MC_size,],
            with stored prior likelihood values, using multivariate Gaussian distribution
        self.cov_mat : np.array[No.Observations, No.Observations],
            covariance matrix, with variance in the diagonal
        self.post_likelihood : np.array[posterior size, ],
            with likelihood values of posterior samples.
        self.posterior : np.array[posterior size, ],
            with posterior parameter sets.
        self.BME : float
            Bayesian model evidence calculated using prior Monte Carlo smapling
        self.ELPD : float
            expected log-predicted density, expected value of posterior likelihoods
        self.RE : float
            relative entropy value between prior and posterior
        self.IE = float
            information entropy (not calculated currently)

    Functions:

    Posterior sampling options:

    Bayesian weighting method: obtains the posterior likelihoods as a weighted average of the prior based likelihood
    values. This avoids small-sized posterior sample sizes.
        + Results are similar to rejection sampling method.
        - Posterior set is not easily available (one could graph parameter  vs post likelihood)

    Rejection sampling: samples from the posterior by dividing all likelihoods by the maximum, and accepting them if
    each likelihood(i)/max(likelihood) > U[0,1].
        + Easy to obtain a posterior distribution
        - Needs a larger MC sampling to obtain a posterior, when the output dimension is large.

    ToDo: Add posterior MCMC sampling methods
    """

    def __init__(self, model_predictions, observations, error, prior=None, prior_log_pdf=None, model_error=None,
                 sampling_method='rejection_sampling'):

        self.use_log = True
        self.observations = observations
        self.error = error
        self.model_predictions = model_predictions
        self.model_error = model_error
        self.prior = prior
        self.post_sampling_method = sampling_method

        self.prior_logpdf = prior_log_pdf

        self.cov_mat = None
        self.likelihood = None
        self.log_likelihood = None

        self.post_likelihood = None
        self.posterior = None
        self.posterior_output = None

        self.BME = None
        self.ELPD = None
        self.RE = None
        self.IE = None

        self.calculate_constants()

    def calculate_constants(self):
        """
        Calculates the covariance matrix based on the input variable "error", which is a vector of variances, one for
        each observation point.

        :return: None
        """
        if type(self.error) is not np.ndarray:
            self.error = np.array([self.error])
        self.cov_mat = np.diag(self.error)

        if self.model_predictions.ndim == 1:
            self.model_predictions = np.reshape(self.model_predictions, (-1, 1))

    def calculate_likelihood(self):
        """
        Function calculates likelihood between measured data and the model output using the stats module equations.

        Notes:
        * Generates likelihood array with size [MCx1].
        * Likelihood function is multivariate normal distribution, considering independent and Gaussian-distributed
        errors.
        """

        self.likelihood = stats.multivariate_normal.pdf(self.model_predictions, cov=self.cov_mat,
                                                        mean=self.observations[0, :])

        # try:
        #     likelihood = stats.multivariate_normal.pdf(self.output, cov=self.measurement_data.cov_matrix,
        #                                                mean=self.measurement_data.meas_values[0])  # ###########
        # except ValueError as e:
        #     logger.exception(e)
        # else:
        #     self.likelihood = likelihood

        # Manual: to check results
        # lh2 = np.full(self.output.shape[0], 0.0)
        # for i in range(0, self.output.shape[0]):
        #     det = np.linalg.det(self.Syn_Data.cov_matrix)  # Calculates det of the covariance matrix
        #     inv = np.linalg.inv(self.Syn_Data.cov_matrix)  # inverse of covariance matrix
        #     diff = self.Syn_Data.meas_data - self.output[i, :]  # Gets the difference between measured and
        #                                                         # modeled value
        #     term1 = 1 / np.sqrt((math.pow(2 * math.pi, 10)) * det)
        #     term2 = -0.5 * np.dot(np.dot(diff, inv), diff.transpose())
        #     lh2[i] = term1 * np.exp(term2)

    def calculate_likelihood_manual(self):
        """
        Function calculates likelihood between observations and the model output manually, using numpy calculations.

        Notes:
        * Generates likelihood array with size [MCxN], where N is the number of measurement data sets.
        * Likelihood function is multivariate normal distribution, considering independent and Gaussian-distributed
        errors.
        * Method is faster than using stats module ('calculate_likelihood' function).
        """
        # Calculate constants:
        det_R = np.linalg.det(self.cov_mat)
        invR = np.linalg.inv(self.cov_mat)
        const_mvn = pow(2 * math.pi, - self.observations.shape[1] / 2) * (1 / math.sqrt(det_R))  # ###########

        # vectorize means:
        means_vect = self.observations[:, np.newaxis]  # ############

        # Calculate differences and convert to 4D array (and its transpose):
        diff = means_vect - self.model_predictions  # Shape: # means
        diff_4d = diff[:, :, np.newaxis]
        transpose_diff_4d = diff_4d.transpose(0, 1, 3, 2)

        # Calculate values inside the exponent
        inside_1 = np.einsum("abcd, dd->abcd", diff_4d, invR)
        inside_2 = np.einsum("abcd, abdc->abc", inside_1, transpose_diff_4d)
        total_inside_exponent = inside_2.transpose(2, 1, 0)
        total_inside_exponent = np.reshape(total_inside_exponent,
                                           (total_inside_exponent.shape[1], total_inside_exponent.shape[2]))

        # likelihood = const_mvn * np.exp(-0.5 * total_inside_exponent)
        likelihood = np.exp(-0.5 * total_inside_exponent)
        log_likelihood = -0.5 * total_inside_exponent

        # Convert likelihoods to vector:
        if log_likelihood.shape[1] == 1:
            likelihood = likelihood[:, 0]
            log_likelihood = log_likelihood[:, 0]
        self.log_likelihood = log_likelihood
        self.likelihood = likelihood

    def calculate_likelihood_with_error(self):
        """
        Function calculates likelihood between observations and the model output manually, using numpy calculations. It
        considers model error, with an error associated to each model prediction.

        Notes:
        * Generates likelihood array with size [MCxN], where N is the number of measurement data sets.
        * Likelihood function is multivariate normal distribution, considering independent and Gaussian-distributed
        errors.
        * Method is faster than using stats module ('calculate_likelihood' function).
        """
        # Calculate augmented covariance:
        mc_size = self.model_predictions.shape[0]
        cov_3d = np.tile(self.cov_mat[np.newaxis, :, :], (mc_size, 1, 1))   # make 3D (1 cov per MC run)
        std_3d = np.array([np.diag(row) for row in self.model_error])     # make 3D matrix for std
        self.augmented_cov = cov_3d + std_3d**2                           # combine covariances

        det_R = np.linalg.det(self.augmented_cov)
        invR = np.linalg.inv(self.augmented_cov)
        # Can't ignore constant
        const_mvn = pow(2 * math.pi, - self.observations.shape[1] / 2) * (1 / np.sqrt(det_R)).reshape(-1, 1)
        log_constant = self.observations.shape[1] * math.log(2 * math.pi) + np.log(det_R.reshape(-1, 1))

        # vectorize means:
        means_vect = self.observations[:, np.newaxis]  # ############

        # Calculate differences and convert to 4D array (and its transpose):
        diff = means_vect - self.model_predictions  # Shape: # means
        diff_4d = diff[:, :, np.newaxis]
        transpose_diff_4d = diff_4d.transpose(0, 1, 3, 2)

        # Calculate values inside the exponent
        inside_1 = np.einsum("abcd, bdd->abcd", diff_4d, invR)
        inside_2 = np.einsum("abcd, abdc->abc", inside_1, transpose_diff_4d)
        total_inside_exponent = inside_2.transpose(2, 1, 0)
        total_inside_exponent = np.reshape(total_inside_exponent,
                                           (total_inside_exponent.shape[1], total_inside_exponent.shape[2]))

        likelihood = const_mvn * np.exp(-0.5 * total_inside_exponent)
        log_likelihood = -0.5*(log_constant + total_inside_exponent)

        # Convert likelihoods to vector:
        if log_likelihood.shape[1] == 1:
            log_likelihood = log_likelihood[:, 0]
            likelihood = likelihood[:, 0]
        self.log_likelihood = log_likelihood
        self.likelihood = likelihood

    def rejection_sampling(self):
        """
        Function runs rejection sampling: generating N(MC) uniformly distributed random numbers (RN). If the normalized
        value of the likelihood {likelihood/max(likelihood} is smaller than RN, reject prior sample. The values that
        remain are the posterior.

        Notes:
            *Generates the posterior likelihood, posterior values, and posterior density arrays
            *If max likelihood = 0, then there is no posterior distributions, or the posterior is the same as the
            prior.
        """
        # Generate MC number of random values between 1 and 0 (uniform dist) ---------------------------------------- #
        rn = stats.uniform.rvs(size=self.model_predictions.shape[0])  # random numbers

        if self.log_likelihood is not None and self.use_log:
            log_rn = np.log(rn)
            max_likelihood = np.max(self.log_likelihood)
            # post_index = np.array(np.where(np.exp(self.log_likelihoodlikelihood - max_likelihood) > rn)[0])  # using
            post_index = np.array(np.where(self.log_likelihood - max_likelihood > log_rn)[0])

            # 2. Get posterior distributions:
            self.post_loglikelihood = np.take(self.log_likelihood, post_index, axis=0)

        else:
            max_likelihood = np.max(self.likelihood)  # Max likelihood
            post_index = np.array(np.where(self.likelihood / max_likelihood > rn)[0])
            self.post_likelihood = np.take(self.likelihood, post_index, axis=0)
            self.post_loglikelihood = np.log(self.post_likelihood)

        # Get posterior samples
        self.posterior_output = np.take(self.model_predictions, post_index, axis=0)
        if self.prior is not None:
            self.posterior = np.take(self.prior, post_index, axis=0)
        # 4. Get posterior log-pdf values:
        if self.prior_logpdf is not None:
            self.post_logpdf = self.prior_logpdf[post_index]

    def estimate_bme(self):
        """
        Function calculates likelihood and BME (prior based) and then, based on the given posterior sampling criteria,
        obtains a posterior likelihood, ELPD and RE.

        :return:

        Note:
            If BME = 0, then it means that the model was not able to reproduce the observed data, and so we assume
            BME = ELPD, and thus RE is also 0, since nothing mas learned.
        """
        # 1. Prior Likelihood
        if self.model_error is None:
            self.calculate_likelihood_manual()
        else:
            self.calculate_likelihood_with_error()

        # Posterior sampling:
        if 'bayesian_weighting' in self.post_sampling_method.lower():  # Bayesian weighting
            # 3. Posterior estimation using Bayesian weighting
            non_zero_lk = self.likelihood[np.where(self.likelihood != 0)]
            post_w = non_zero_lk / np.sum(non_zero_lk)

            # 4. Posterior-based scores
            self.BME = np.mean(self.likelihood)
            self.ELPD = np.sum(post_w * np.log(non_zero_lk))

        elif 'rejection_sampling' in self.post_sampling_method.lower():  # rejection sampling
            # 3. Posterior estimation/sampling
            self.rejection_sampling()

            # 2. prior-based BME
            if self.log_likelihood is not None and self.use_log:  # estimate using log-likelihoods
                self.BME = np.mean(np.exp(self.log_likelihood))
                self.ELPD = np.mean(self.post_loglikelihood)

            else:
                self.BME = np.mean(self.likelihood)
                self.ELPD = np.mean(np.log(self.post_likelihood))

        # If BME is 0:
        if self.BME == 0:
            # print('BME is 0, so we use stupid number ***********************')
            # RE = ELPD - np.log(1e-300)
            self.RE = math.nan
            self.IE = math.nan
        else:
            self.RE = self.ELPD - np.log(self.BME)
            if self.prior_logpdf is not None:
                self.IE = np.log(self.BME) - np.mean(self.post_logpdf) - self.ELPD
            else:
                self.IE = None


class BAL:
    """
    Class runs Bayesian active learning to select the next training point (TP) to train the Gaussian process emulator
    (GPE).

    Procedure is based on:
    Oladyshkin, S., Mohammadi, F., Kroeker I. and Nowak, W. Bayesian active learning for Gaussian process emulator using
        information theory. Entropy, X(X), X, 2020. DOI:

    Parameters:
        gpe_mean: np.array(MC_size, N_obs), with GPE predictions (posterior mean) values
        gpe_std: np.array(MC_size, N_obs), with GPE variance (posterior standard deviation) values
        mc_bal: int, with number of posterior samples to take from the GPE posterior distribution, for each parameter
        set being explored.
        d_bal: int, number of parameter sets to explore for BAL.

        observations: np.array(1, No. of observations), array with true observations to calculate Bayesian scores.
        observation_error: np.array(No. of observations,), array with measurement error values for each observation
        bal_strategy: str, which score to use to select the next TP. Options are: 1) RE: argmax(RE), 2) BME: argmax(BME)

    Attributes:
        self.mean = gpe_mean
        self.std = gpe_std
        self.mc_bal = mc_bal
        self.d_size_bal = d_bal

        self.obs = observations
        self.obs_error = observation_error
        self.al_strategy = bal_strategy

        self.bal_samples = np.array(self.d_size_bal,) with the parameter sets being explored
        self.al_unique_index = np.array(self.d_size_bal,), with indexes of the N(mean, std) being explored.
        self.BME_bal = np.zeros(self.d_size_bal), array to save the BME values of each parameter set explored
        self.ELPD_bal = np.zeros(self.d_size_bal) array to save the ELPD values of each parameter set explored
        self.RE_bal = np.zeros(self.d_size_bal) array to save the RE values of each parameter set explored

    Note:
        - BAL strategies:
        1) RE: selects the parameter set with the highest RE value (parameter set learned the most from the
        observations)
        2) BME: selects the parameter set with the highest BME value (parameter set with the best fit to the data)

    """

    def __init__(self, gpe_mean, gpe_std, mc_bal, d_bal,
                 observations, observation_error, bal_strategy="RE"):

        self.mean = gpe_mean
        self.std = gpe_std
        self.mc_bal = mc_bal
        self.d_size_bal = d_bal

        self.obs = observations
        self.obs_error = observation_error
        self.al_strategy = bal_strategy

        self.bal_samples = None
        self.al_unique_index = None
        self.BME_bal = np.zeros(self.d_size_bal)
        self.ELPD_bal = np.zeros(self.d_size_bal)
        self.RE_bal = np.zeros(self.d_size_bal)

    def select_indexes(self, prior_samples, collocation_points, i):
        """
        Function selects the parameter sets (from 'prior_samples') to explore in BAL
        :param prior_samples: np.array(MC_size, N.parameters), all parameter sets which have already been evaluated in
        the trained GPE
        :param collocation_points: np.array(number of TP used already, N.parameters), array with TP used to train the
        GPE, and should therefore not be explored
        :param i: int, number of TP that have already been selected using BAL (to avoid resampling them and assuring
        'd_size_bal' parameters explored
        :return: None
        """

        # a) get index of elements that have already been used
        aux1 = np.where((prior_samples[:self.d_size_bal + i, :] == collocation_points[:, None]).all(-1))[1]
        # b) give each element in the prior a True if it has not been used before
        aux2 = np.invert(np.in1d(np.arange(prior_samples[:self.d_size_bal + i, :].shape[0]), aux1))
        # c) Select the first d_size_bal elements in prior_sample that have not been used before
        self.al_unique_index = np.arange(prior_samples[:self.d_size_bal + i, :].shape[0])[aux2]

        self.bal_samples = prior_samples[self.al_unique_index]

    def explore_posterior(self):
        """
        Function explores the posterior distribution, generated by the GPE, for each parameter set in 'self.bal_samples'
        It Loops through each parameter set and
            a) Explores the GPE posterior output space obtained from each.
            b) Using the output samples, calculates the BME, ELPD and RE for each
        :return: ---
        """
        # Get number of observations:
        n_obs = self.mean.shape[1]
        # Loop through each (d_size_bal) parameter set and explore its posterior output space N(mean, std)
        for iAL in range(0, len(self.al_unique_index)):
            # Explore parameter set: N(0,1) * gpe_std + gpe_mean
            al_exploration = np.random.normal(size=(self.mc_bal, n_obs)) * self.std[self.al_unique_index[iAL], :] \
                             + self.mean[self.al_unique_index[iAL], :]

            # BAL scores computation
            bi_bal = BayesianInference(model_predictions=al_exploration, observations=self.obs, error=self.obs_error)
            bi_bal.estimate_bme()
            self.BME_bal[iAL], self.ELPD_bal[iAL], self.RE_bal[iAL] = bi_bal.BME, bi_bal.ELPD, bi_bal.RE

    def bal_select_tp(self):
        """
        Function selects the next training point based on the user input criteria.
        :return: float (BME or RE value chosen as best criteria), int (index where the best criteria value is located
        in the prior_samples array)
        """
        if self.al_strategy == "BME":
            al_value = np.amax(self.BME_bal)
            al_value_index = np.argmax(self.BME_bal)

            if np.amax(self.BME_bal) == 0:
                print("Warning Active Learning: all values of Bayesian model evidences equal to 0")
                print("Active Learning Action: training point has been randomly selected")

        elif self.al_strategy == "RE":
            al_value = np.amax(self.RE_bal)
            al_value_index = np.argmax(self.RE_bal)

            if np.amax(self.RE_bal) == 0 and np.amax(self.BME_bal) != 0:
                al_value = np.amax(self.BME_bal)
                al_value_index = np.argmax(self.BME_bal)
                print("Warning Active Learning: all values of Relative entropies equal to 0")
                print("Active Learning Action: training point has been selected using BME.")

            elif np.amax(self.RE_bal) == 0 and np.amax(self.BME_bal) == 0:
                al_value = np.amax(self.BME_bal)
                al_value_index = np.argmax(self.BME_bal)
                print("Warning Active Learning: all values of Relative entropy and BME are equal to 0")
                print("Active Learning Action: training point has been randomly selected")

        prior_index = self.al_unique_index[al_value_index]  # Get which index in the prior sample array corresponds to
        #                                                     the TP selected  by BAL
        return al_value, prior_index


class SequentialDesign:
    """
    Class runs the optimal design of experiments (sequential design) to select the new training points, to add to the
    existing training points for surrogate model training.
    Args:
        exp_design: ExpDesign object
            Used to sample from the prior distribution, and extract exploit and explore methods.
        sm_object: object
            surrogate model class object, either SklTraining, GPyTraining,  must have a 'self.predict_(input_params)'
            function to evaluate surrogate.
        n_cand_groups: int
            in how many lists to split the candidate set, to do MultiProcessing.
        multiprocessing: bool
            True to use multiprocessing (parallelize) tasks. False to set n_cand_groups=1

        obs : array [n_obs, ]     (ToDo: dict, with a key for each output type )
            array with observation values
        errors: array  [n_obs, ]  (ToDo: dict, with a key for each output type)
            array with measurement error for each observation. Default is None

        do_tradeoff : bool
            True to consider the total score a combination of exploration and exploitation score. False to just use
            either, depending on the exploitation method.

        secondary_sm: object
            surrogate model class object, either SklTraining, GPyTraining,  must have a 'self.predict_(input_params)'
            function to evaluate surrogate. It corresponds to the secondary, or error model, which is added to the
            sm_object main surrogate.
        gaussian_assumption: bool
            True to assume a Gaussian prior and likelihood, so analytical equations for BAl are used. False to
            follow the traditional sampling approach.

    Attributes:

    """
    def __init__(self, exp_design, sm_object, obs,
                 n_candidates=1000, n_cand_groups=4,
                 secondary_sm=None,
                 parallel=True, n_jobs=-2,
                 errors=None,
                 do_tradeoff=False,
                 gaussian_assumption=False):

        self.ExpDesign = exp_design
        self.SM = sm_object
        self.EM = secondary_sm

        self.observations = obs
        self.m_error = errors

        self.n_cand_groups = n_cand_groups
        self.parallel = parallel
        self.n_jobs = n_jobs

        self.do_tradeoff = do_tradeoff
        self.gaussian_assumption = gaussian_assumption

        self.candidates = None
        self.exploit_score = None        # exploitation score (non-normalized values)
        self.exploit_score_norm = None   # normalized exploitation score
        self.explore_score = None        # Normalized exploration score
        self.total_score = None          # Total score, either weighted sum of both norm score
        self.selected_criteria = None    # value(s) of selected TP(s) utility function

        self.mc_samples = n_candidates  # Number of parameter sets to sample and explore
        self.mc_exploration = 10_000   # number of output sets to sample during posterior exploration

        self.check_inputs()

    def check_inputs(self):
        if self.observations.ndim != 1:
            self.observations = self.observations.reshape(-1)
            print(f'The observations input was changed to a 1D vector with size {self.observations.shape}')
        if self.m_error.ndim != 1:
            self.m_error = self.m_error.reshape(-1)
            print(f'The error input was changed to a 1D vector with size {self.m_error.shape}')

    def run_sequential_design(self, prior_samples=None):

        exp_design = self.ExpDesign
        explore_method = exp_design.explore_method
        exploit_method = exp_design.exploit_method
        util_fun = exp_design.util_func

        # Sample the candidate parameter sets, to evaluate in the surrogate model and use for sequential design -->
        # to select the new training point
        if prior_samples is not None:
            candidate_idx = self.select_indexes(prior_samples=prior_samples,
                                                collocation_points=self.SM.training_points)
            all_candidates = prior_samples[candidate_idx, :]
            # Update number of mc_samples:
            score_exploration = np.zeros(self.mc_samples)
        else:
            explore = Exploration(n_candidate=self.mc_samples,
                                  old_tp=self.SM.training_points,
                                  exp_design=exp_design,
                                  mc_criterion='mc-intersite-proj'  # mc-intersite-proj-th, mc-intersite-proj
                                  )

            all_candidates, score_exploration = explore.get_exploration_samples()

        self.explore_score = score_exploration
        self.candidates = all_candidates

        # Implement the sequential design method
        if 'bal' in exp_design.exploit_method.lower():  # for BAL and AL methods
            # Split the candidates in groups for multiprocessing

            if 'dkl' in util_fun.lower() and 'post_bme' in util_fun.lower():
                # randomly select whether we use bme or dkl, with a 50-50 chance each
                rand_n = stats.uniform.rvs(size=1)  # random numbers
                if rand_n < 0.5:
                    util_fun = 'post_bme'
                else:
                    util_fun = 'dkl'

            elif 'dkl' in util_fun.lower() and 'bme' in util_fun.lower():
                # randomly select whether we use bme or dkl, with a 50-50 chance each
                rand_n = stats.uniform.rvs(size=1)  # random numbers
                if rand_n < 0.5:
                    util_fun = 'bme'
                else:
                    util_fun = 'dkl'

            # for traditional approach, we can't get a posterior logBME
            if 'post_bme' in util_fun.lower() and not self.gaussian_assumption:
                util_fun = 'bme'

            if self.parallel and self.n_cand_groups > 1:
                split_cand = np.array_split(all_candidates, self.n_cand_groups, axis=0)
                results = Parallel(n_jobs=self.n_jobs, backend='threading')(
                    delayed(self.run_al_functions)(exploit_method, split_cand[i], i,
                                                   self.m_error, util_fun)
                    for i in range(self.n_cand_groups))

                score_exploit = np.concatenate([results[NofE][1] for NofE in range(self.n_cand_groups)])

            else:
                results = self.run_al_functions(exploit_method=exploit_method, candidates=all_candidates,
                                                index=1, m_error=self.m_error, utility_func=util_fun)
                score_exploit = results[1]

            # Normalize exploit score
            score_exploit_norm = score_exploit / np.nansum(np.abs(score_exploit))
            self.exploit_score_norm = score_exploit_norm
            self.exploit_score = score_exploit

            if self.do_tradeoff:
                # assigning 50-50 score to exploitation and exploration
                total_score = (0.5 * score_exploration) + (0.5 * score_exploit_norm)
                self.total_score = total_score
            else:
                total_score = score_exploit_norm
                self.total_score = score_exploit

            if np.nansum(score_exploit) == 0:
                util_fun = 'global_mc'

            # Order in descending order
            temp = total_score.copy()  # copy
            temp[np.isnan(total_score)] = -np.inf  # make all nan entries -inf
            if 'ie' in util_fun.lower():                         # To maximize -IE
                temp = [val if val != 0 else -np.inf for val in temp]
            sorted_idx = np.argsort(temp)[::-1]  # order from largest to smallest

            # Select new TP(s)
            new_tp = all_candidates[sorted_idx[:exp_design.training_step]]
            self.selected_criteria = self.total_score[sorted_idx[:exp_design.training_step]]

        elif 'space_filling' in exp_design.exploit_method.lower():
            self.total_score = score_exploration
            temp = score_exploration.copy()                      # copy
            temp[np.isnan(score_exploration)] = -np.inf          # make all nan entries -inf
            sorted_idx = np.argsort(temp)[::-1]                  # order from largest to smallest

            # select the requested number of samples
            new_tp = all_candidates[sorted_idx[:exp_design.training_step]]
            self.selected_criteria = self.total_score[sorted_idx[:exp_design.training_step]]
            util_fun = 'global_mc'

        elif 'sobol' in exp_design.exploit_method.lower():
            new_tp_size = self.SM.training_points.shape[0] + exp_design.training_step
            # Sample new TPs
            new_tp = exp_design.generate_samples(n_samples=new_tp_size, sampling_method='sobol')
            self.selected_criteria = 0
            util_fun = ''

        return new_tp, util_fun

    def bayesian_active_learning(self, y_mean, y_std, observations, error, utility_function='dkl'):
        """
        Computes scores based on Bayesian active design criterion (utility_criteria).

        It is based on the following paper:
        Oladyshkin, Sergey, Farid Mohammadi, Ilja Kroeker, and Wolfgang Nowak.
        "Bayesian3 active learning for the gaussian process emulator using
        information theory." Entropy 22, no. 8 (2020): 890.

        Parameters
        ----------
        y_mean : array [n_samples, n_obs]   ToDo: Dictionary, with a key for each output type, each array [mc_size, n_obs]
            Array with surrogate model outputs (mean)
        y_std : array [n_samples, n_obs]   ToDo: Dictionary, with a key for each output type, each array [mc_size, n_obs]
            Array with output standard deviation
        observations : array [n_obs, ] ToDo: Dictionary, with a key for each output type, each array [1, n_obs]
            array with measured observations
        error : array [n_obs]   ToDO: dict A dictionary containing the measurement errors (sigma^2). One dictionary for each output type
            an array with the observation errors associated to each output
        utility_function : string, optional
            BAL design criterion. The default is 'DKL'.

        Returns
        -------
        float
            Score.

        """
        obs_data = self.observations
        n_obs = self.observations.shape[0]

        # Explore posterior:
        # '''ToDo: when outputs are modified to dictionaries'''
        # y_mc, std_mc = {}, {}
        # logPriorLikelihoods = np.zeros(self.mc_exploration)
        # for key in list(y_mean):
        #     cov = np.diag(y_std[key] ** 2)
        #     rv = stats.multivariate_normal(mean=y_mean[key], cov=cov)    # stats object with y_mean, y_var
        #     y_mc[key] = rv.rvs(size=self.mc_exploration)                 # sample from posterior space
        #     logPriorLikelihoods += rv.logpdf(y_mc[key])                  # get prior probability
        #     std_mc[key] = np.zeros((self.mc_exploration, y_mean[key].shape[0]))

        cov = np.diag(y_std ** 2)
        rv = stats.multivariate_normal(mean=y_mean, cov=cov)    # stats object with y_mean, y_var
        y_mc = rv.rvs(size=self.mc_exploration)                 # sample from posterior space
        logPriorLikelihoods = rv.logpdf(y_mc)                   # get prior probability

        bi_bal = BayesianInference(model_predictions=y_mc, observations=observations, error=error,
                                   prior_log_pdf=logPriorLikelihoods,                    # Needed to estimate IE
                                   sampling_method='rejection_sampling')
        bi_bal.estimate_bme()

        if utility_function.lower() == 'dkl':    # '''ToDo: Make it prior/posterior based'''
            u_j_d = bi_bal.RE           # Max is better: leave as positive
        elif utility_function.lower() == 'bme':  # '''ToDo: Make it prior/posterior based'''
            u_j_d = bi_bal.BME          # Max is better: leave as positive
        elif utility_function.lower() == 'ie':
            u_j_d = bi_bal.IE * -1      # Min is better: multiply by -1 to maximize
        else:
            print('The selected utility function is not yet available. ')
            # '''ToDO: add BIC, DIC, KIC (from BayesValidRox) and posterior-based criteria.'''

        # Catch if U_j_d is nan or inf, and replace by 0, since we are maximizing
        if np.isnan(u_j_d) or u_j_d == -np.inf or u_j_d == np.inf:
            u_j_d = 0.0

        return u_j_d

    def analytical_bal(self, y_mean, y_std, observations, error, utility_function='dkl'):
        """Function computes the analytical BAL criteria (IE or DKL), when the prior and likelihood are both Gaussian
        distributions. It first estimates the posterior distribution, and then estimates either the Dkl or IE.
        For ill-posed priors, we check if the prior and posterior MG distributions overlap in any dimension, if not,
        then the BAl criteria are not estimated.

        The post logBME equation was obtained from Oladyshkin and Nowak (2019) (doi: 10.3390/e21111081), eq.(28)

        Args:
            y_mean : array [n_samples, n_obs]  array with ith surrogate model outputs (mean)
            y_std : array [n_samples, n_obs]   array with output standard deviation
            observations : array [n_obs, ] array with measured observations
            error : array [n_obs]  array with the observation errors associated to each output
            utility_function : string, optional, BAL design criterion. The default is 'DKL'.

        Returns:
            float: analytical BAL criteria for the given input distribution
        """
        n_obs = observations.shape[0]

        if observations.ndim == 2:
            observations = observations.reshape(-1)

        # Prior
        prior_cov = np.diag(y_std ** 2)

        # Likelihood: observation info
        error_cov = np.diag(error)

        # Estimate posterior analytically
        posterior_var_mv = np.linalg.inv(np.add(np.linalg.inv(error_cov), np.linalg.inv(prior_cov)))
        posterior_mean_mv = np.dot(posterior_var_mv,
                                   np.dot(np.linalg.inv(prior_cov), y_mean) + np.dot(np.linalg.inv(error_cov),
                                                                                     observations))
        # Check if MG overlap
        if self.gaussian_overlap(mu1=y_mean, cov1=prior_cov, mu2=posterior_mean_mv, cov2=posterior_var_mv):
            if utility_function == 'dkl':
                u_j_d = self.multivariate_gaussian_kl_divergence(mu_p=posterior_mean_mv, cov_p=posterior_var_mv,
                                                                 mu_q=y_mean, cov_q=prior_cov)
            elif utility_function == 'ie':
                u_j_d = 0.5 * np.log(((2 * math.pi * math.e) ** n_obs) * np.linalg.det(posterior_var_mv))
                u_j_d = -1 * u_j_d

            elif utility_function == 'post_bme':  # post log(BME)
                # Sample from posterior
                posterior_mv = stats.multivariate_normal(mean=posterior_mean_mv, cov=posterior_var_mv)
                posterior_samples_mv = posterior_mv.rvs(size=self.mc_exploration)

                IE = 0.5 * np.log(((2 * math.pi * math.e) ** n_obs) * np.linalg.det(posterior_var_mv))
                prior_log_pdf = self.posterior_log_likelihood(samples=posterior_samples_mv.reshape(-1, n_obs),
                                                              mean=y_mean.reshape(1, -1), cov_mat=prior_cov)
                post_log_likelihoods = self.posterior_log_likelihood(samples=posterior_samples_mv.reshape(-1, n_obs),
                                                                     mean=observations.reshape(1, -1),
                                                                     cov_mat=error_cov)
                u_j_d = np.mean(post_log_likelihoods) + np.mean(prior_log_pdf) + IE
        else:
            # If the 99% confidence intervals of the prior and posteior multivariate gaussian doesn't overlap in any
            # dimension, then I cannot calculate RE or IE, so I assign a nan value, so the parameter set is not
            # considered in the BAL
            u_j_d = np.nan  # Used nan instead of 0, so if log-based criteria are negative, 0 can be the best case
                            # or in minimization problems, 0 will always be chosen.

        # Catch if U_j_d is nan or inf, and replace by 0, since we are maximizing
        if np.isnan(u_j_d) or u_j_d == -np.inf or u_j_d == np.inf:
            u_j_d = np.nan  # Used nan instead of 0, so if log-based criteria are negative, 0 can be the best case
                            # or in minimization problems, 0 will always be chosen.

        return u_j_d

    def run_al_functions(self, exploit_method, candidates, index, m_error, utility_func):
        """
        Runs the utility function based on the given method.
        Args:
            exploit_method: string
                Exploitation method. Current options: 'bal': Bayesian active learning
            candidates: array [mc_size, n_params]
                with candidate parameter sets to explore, one at a time, and assign a score based on the utility
                function
            index:
            m_error: array [n_obs, ]
                With measurement error for each observation
            utility_func: string
                with name of utility function/active learning criteria to use to assign a score to each candidate set

        Returns: array [n_candidates,]
            with the scores assigned to each candidate, in descending order (according to the score)
        """

        if exploit_method.lower() == 'varoptdesign':
            print('Not yet implemented: method select TP based on surrogate model variance')
            # U_J_d = np.zeros((candidates.shape[0]))
            # for idx, X_can in tqdm(enumerate(candidates), ascii=True,
            #                        desc="varoptdesign"):
            #     U_J_d[idx] = self.util_VarBasedDesign(X_can, index, utility_func)

        elif exploit_method.lower() == 'bal':
            n_candidate = candidates.shape[0]   # number of candidates
            U_J_d = np.zeros(n_candidate)       # array to save scores for each candidate

            # Evaluate candidates in surrogate model
            output = self.SM.predict_(input_sets=candidates)
            y_cand = output['output']
            std_cand = output['std']
            if self.EM is not None:
                error_out = self.EM.predict_(input_sets=candidates)

                y_cand = y_cand + error_out['output']
                std_cand = std_cand + error_out['std']

            for idx, cand in tqdm(enumerate(candidates), ascii=True, desc='Exploring posterior in BAL Design'):
                '''ToDo: When SM output has dictionaries for each output type'''
                # y_mean = {key: items[idx] for key, items in y_cand.items()}
                # y_std = {key: items[idx] for key, items in std_cand.items()}
                y_mean = y_cand[idx, :]
                y_std = std_cand[idx, :]
                if utility_func.lower() == "bme" or not self.gaussian_assumption:
                    U_J_d[idx] = self.bayesian_active_learning(y_mean=y_mean, y_std=y_std,
                                                               observations=self.observations, error=m_error,
                                                               utility_function=utility_func)
                else:
                    U_J_d[idx] = self.analytical_bal(y_mean=y_mean, y_std=y_std,
                                                     observations=self.observations, error=m_error,
                                                     utility_function=utility_func)

        return index, U_J_d

    @staticmethod
    def gaussian_overlap(mu1, cov1, mu2, cov2):
        """Function to determine if 2 multivariate Gaussian distributions overlap in any dimension.
        If they overlap in any dimension, then the analytical posterior-based criteria can be estimated. As overlap
        criteria we arbitrarily selected that, if the 2 distributions overlap anywhere within the 99% confidence
        intervals, then they do overlap.

        Args:
            mu1 (np.array [n_dim, ]): array with mean values for distribution 1 (prior)
            cov1 (np.array [n_dim, n_dim]): diagonal matrix with the variances for distribution 1 (prior)
            mu2 (np.array [n_dim, ]): array with mean values for distribution 2 (posterior)
            cov2 (np.array [n_dim, n_dim]): diagonal matrix with the variances for distribution 2 (posterior)

        Returns:
            bool: True if they overlap, False if they don't.
        """

        # Calculate the standard deviations for each distribution
        std_dev1 = np.sqrt(np.diag(cov1))
        std_dev2 = np.sqrt(np.diag(cov2))

        # Calculate quantiles for the 1% and 99% confidence levels for both distributions
        quantiles1_left = mu1 - stats.norm.ppf(0.99) * std_dev1
        quantiles1_right = mu1 + stats.norm.ppf(0.99) * std_dev1

        quantiles2_left = mu2 - stats.norm.ppf(0.99) * std_dev2
        quantiles2_right = mu2 + stats.norm.ppf(0.99) * std_dev2

        # Check if any dimension overlaps
        overlap = np.any((quantiles1_right >= quantiles2_left) & (quantiles1_left <= quantiles2_right))

        return overlap

    @staticmethod
    def multivariate_gaussian_kl_divergence(mu_p, cov_p, mu_q, cov_q):
        """Function estimates the analytical solution for the Kullback-Leibler divergence when going from the
        prior (q) to the posterior (p) when oth prior and posteriors are Gaussian distributions.

        Args:
            mu_p (np.array [n_obs, ]): array with the mean values for the posterior distribution
            cov_p (np.array [n_obs, n_obs]): diagonal matrix with the variance for the posterior distribution
            mu_q (np.array [n_obs, ]): array with the mean values for the prior distribution
            cov_q (np.array [n_obs, n_obs]): diagonal matrix with the variance for the prior distribution distribution

        Returns:
            float: Kullback-Leibler divergence between prior and posterior
        """
        # Dimensionality of the distributions
        k = len(mu_p)

        # Inverse and determinant of the covariance matrices
        inv_cov_q = np.linalg.inv(cov_q)
        det_cov_q = np.linalg.det(cov_q)
        # inv_cov_p = np.linalg.inv(cov_p)
        det_cov_p = np.linalg.det(cov_p)

        # Terms in the KL divergence formula
        term1 = np.trace(inv_cov_q @ cov_p)
        term2 = (mu_q - mu_p).T @ inv_cov_q @ (mu_q - mu_p)
        term3 = np.log(det_cov_q / det_cov_p)

        # Calculate the KL divergence
        kl_divergence = 0.5 * (term1 + term2 - k + term3)

        return kl_divergence

    @staticmethod
    def posterior_log_likelihood(samples, mean, cov_mat):
        """Function estimates the log pdf of a Gaussian distribution manually (faster than using stats)

        Args:
            samples (np.array [mc_exploration, n_obs]): array with samples to get the pdf from
            mean (np.array [1, n_obs]): mean array of Gaussian distribution
            cov_mat (np.array [n_obs, n_obs]): covariance of the Gaussian distribution

        Returns:
            np.array [mc_exploration, ]: array with pdf value of each sample
        """
        det_R = np.linalg.det(cov_mat)
        invR = np.linalg.inv(cov_mat)
        log_constant = samples.shape[1] * math.log(2 * math.pi) + math.log(det_R)

        # vectorize means:
        means_vect = mean[:, np.newaxis]  # ############

        # Calculate differences and convert to 4D array (and its transpose):
        diff = means_vect - samples  # Shape: # means
        diff_4d = diff[:, :, np.newaxis]
        transpose_diff_4d = diff_4d.transpose(0, 1, 3, 2)

        # Calculate values inside the exponent
        inside_1 = np.einsum("abcd, dd->abcd", diff_4d, invR)
        inside_2 = np.einsum("abcd, abdc->abc", inside_1, transpose_diff_4d)
        total_inside_exponent = inside_2.transpose(2, 1, 0)
        total_inside_exponent = np.reshape(total_inside_exponent,
                                           (total_inside_exponent.shape[1], total_inside_exponent.shape[2]))

        log_likelihood = -0.5 * (log_constant + total_inside_exponent)

        # Convert likelihoods to vector:
        if log_likelihood.shape[1] == 1:
            log_likelihood = log_likelihood[:, 0]

        return log_likelihood

    def select_indexes(self, prior_samples, collocation_points):
        """

        Args:
            prior_samples: array [mc_size, n_params]
                Pre-defined samples from the parameter space, out of which the sample sets should be extracted.
            collocation_points: [tp_size, n_params]
                array with training points which were already used to train the surrogate model, and should therefore
                not be re-explored.

        Returns: array[self.mc_size,]
            With indexes of the new candidate parameter sets, to be read from the prior_samples array

        """
        n_tp = collocation_points.shape[0]
        # a) get index of elements that have already been used
        aux1_ = np.where((prior_samples[:self.mc_samples + n_tp, :] == collocation_points[:, None]).all(-1))[1]
        # b) give each element in the prior a True if it has not been used before
        aux2_ = np.invert(np.in1d(np.arange(prior_samples[:self.mc_samples + n_tp, :].shape[0]), aux1_))
        # c) Select the first d_size_bal elements in prior_sample that have not been used before
        al_unique_index = np.arange(prior_samples[:self.mc_samples + n_tp, :].shape[0])[aux2_]
        al_unique_index = al_unique_index[:self.mc_samples]

        return al_unique_index
