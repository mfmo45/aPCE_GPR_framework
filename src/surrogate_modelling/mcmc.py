"""
Based on bayesValidRox MCMC class

Currently has only been tested without multiprocessing.
"""
import emcee
import os
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import multiprocessing
import scipy


class MCMC:
    def __init__(self, observations, error_var,
                 exp_design, surrogate_object,
                 mcmc_opts,
                 include_error=False,
                 init_samples=None,
                 output_dir='mcmc_', n_cpus=None):

        self.observations = observations
        self.var = error_var
        self.calculate_constants()

        self.initsamples = init_samples

        self.sm = surrogate_object
        self.exp_design = exp_design

        self.mcmc_params = mcmc_opts
        self.n_cpus = n_cpus

        self.include_error = include_error

        self.output_dir = output_dir

    def run_sampler(self):

        ndim = len(self.exp_design.Inputs.Marginals)

        output_dir = f'{self.output_dir}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Set MCMC parameters:
        self.initsamples = None
        self.nwalkers = 100
        self.nburn = 300
        self.nsteps = 100000
        self.moves = None

        self.mp = False
        self.verbose = False

        # Not yet used:
        # self.move_name = 'stretch'
        # scale = 1

        # Read from mcmc_opts:
        # Extract number of steps per walker
        if 'n_steps' in self.mcmc_params:
            self.nsteps = int(self.mcmc_params['n_steps'])
        # Extract number of walkers (chains)
        if 'n_walkers' in self.mcmc_params:
            self.nwalkers = int(self.mcmc_params['n_walkers'])
        # Extract moves
        if 'moves' in self.mcmc_params:
            self.moves = self.mcmc_params['moves']
        if 'move_name' in self.mcmc_params:
            self.move_name = self.mcmc_params['move_name']
        if 'scale' in self.mcmc_params:
            scale = self.mcmc_params['scale']
        # Extract multiprocessing
        if 'multiprocessing' in self.mcmc_params:
            self.mp = self.mcmc_params['multiprocessing']
        # Extract verbose
        if 'verbose' in self.mcmc_params:
            self.verbose = self.mcmc_params['verbose']

        # Set initial samples: randomly
        init_samples = self.exp_design.generate_samples(n_samples=self.nwalkers)

        print("\n>>>> Bayesian inference with MCMC sarted <<<<<<")
        #   f"{self.BayesOpts.name} started. <<<<<<")

        # Set up the backend
        filename = f"{self.output_dir}/emcee_sampler.h5"
        backend = emcee.backends.HDFBackend(filename)
        # Clear the backend in case the file already exists
        backend.reset(self.nwalkers, ndim)

        if self.mp:   # For Multiprocessing
            if self.n_cpus is None:
                n_cpus = multiprocessing.cpu_count()
            else:
                n_cpus = self.n_cpus

            with multiprocessing.Pool(n_cpus) as pool:
                sampler = emcee.EnsembleSampler(
                    self.nwalkers, ndim, self.log_posterior, moves=self.moves,
                    pool=pool, backend=backend
                )

                # Check if a burn-in phase is needed!
                if self.initsamples is None:
                    # Burn-in
                    print("\n Burn-in period is starting:")
                    pos = sampler.run_mcmc(
                        init_samples, self.nburn, progress=True
                    )

                    # Reset sampler
                    sampler.reset()
                    pos = pos.coords
                else:
                    pos = self.initsamples

                # Production run
                print("\n Production run is starting:")
                pos, prob, state = sampler.run_mcmc(
                    pos, self.nsteps, progress=True
                )
        else:
            # Run in series and monitor the convergence
            sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self.log_posterior,
                                            moves=self.moves,
                                            backend=backend, vectorize=True)

            # Burnout period:
            print("\n Burn-in period is starting:")
            pos = sampler.run_mcmc(init_samples, self.nburn, progress=True)

            # Reset sampler
            sampler.reset()
            pos = pos.coords

            print("\n Production run is starting:")
            # Track how the average autocorrelation time estimate changes
            autocorrIdx = 0
            autocorr = np.empty(self.nsteps)
            tauold = np.inf
            autocorreverynsteps = 50
            adapteverynsteps = 50

            # sample step by step using the generator sampler.sample
            for sample in sampler.sample(pos,
                                         iterations=self.nsteps,
                                         tune=True,
                                         progress=True):

                # only check convergence every autocorreverynsteps steps
                if sampler.iteration % autocorreverynsteps:
                    continue

                if self.verbose:
                    print("\nStep: {}".format(sampler.iteration))
                    acc_fr = np.mean(sampler.acceptance_fraction)
                    print(f"Mean acceptance fraction: {acc_fr:.3f}")

                # compute the autocorrelation time so far using tol=0 means that we'll always get an estimate even if
                # it isn't trustworthy
                tau = sampler.get_autocorr_time(tol=0)
                # average over walkers
                autocorr[autocorrIdx] = np.nanmean(tau)
                autocorrIdx += 1
                # output current autocorrelation estimate
                if self.verbose:
                    print(f"Mean autocorr. time estimate: {np.nanmean(tau):.3f}")
                    list_gr = np.round(self.gelman_rubin(sampler.chain), 3)
                    print("Gelman-Rubin Test*: ", list_gr)

                # check convergence
                converged = np.all(tau * autocorreverynsteps < sampler.iteration)
                converged &= np.all(np.abs(tauold - tau) / tau < 0.01)
                converged &= np.all(self.gelman_rubin(sampler.chain) < 1.1)

                if converged:
                    print('converged')
                    break
                tauold = tau

        # Posterior diagnostics
        try:
            tau = sampler.get_autocorr_time(tol=0)
        except emcee.autocorr.AutocorrError:
            tau = 5

        if all(np.isnan(tau)):
            tau = 5

        burnin = int(2 * np.nanmax(tau))
        thin = int(0.5 * np.nanmin(tau)) if int(0.5 * np.nanmin(tau)) != 0 else 1
        finalsamples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
        acc_fr = np.nanmean(sampler.acceptance_fraction)
        # list_gr = np.round(self.gelman_rubin(sampler.chain[:, burnin:]), 3)

        # Print summary
        print('\n')
        print('-' * 15 + 'Posterior diagnostics' + '-' * 15)
        print(f"Mean auto-correlation time: {np.nanmean(tau):.3f}")
        print(f"Thin: {thin}")
        print(f"Burn-in: {burnin}")
        print(f"Flat chain shape: {finalsamples.shape}")
        print(f"Mean acceptance fraction*: {acc_fr:.3f}")
        print("Gelman-Rubin Test**: ", list_gr)

        print("\n* This value must lay between 0.234 and 0.5.")
        print("** These values must be smaller than 1.1.")
        print('-' * 50)

        print(f"\n>>>> Bayesian inference with MCMC  "
              "successfully completed. <<<<<<\n")

        # Save and return
        self.sampler = sampler
        self.finalsamples = finalsamples
        Posterior_df = pd.DataFrame(finalsamples)

        return Posterior_df

    def log_posterior(self, samples):
        """
        Calls the functions to estimate the log likliehood and log prior for each input sample, and returns the sum
        of them to the emcee sampler
        Args:
            samples: (np.array [n_samples, n_dim]): vector with input parameter set(s)

        Returns:

        """
        samples = samples if samples.ndim != 1 else samples.reshape((1, -1))
        nsamples = samples.shape[0]

        log_prior = self.log_prior(samples)
        log_likelihood = self.log_likelihood(samples)

        # print(f'Prior: {log_prior.shape} - Likelihood: {log_likelihood.shape}')
        return log_prior + log_likelihood

    def log_prior(self, samples):
        """
        Estimates the log pdf from the prior distribution
        Args:
            samples (np.array [n_samples, n_dim]): vector with input parameter set(s)
        """

        prior_dist = self.exp_design.JDist
        # params_range = self.BayesOpts.engine.ExpDesign.bound_tuples
        samples = samples if samples.ndim != 1 else samples.reshape((1, -1))
        nsamples = samples.shape[0]
        logprior = -np.inf * np.ones(nsamples)

        for i in range(nsamples):
            logprior[i] = np.log(prior_dist.pdf(samples[i, :]))

        if nsamples == 1:
            return logprior[0]
        else:
            return logprior

    def log_likelihood(self, samples):
        """
        Function estimates the log likelihood for a single sample or a set of samples (from different walkers)
        Args:
            samples: np.array [n_samples, ndim] or [ndim, ]
                With samples to evaluate in the model and obtain the likelihood for

        Returns: np.array(n_samples, ) with the loglikelihood for each input sample
        """

        samples = samples if samples.ndim != 1 else samples.reshape((1, -1))
        nsamples = samples.shape[0]

        out = self.sm.predict_(input_sets=samples)
        mean_pred, std_pred = out['output'], out['std']

        if self.include_error:
            log_likelihood = self.log_likelihood_error(model_predictions=mean_pred,
                                                       model_std=std_pred)
            # print(f'With error: likelihood shape {log_likelihood.shape}')
        else:
            log_likelihood = scipy.stats.multivariate_normal.pdf(mean_pred, cov=self.cov_mat,
                                                                 mean=self.observations[0, :])
            # print(f'NO error: likelihood shape {log_likelihood.shape}')

        return log_likelihood

    def log_likelihood_error(self, model_predictions, model_std):
        """
        Function calculates likelihood between observations and the model output manually, using numpy calculations. It
        considers model error, with an error associated to each model prediction.

        Notes:
        * Generates likelihood array with size [MCxN], where N is the number of measurement data sets.
        * Likelihood function is multivariate normal distribution, considering independent and Gaussian-distributed
        errors.
        * Method is faster than using stats module ('calculate_likelihood' function).

        Args:
            model_predictions: np.array [n_samples, n_out] with output values from the model
            model_std: np.array [n_Samples, n_out] with standard deviation values from the model (to augment the
            covariance function)

        Returns: np.array [n_Samples,] with log likelihood values for each input sample

        """
        # Calculate augmented covariance:
        mc_size = model_predictions.shape[0]
        cov_3d = np.tile(self.cov_mat[np.newaxis, :, :], (mc_size, 1, 1))  # make 3D (1 cov per MC run)
        std_3d = np.array([np.diag(row) for row in model_std])  # make 3D matrix for std
        augmented_cov = cov_3d + std_3d ** 2  # combine covariances

        det_R = np.linalg.det(augmented_cov)
        invR = np.linalg.inv(augmented_cov)

        # Can't ignore constant
        log_constant = self.observations.shape[1] * math.log(2 * math.pi) + np.log(det_R.reshape(-1, 1))

        # vectorize means:
        means_vect = self.observations[:, np.newaxis]  # ############

        # Calculate differences and convert to 4D array (and its transpose):
        diff = means_vect - model_predictions  # Shape: # means
        diff_4d = diff[:, :, np.newaxis]
        transpose_diff_4d = diff_4d.transpose(0, 1, 3, 2)

        # Calculate values inside the exponent
        inside_1 = np.einsum("abcd, bdd->abcd", diff_4d, invR)
        inside_2 = np.einsum("abcd, abdc->abc", inside_1, transpose_diff_4d)
        total_inside_exponent = inside_2.transpose(2, 1, 0)
        total_inside_exponent = np.reshape(total_inside_exponent,
                                           (total_inside_exponent.shape[1], total_inside_exponent.shape[2]))

        log_likelihood = -0.5 * (log_constant + total_inside_exponent)

        # Convert likelihoods to vector:
        if log_likelihood.shape[1] == 1:
            log_likelihood = log_likelihood[:, 0]

        return log_likelihood

    def calculate_constants(self):
        """
        Calculates the covariance matrix based on the input variable "error", which is a vector of variances, one for
        each observation point.

        :return: None
        """
        if type(self.var) is not np.ndarray:
            self.var = np.array([self.var])
        self.cov_mat = np.diag(self.var)

    def gelman_rubin(self, chain, return_var=False):
        """
        The potential scale reduction factor (PSRF) defined by the variance
        within one chain, W, with the variance between chains B.
        Both variances are combined in a weighted sum to obtain an estimate of
        the variance of a parameter \\( \\theta \\).The square root of the
        ratio of this estimates variance to the within chain variance is called
        the potential scale reduction.
        For a well converged chain it should approach 1. Values greater than
        1.1 typically indicate that the chains have not yet fully converged.

        Source: http://joergdietrich.github.io/emcee-convergence.html

        https://github.com/jwalton3141/jwalton3141.github.io/blob/master/assets/posts/ESS/rwmh.py

        Parameters
        ----------
        chain : array (n_walkers, n_steps, n_params)
            The emcee ensamples.

        Returns
        -------
        R_hat : float
            The Gelman-Robin values.

        """
        m_chains, n_iters = chain.shape[:2]

        # Calculate between-chain variance
        θb = np.mean(chain, axis=1)
        θbb = np.mean(θb, axis=0)
        B_over_n = ((θbb - θb) ** 2).sum(axis=0)
        B_over_n /= (m_chains - 1)

        # Calculate within-chain variances
        ssq = np.var(chain, axis=1, ddof=1)
        W = np.mean(ssq, axis=0)

        # (over) estimate of variance
        var_θ = W * (n_iters - 1) / n_iters + B_over_n

        if return_var:
            return var_θ
        else:
            # The square root of the ratio of this estimates variance to the
            # within chain variance
            R_hat = np.sqrt(var_θ / W)
            return R_hat
