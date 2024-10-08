import numpy as np
import chaospy
import scipy.stats as stats
import sys
import math
# from tqdm import tqdm


class ExpDesign:
    """
    Based on ExpDesign class object in BayesValidRox library (see https://pypi.org/project/bayesvalidrox/)
    Author: Farid Mohammadi, M.Sc.
    Institute for Modelling Hydraulic and Environmental Systems (IWS), University
    of Stuttgart

    This class generates samples from the prescribed marginals for the model
    parameters using the `Input` object.

    Attributes
    ----------
    Inputs : obj
        Input object containing the parameter marginals, i.e. name,
        distribution type and distribution parameters or available raw data.
    training_method : str
        Type of the experimental design. The default is `'normal'`. Other
        option is `'sequential'`.
    main_meta_model : str
        Type of the meta_model.
    secondary_meta_model : str
        Type of the meta_model used for the error model. Default is None (no secondary is used)
    sampling_method : str
        Name of the sampling method for the experimental design. The following
        sampling method are supported:

        * random
        * latin_hypercube
        * sobol
        * halton
        * hammersley
        * chebyshev(FT)
        * grid(FT)
        * user

    n_init_tp : int
        Number of (initial) training points.
    n_max_tp : int
        Number of maximum training points.
    pce_mc_size : int
        Number of parameter sets to sample to determine raw moments from data (for apce methods)
    exploit_method : str
        Type of the exploitation method for the sequential design. The following methods are supported:

        * space_filling
        * BAL
        * sobol (temporarily)

    explore_method : str
        Method to select training points for sequential design. The following methods are supported:

        * random
        * latin_hypercube
        * sobol
        * Voronoi

    training_step : int
        how many samples to take during each training iteration. Default=1 (for BAL)
    util_func : str
        The utility function for the exploit_method (if BAL is selected)
        Option available:

        * DKL
        * BME
        * IE
        * DKL_BME

    eval_step : int
        Every how many iterations to evaluate the model. Default is 1: for only apce and for space_filling exploration.

    do_validation = bool
        True to do validation, False to do cross-validation
    val_x = array [val_size, ndim]          ToDo: Dictionary?
        with parameter sets to use for validation. Default is None, of do_validation==False
    val_y = array [val_size, ndim]          ToDo: Dictionary with key=output type
        with model outputs for val_x. Default is None, of do_validation==False
    Generated attributes:
    ---------------------
    ndim : int
        number of uncertain parameters
    apce : bool
        True if aPCE is the main meta model
    JDist : object
        chaospy JDist object, with multivariate probability distribution
    poly_types : list
        with polynomial type for each input parameter, used for PCE generation
    n_iter : int
        Number of additional TP to sample through sequential methods (e.g. space_filling or BAL)
    """

    def __init__(self, input_object, training_method='normal', sampling_method='random', main_meta_model='apce',
                 secondary_meta_model=None, exploit_method='BAL', explore_method='random',
                 training_step=1,
                 n_initial_tp=1, n_max_tp=None, pce_mc_size=50_000,
                 util_func=None,
                 eval_step=1):

        self.Inputs = input_object
        self.training_method = training_method
        self.main_meta_model = main_meta_model
        self.secondary_model = secondary_meta_model
        self.sampling_method = sampling_method
        self.exploit_method = exploit_method
        self.training_step = training_step
        self.util_func = util_func

        self.explore_method = explore_method

        self.eval_step = eval_step

        self.n_init_tp = n_initial_tp
        self.n_max_tp = n_max_tp
        self.pce_mc_size = pce_mc_size

        self.do_validation = True      # true to do validation, False to do cross-validation
        self.val_x = None
        self.val_y = None

        self.ndim = None
        self.apce = None
        self.input_data_given = None
        self.JDist = None
        self.poly_types = None
        self.n_iter = None
        self.n_evals = None

        # Not used yet:

    def setup_ED_(self):
        """
        Based on ExpDesign.generateED() and ExpDesign.init_param_space() from BayesValidRox
        Author: Farid Mohammadi, M.Sc.

        Function set up the main variables needed to characterize the experimental design. Mainly:
        - Sets number of parameters
        - Sets if aPCE is being used (for further functions)
        - Sets if input data is given in Inputs (determines whether samples are read or taken)
        - Generates a multivariate prob. distribution as a chaospy JDist object, to be used for taking samples
        - Generates a list with the type of polynomial associated to each parameter

        Returns
        -------

        """

        # Extract info
        Inputs = self.Inputs
        self.ndim = len(Inputs.Marginals)

        if self.main_meta_model.lower() == 'apce':
            self.apce = True
        else:
            self.apce = False

        # Check if input is given as dist or input_data.
        if len(Inputs.Marginals[0].input_data):
            self.input_data_given = True
        else:
            self.input_data_given = False

        # Get the bounds if input_data are directly defined by user:
        if self.input_data_given:
            for i in range(self.ndim):
                low_bound = np.min(Inputs.Marginals[i].input_data)
                up_bound = np.max(Inputs.Marginals[i].input_data)
                Inputs.Marginals[i].parameters = [low_bound, up_bound]

        # Generate multivariate probability distribution based on Input class object
        self.JDist, self.poly_types = self.build_dist()

        # Set number of training iterations - additional TP to sample (for BAL)
        if self.training_method == 'normal':
            self.n_max_tp = self.n_init_tp
            self.training_step = 1

        self.n_iter = int((self.n_max_tp - self.n_init_tp)/self.training_step)

        # Number of evaluations:
        if self.eval_step == 1 or self.exploit_method == 'sobol':
            self.n_evals = self.n_iter + 1
            self.eval_step = 1
        else:
            self.n_evals = math.ceil(self.n_iter/self.eval_step) + 1

    def build_dist(self, rosenblatt=False):
        """
        Based on ExpDesign.build_dist() from BayesValidRox Library (see https://pypi.org/project/bayesvalidrox/)
        Author: Farid Mohammadi, M.Sc.

        Creates the multivariate probability distribution based on the Input data and a list with the polynomial types
        to be used for the PCE basis function coefficients.

        Parameters
        ----------
        rosenblatt: bool
            True for dependent parameters, False for iid

        Returns
        -------
        orig_space_dist : object
            A chaospy JDist object or a gaussian_kde object.
        poly_types : list
            List of polynomial types for the parameters.
        """

        Inputs = self.Inputs
        all_data = []
        all_dist_types = []
        orig_joints = []
        poly_types = []

        for parIdx in range(self.ndim):

            if Inputs.Marginals[parIdx].dist_type is None:
                data = Inputs.Marginals[parIdx].input_data
                all_data.append(data)
                dist_type = None
            else:
                dist_type = Inputs.Marginals[parIdx].dist_type
                params = Inputs.Marginals[parIdx].parameters

            if rosenblatt:
                polytype = 'hermite'
                dist = chaospy.Normal()

            elif dist_type is None:
                polytype = 'arbitrary'
                dist = None

            elif 'unif' in dist_type.lower():
                polytype = 'legendre'
                dist = chaospy.Uniform(lower=params[0], upper=params[1])

            elif 'norm' in dist_type.lower() and \
                    'log' not in dist_type.lower():
                polytype = 'hermite'
                dist = chaospy.Normal(mu=params[0], sigma=params[1])

            elif 'gamma' in dist_type.lower():
                polytype = 'laguerre'
                dist = chaospy.Gamma(shape=params[0],
                                     scale=params[1],
                                     shift=params[2])

            elif 'beta' in dist_type.lower():
                polytype = 'jacobi'
                dist = chaospy.Beta(alpha=params[0], beta=params[1],
                                    lower=params[2], upper=params[3])

            elif 'lognorm' in dist_type.lower():
                polytype = 'hermite'
                mu = np.log(params[0] ** 2 / np.sqrt(params[0] ** 2 + params[1] ** 2))
                sigma = np.sqrt(np.log(1 + params[1] ** 2 / params[0] ** 2))
                dist = chaospy.LogNormal(mu, sigma)
                # dist = chaospy.LogNormal(mu=params[0], sigma=params[1])

            elif 'expon' in dist_type.lower():
                polytype = 'arbitrary'
                dist = chaospy.Exponential(scale=params[0], shift=params[1])

            elif 'weibull' in dist_type.lower():
                polytype = 'arbitrary'
                dist = chaospy.Weibull(shape=params[0], scale=params[1],
                                       shift=params[2])

            else:
                message = (f"DistType {dist_type} for parameter"
                           f"{parIdx + 1} is not available.")
                raise ValueError(message)

            if self.input_data_given or self.apce:
                polytype = 'arbitrary'

            # Store dists and poly_types
            orig_joints.append(dist)
            poly_types.append(polytype)
            all_dist_types.append(dist_type)

        # Prepare final output to return
        if None in all_dist_types:
            # Naive approach: Fit a gaussian kernel to the provided data
            Data = np.asarray(all_data)
            orig_space_dist = stats.gaussian_kde(Data)
            # self.prior_space = orig_space_dist
        else:
            orig_space_dist = chaospy.J(*orig_joints)
            # self.prior_space = stats.gaussian_kde(orig_space_dist.sample(10000))

        return orig_space_dist, poly_types

    def generate_samples(self, n_samples, sampling_method='random',
                         transform=False):
        """
        Based on ExpDesign.generate_samples() from BayesValidRox lib (https://pypi.org/project/bayesvalidrox/)
        Author: Farid Mohammadi, M.Sc.

        Generates samples with given sampling method

        Parameters
        ----------
        n_samples : int
            Number of requested samples.
        sampling_method : str, optional
            Sampling method. The default is `'random'`.
        transform : bool, optional
            Transformation via an isoprobabilistic transformation method. The
            default is `False`.

        Returns
        -------
        samples: array of shape (n_samples, n_params)
            Generated samples from defined model input object.

        ToDO: Add possibility to transform data to original space (find benefir of this)
        """
        try:
            samples = chaospy.generate_samples(
                int(n_samples), domain=self.JDist, rule=sampling_method
                )
        except:
            samples = self.random_sampler(int(n_samples)).T

        return samples.T

    def random_sampler(self, n_samples):
        """
        Based on ExpDesign.random_sampler() from BayesValidRox lib (https://pypi.org/project/bayesvalidrox/)
        Author: Farid Mohammadi, M.Sc.

        Samples the given raw data randomly.

        Parameters
        ----------
        n_samples : int
            Number of requested samples.

        Returns
        -------
        samples: array of shape (n_samples, n_params)
            The sampling locations in the input space.

        """
        samples = np.zeros((n_samples, self.ndim))
        sample_size = self.raw_data.shape[1]

        # Use a combination of raw data
        if n_samples < sample_size:
            for pa_idx in range(self.ndim):
                # draw random indices
                rand_idx = np.random.randint(0, sample_size, n_samples)
                # store the raw data with given random indices
                samples[:, pa_idx] = self.raw_data[pa_idx, rand_idx]
        else:
            try:
                samples = self.JDist.resample(int(n_samples)).T
            except AttributeError:
                samples = self.JDist.sample(int(n_samples)).T
            # Check if all samples are in the bound_tuples
            for idx, param_set in enumerate(samples):
                if not self._check_ranges(param_set, self.bound_tuples):
                    try:
                        proposed_sample = chaospy.generate_samples(
                            1, domain=self.JDist, rule='random').T[0]
                    except:
                        proposed_sample = self.JDist.resample(1).T[0]
                    while not self._check_ranges(proposed_sample,
                                                 self.bound_tuples):
                        try:
                            proposed_sample = chaospy.generate_samples(
                                1, domain=self.JDist, rule='random').T[0]
                        except:
                            proposed_sample = self.JDist.resample(1).T[0]
                    samples[idx] = proposed_sample

        return samples

    def transform(self, X, params=None, method=None):
        """
        Author: Farid Mohammadi, M.Sc.
        Transform the samples via either a Rosenblatt or an isoprobabilistic
        transformation.

        Parameters
        ----------
        X : array of shape (n_samples,n_params)
            Samples to be transformed.
        method : string
            If transformation method is 'user' transform X, else just pass X.

        Returns
        -------
        tr_X: array of shape (n_samples,n_params)
            Transformed samples.

        """
        if self.Inputs.Rosenblatt:
            self.origJDist, _ = self.build_dist(False)
            if method == 'user':
                tr_X = self.JDist.inv(self.origJDist.fwd(X.T)).T
            else:
                # Inverse to original spcace -- generate sample ED
                tr_X = self.origJDist.inv(self.JDist.fwd(X.T)).T
        else:
            # Transform samples via an isoprobabilistic transformation
            n_samples, n_params = X.shape
            Inputs = self.Inputs
            origJDist = self.JDist
            poly_types = self.poly_types

            disttypes = []
            for par_i in range(n_params):
                disttypes.append(Inputs.Marginals[par_i].dist_type)

            # Pass non-transformed X, if arbitrary PCE is selected.
            if None in disttypes or self.input_data_given or self.apce:
                return X

            cdfx = np.zeros((X.shape))
            tr_X = np.zeros((X.shape))

            for par_i in range(n_params):

                # Extract the parameters of the original space
                disttype = disttypes[par_i]
                if disttype is not None:
                    dist = origJDist[par_i]
                else:
                    dist = None
                polytype = poly_types[par_i]
                cdf = np.vectorize(lambda x: dist.cdf(x))

                # Extract the parameters of the transformation space based on
                # polyType
                if polytype == 'legendre' or disttype == 'uniform':
                    # Generate Y_Dists based
                    params_Y = [-1, 1]
                    dist_Y = stats.uniform(loc=params_Y[0],
                                        scale=params_Y[1]-params_Y[0])
                    inv_cdf = np.vectorize(lambda x: dist_Y.ppf(x))

                elif polytype == 'hermite' or disttype == 'norm':
                    params_Y = [0, 1]
                    dist_Y = stats.norm(loc=params_Y[0], scale=params_Y[1])
                    inv_cdf = np.vectorize(lambda x: dist_Y.ppf(x))

                elif polytype == 'laguerre' or disttype == 'gamma':
                    params_Y = [1, params[1]]
                    dist_Y = stats.gamma(loc=params_Y[0], scale=params_Y[1])
                    inv_cdf = np.vectorize(lambda x: dist_Y.ppf(x))

                # Compute CDF_x(X)
                cdfx[:, par_i] = cdf(X[:, par_i])

                # Compute invCDF_y(cdfx)
                tr_X[:, par_i] = inv_cdf(cdfx[:, par_i])

        return tr_X

