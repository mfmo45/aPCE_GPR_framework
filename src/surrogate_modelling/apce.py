"""

"""


import numpy as np
from numpy.polynomial.polynomial import polyval
import scipy
import sklearn
import sklearn.linear_model as lm
from tqdm import tqdm
from joblib import Parallel, delayed

from src.surrogate_modelling.additional_pce_solvers import RegressionFastARD, RegressionFastLaplace, OrthogonalMatchingPursuit


class PCEConfig:
    """
    Class generates the constant attributes for the aPCE estimations.

    Parameters:
        :param max_degree: int
            Maximum polynomial degree to consider
        :param prior_samples: np.array, [mc_size, ndim]
            with samples from the prior distribution, to build the moments and the orthogonal univariate basis functions
        :param q_norm: float
            Value to be used for sparse-truncation of terms. Default=1 (no sparsity)
    """
    def __init__(self, max_degree=1, prior_samples=None, q_norm=1):

        self.max_degree = max_degree
        self.raw_data = prior_samples
        self.q_norm = q_norm

        self.ndim = prior_samples.shape[1]

        # Initialize other attributed needed:
        self.univ_basis_indices = None
        self.non_mixed_basis_indices = None

    @staticmethod
    def sort_basis_indices(keys, graded=True, reverse=False):
        """
        Sort keys using graded lexicographical ordering. It gives the first dimension a higher importance than subsequent
        dimensions.
        Same as ``numpy.lexsort``, but also support graded and reverse lexicographical ordering.
        Args:
            keys:
                Values to sort.
            graded:
                Graded sorting, meaning the indices are always sorted by the index
                sum. E.g. ``(2, 2, 2)`` has a sum of 6, and will therefore be
                considered larger than both ``(3, 1, 1)`` and ``(1, 1, 3)``.
            reverse:
                Reverse lexicographical sorting meaning that ``(1, 3)`` is
                considered smaller than ``(3, 1)``, instead of the opposite.
        Returns:
            Array of indices that sort the keys along the specified axis.
        Examples:
            >>> indices = np.array([[0, 0, 0, 1, 2, 1],
            ...                        [1, 2, 0, 0, 0, 1]])
            >>> indices[:, np.lexsort(indices)]
            array([[0, 1, 2, 0, 1, 0],
                   [0, 0, 0, 1, 1, 2]])
            >>> indices[:, numpoly.glexsort(indices)]
            array([[0, 1, 2, 0, 1, 0],
                   [0, 0, 0, 1, 1, 2]])
            >>> indices[:, numpoly.glexsort(indices, reverse=True)]
            array([[0, 0, 0, 1, 1, 2],
                   [0, 1, 2, 0, 1, 0]])
            >>> indices[:, numpoly.glexsort(indices, graded=True)]
            array([[0, 1, 0, 2, 1, 0],
                   [0, 0, 1, 0, 1, 2]])
            >>> indices[:, numpoly.glexsort(indices, graded=True, reverse=True)]
            array([[0, 0, 1, 0, 1, 2],
                   [0, 1, 0, 2, 1, 0]])
            >>> indices = numpy.array([4, 5, 6, 3, 2, 1])
            >>> indices[numpoly.glexsort(indices)]
            array([1, 2, 3, 4, 5, 6])
        """
        keys_ = np.atleast_2d(keys)  # convert to a 2D array
        if reverse:
            keys_ = keys_[::-1]
        # get indices from smallest to largest, giving the 1st row a higher importance
        indices = np.array(np.lexsort(keys_))
        if graded:
            indices = indices[np.argsort(
                np.sum(keys_[:, indices], axis=0))].T
        return indices

    def create_multi_basis_indices(self, max_degree, q_norm=1):
        """
        Get the order combinations for all dimensions
        :param max_degree: int
            Max polynomial degree
        :param q_norm: float
            Get the order combinations for all dimensions, truncating the basis orders using the L_p norm.
            math:
            L_p(x) = \sum_i |x_i/b_i|^p ^{1/p} \leq 1
             where :math:`b_i` are bounds that each :math:`x_i` should follow
        :return:
        """
        # Arrays with smallest order (0) to the max degree + 1
        start = np.zeros(self.ndim, dtype=int)
        stop = np.full(self.ndim, max_degree + 1, dtype=int)  # Add +1 so np.arange(0, stop) fills up to degree "d"
        bound = stop.max()

        # To control the size of the arrays:
        dtype = np.uint8 if bound < 256 else np.uint16
        range_ = np.arange(bound, dtype=dtype)  # vector with values of "d" to consider
        # Initialize the indices for the first parameter (row-wise), based on the order range
        indices = range_[:, np.newaxis]  # list of orders, in order

        # Fill the combinatorics array, one dimension at a time:
        for idx in tqdm(range(self.ndim-1), ascii=True,
                           desc="Computing basis indices combinations"):
            # Repeats the current set of indices ndim times
            # e.g. [0,1,2] -> [0,1,2,0,1,2,...,0,1,2]
            indices = np.tile(indices, (bound, 1))

            # Stretches ranges over the new dimension.
            # e.g. [0,1,2] -> [0,0,...,0,1,1,...,1,2,2,...,2]
            front = range_.repeat(len(indices) // bound)[:, np.newaxis]

            # Put the array "front" in front of the previous "indices" array, to do the combinations of dimensions <= idx
            indices = np.column_stack((front, indices))

            # Truncate at each step to keep memory usage low, dor idx > 0
            if q_norm == 1:  # no sparsity
                idx_to_keep = np.sum(indices, axis=-1) <= max_degree
            else:   # with q_norm sparsity
                idx_to_keep = np.sum((indices / bound) ** q_norm, axis=-1) ** (1. / q_norm) <= 1

            indices = indices[idx_to_keep]

        # Order in descending norm value (sum of all orders), giving priority to the first dimensions
        new_order = self.sort_basis_indices(keys=indices.T, reverse=False, graded=True)
        indices = indices[new_order]

        return indices.astype(int)

    def create_uni_basis_indices(self, max_degree):
        """
        Creates a set of univariate basis indices for PCE purposes, where we ignore mixed terms and only keep
        univariate terms up to a give degree.

        Parameters
        ----------
        max_degree: int
            Maximum polynomial degree to consider.

        Returns
        -------

        """
        range_ = np.arange(max_degree+1)
        n_terms = max_degree+1

        indices = np.zeros(((n_terms*self.ndim), self.ndim))

        # Fill with the basis indices for each parameter (each column)
        i = 0  # counter for rows
        for j in range(self.ndim):
            indices[i:i+n_terms, j] = range_

            # update counter:
            i += n_terms

        # remove duplicates, and keep order of parameters
        indices = np.flip(np.unique(indices, axis=0), axis=1)

        return indices.astype(int)

    def setup_apce(self):
        """ Function sets up the main and constant attributes of the aPCE class"""

        # 1. Estimate the orthonormal, univariate basis functions based on the data
        self.polycoeffs = {}
        for parIdx in tqdm(range(self.ndim), ascii=True,
                           desc="Computing orth. polynomial coeffs"):
            poly_coeffs = self.apoly_construction(self.raw_data[:, parIdx], self.max_degree, transform=True)
            self.polycoeffs[f'p_{parIdx + 1}'] = poly_coeffs

        # Calculate basis indices (polynomial order combination of parameters), up to order 'max_degree' for:
        # Mixed terms
        self.univ_basis_indices = self.create_multi_basis_indices(max_degree=self.max_degree,
                                                                  q_norm=self.q_norm)
        # Non-mixed terms
        self.non_mixed_basis_indices = self.create_uni_basis_indices(max_degree=self.max_degree)

    @staticmethod
    def apoly_construction(Data, degree, transform=True):
        """
        Construction of Data-driven Orthonormal Polynomial Basis
        Author: Dr.-Ing. habil. Sergey Oladyshkin
        Department of Stochastic Simulation and Safety Research for Hydrosystems
        Institute for Modelling Hydraulic and Environmental Systems
        Universitaet Stuttgart, Pfaffenwaldring 5a, 70569 Stuttgart
        E-mail: Sergey.Oladyshkin@iws.uni-stuttgart.de
        http://www.iws-ls3.uni-stuttgart.de
        The current script is based on definition of arbitrary polynomial chaos
        expansion (aPC), which is presented in the following manuscript:
        Oladyshkin, S. and W. Nowak. Data-driven uncertainty quantification using
        the arbitrary polynomial chaos expansion. Reliability Engineering & System
        Safety, Elsevier, V. 106, P.  179-190, 2012.
        DOI: 10.1016/j.ress.2012.05.002.

        Parameters
        ----------
        Data : array [mc_size, ]
            Raw data.
        degree : int
            Maximum polynomial degree.
        transform : bool
            True to normalize Data, to avoid numerical issues. Default is True

        Returns
        -------
        Polynomial : array [degree+2, degree+2]
            The coefficients of the univariate orthonormal polynomials, where each row corresponds to degree i, for
            i in {0...degree+1}, and each column is the coefficient for the given j degree term, for j in {0...degree+1}

        """

        # Initialization of variables
        dd = degree + 1  # one degree more, for use in other methods (Gaussian quadrature TP selection)
        nsamples = len(Data)

        # Forward linear transformation (Avoiding numerical issues)
        if transform:
            MeanOfData = np.mean(Data)
            Data = Data / MeanOfData

        # Compute raw moments from input data
        # We estimate 2*dd + 2 to be able to solve for the coefficients'matrices
        raw_moments = [np.sum(np.power(Data, p)) / nsamples for p in range(2 * dd + 2)]
        # Step-wise
        # for p in range(2 * dd + 2):
        #     print(p)
        #     step1 = np.sum(np.power(data, p))
        #     step2 = step1/nsamples

        # Main Loop for Polynomial with degree up to dd
        PolyCoeff_NonNorm = np.empty((0, 1))
        Polynomial = np.zeros((dd + 1, dd + 1))

        for degree in range(dd + 1):
            Mm = np.zeros((degree + 1, degree + 1))
            Vc = np.zeros((degree + 1))

            # Define Moments Matrix Mm
            for i in range(degree + 1):
                for j in range(degree + 1):
                    if i < degree:
                        Mm[i, j] = raw_moments[i + j]

                    elif (i == degree) and (j == degree):
                        Mm[i, j] = 1

                # Numerical Optimization for Matrix Solver
                Mm[i] = Mm[i] / max(abs(Mm[i]))

            # Definition of Right Hand side orthogonality conditions: Vc
            for i in range(degree + 1):
                Vc[i] = 1 if i == degree else 0

            # Solution: Coefficients of Non-Normal Orthogonal Polynomial: Vp Eq.(4)
            try:
                Vp = np.linalg.solve(Mm, Vc)
            except:
                inv_Mm = np.linalg.pinv(Mm)
                Vp = np.dot(inv_Mm, Vc.T)

            if degree == 0:
                PolyCoeff_NonNorm = np.append(PolyCoeff_NonNorm, Vp)

            if degree != 0:
                if degree == 1:
                    zero = [0]
                else:
                    zero = np.zeros((degree, 1))
                PolyCoeff_NonNorm = np.hstack((PolyCoeff_NonNorm, zero))

                PolyCoeff_NonNorm = np.vstack((PolyCoeff_NonNorm, Vp))

            if 100 * abs(sum(abs(np.dot(Mm, Vp)) - abs(Vc))) > 0.5:
                print('\n---> Attention: Computational Error too high !')
                print('\n---> Problem: Convergence of Linear Solver')

            # Original Numerical Normalization of Coefficients with Norm and orthonormal Basis computation
            # Note: Polynomial(i,j) corresponds to coefficient number "j-1" of polynomial degree "i-1"
            # This evaluates the orthogonal polynomial in each sample

            P_norm = 0
            for i in range(nsamples):
                Poly = 0
                for k in range(degree + 1):
                    if degree == 0:
                        Poly += PolyCoeff_NonNorm[k] * (Data[i] ** k)
                    else:
                        Poly += PolyCoeff_NonNorm[degree, k] * (Data[i] ** k)

                P_norm += Poly ** 2 / nsamples

            P_norm = np.sqrt(P_norm)

            for k in range(degree + 1):
                if degree == 0:
                    Polynomial[degree, k] = PolyCoeff_NonNorm[k] / P_norm
                else:
                    Polynomial[degree, k] = PolyCoeff_NonNorm[degree, k] / P_norm

        # Backward linear transformation to the real data space
        if transform:
            Data *= MeanOfData
            for k in range(len(Polynomial)):
                Polynomial[:, k] = Polynomial[:, k] / (MeanOfData ** (k))

        return Polynomial


class aPCE:
    """
    Based on  MetaModel class, written by Farid Mohammadi for BayesValidRox (see )
    Attributes:
        pce_config : object
            PCEConfig class object, with aPCE constants
        pce_reg_method : str
            PCE regression method to compute the coefficients. The following
            regression methods are available:

        1. OLS: Ordinary Least Square method
        2. BRR: Bayesian Ridge Regression
        3. LARS: Least angle regression
        4. ARD: Bayesian ARD Regression
        5. FastARD: Fast Bayesian ARD Regression
        6. 'BCS': Bayesian Fast LaPlace
        7. OMP: Orthogonal Matching Pursuit

        Not yet:
        8. VBL: Variational Bayesian Learning
        9. EBL: Emperical Bayesian Learning
        Default is `OLS`.

        pce_degree = int  (ToDO: Not used right now, it must be the same as the pce_config.max_degree value. )
            Polynomial degree(s)  ToDo: If a list is given, an adaptive algorithm is used to find the best degree with
                                   the lowest Leave-One-Out cross-validation (LOO) error (or the highest score=1-LOO)
                                   ToDo: Make it such that, if it is given, pce_config is recalculated.
        collocation_points : array [n_tp, ndim]
            Input parameter sets corresponding to the input data
        model_evaluations : array [n_tp, n_obs]
            Output data, obtained from forward model, corresponding to the 'collocation_points'
        parallel : bool
            True to parallelize training, False to train one at a time
        sparsity : bool
            True, to include sparsity, False to ignore additional sparsity (if q_norm is already used)
        variance_cutoff : str
            method for sparsity (in addition to q_norm, in case it was already used)

            1. var=0: ignores all coefficients that are zero
            2. 1<var<0: orders coefficients, and removes all coefficients that account for less than var% of the
            variance.

    Generated attributes:
    psi : array [n_tp, n_terms]
        array with LHS matrix to train the surrogate, constant for each observation location
    trained_objs : list
        with output dictionaries, including:
    n_obs : int
        number of observation locations

    ToDo: Farid has the option to add a list of 'pce_degree' to do a LOOCV analysis to select best one. This could be
        added here
    ToDO: (Output) dimension reduction method using PCA

    Notes:
        We do not use an adaptive training, as in BayesValidRox_ library

    """
    def __init__(self, pce_config,
                 pce_reg_method='OLS',
                 parallel=False,
                 sparsity=False, variance_cutoff=0.0,
                 # pce_degree=1,
                 collocation_points=None, model_evaluations=None):

        self.pce_config = pce_config

        self.pce_reg_method = pce_reg_method
        self.pce_degree = pce_config.max_degree
        self.sparsity = sparsity
        self.var_cutoff = variance_cutoff

        self.training_points = collocation_points
        self.model_evaluations = model_evaluations

        # To get the error compared to training data
        self.surrogate_output = None
        self.surrogate_error = None

        self.n_obs = model_evaluations.shape[1]
        self.ndim = collocation_points.shape[1]

        self.pce_list = []

        self.parallel = parallel

    # 1
    def evaluate_univ_basis(self, samples, n_max=None):
        """
        Author: Farid Mohammadi (BayesValidRox)
        Based on MetaModel.univ_basis_vals() and evac_rec_rule.eval_univ_basis()

        Evaluates univariate regressors along input directions, for the apce case.

        Parameters
        ----------
        samples : array of shape [n_samples, n_params]
            Samples (collocation points)
        n_max : int, optional
            Maximum polynomial degree. The default is `None`.

        Returns
        -------
        univ_basis: array of shape (n_params, n_samples, n_max+1)
            All univariate regressors up to n_max.
        """
        if samples.ndim != 2:
            samples = samples.reshape(1, len(samples))

        n_samples, n_params = samples.shape
        max_degree = np.max(self.pce_degree) if n_max is None else n_max

        # Extract orthonormal basis function coefficients
        apoly_coeffs = self.pce_config.polycoeffs

        # Evaluate univariate basis
        univ_basis = np.zeros((n_params, n_samples, max_degree+1))

        for i in range(n_params):  # Evaluate one parameter (dimension) at a time.
            coeffs = apoly_coeffs[f'p_{i + 1}']
            values = np.zeros((n_samples, max_degree + 1))
            for deg in range(max_degree + 1):
                values[:, deg] = polyval(samples[:, i], coeffs[deg, :]).T
            univ_basis[i, :, :] = values

        return univ_basis

    # 2.
    @staticmethod
    def fill_psi(basis_indices, univ_p_val):
        """
        Author: Farid Mohammadi (BayesValidRox)
        Based on MetaModel.create_psi()

        This function assemble the design matrix Psi from the given basis index
        set INDICES and the univariate polynomial evaluations univ_p_val.

        Parameters
        ----------
        basis_indices : array of shape (n_terms, n_params)
            Multi-indices of multivariate polynomials.
        univ_p_val : array of [n_params, n_samples, n_max+1]
            All univariate regressors up to `n_max`, filled with 'evaluate_univ_basis'

        Raises
        ------
        ValueError
            n_terms in arguments do not match.

        Returns
        -------
        psi : array of shape (n_samples, n_terms)
            Multivariate regressors.

        """
        # Check if BasisIndices is a sparse matrix
        sparsity = scipy.sparse.issparse(basis_indices)
        if sparsity:
            basis_indices = basis_indices.toarray()

        # Initialization and consistency checks
        # number of input variables
        n_params = univ_p_val.shape[0]

        # Size of the experimental design
        n_samples = univ_p_val.shape[1]

        # number of basis terms
        n_terms = basis_indices.shape[0]

        # check that the variables have consistent sizes
        if n_params != basis_indices.shape[1]:
            raise ValueError(
                f"The shapes of basis_indices ({basis_indices.shape[1]}) and "
                f"univ_p_val ({n_params}) don't match!!"
                )

        # Preallocate the Psi matrix for performance
        psi = np.ones((n_samples, n_terms))
        # Assemble the Psi matrix
        for m in range(basis_indices.shape[1]):
            aa = np.where(basis_indices[:, m] > 0)[0]
            try:
                basisIdx = basis_indices[aa, m]
                bb = univ_p_val[m, :, basisIdx].T.reshape(psi[:, aa].shape)
                psi[:, aa] = np.multiply(psi[:, aa], bb)
            except ValueError as err:
                raise err
        return psi

    def generate_psi(self, basis_indices=None, max_degree=None, sample=None):
        """
        This function evaluates the univariate basis functions in each input parameter set 'samples' and then generates
        the corresponding PSI (LHS matrix for PCE fitting), which is independent of the output loc being evaluated.

        Parameters
        ----------
        sample : array [n_mc, n_params]
            with n_mc parameter sets to evaluate in the orthonormal basis coefficients.
        basis_indices: np.array [n_terms, n_params]
            with combination of univariate polynomials to use
        max_degree: int
            max degree of polynomial to consider

        Returns
        -------

        """
        if basis_indices is None:
            basis_indices = self.pce_config.univ_basis_indices

        # Evaluate univariate orthonormal polynomial basis functions
        basis_vals = self.evaluate_univ_basis(samples=sample, n_max=max_degree)

        # Build PSI
        psi = self.fill_psi(basis_indices=basis_indices, univ_p_val=basis_vals)

        return psi
    # ------------------------------------------------------------------------------------------------------ #

    def train(self, initial_reg_method=None):
        if self.var_cutoff == 0:
            self._train_normal()
        else:
            self._train_with_retrain(initial_reg_method=initial_reg_method)

    def _train_normal(self):
        """
        Trains a PCE model, given the PCE configuration input. It only trains the PCE once (no retrain if sparsity is
        used)
        -------
        ToDo: Apply adaptive regression, as done in BayesValidRox --> MetaModel.adaptive_regression()
        """
        # generate psi at collocation points:
        basis_indices = self.pce_config.univ_basis_indices
        psi_training = self.generate_psi(max_degree=self.pce_degree, sample=self.training_points)

        if self.parallel and self.n_obs > 1:
            out = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(self._fit)(psi=psi_training,
                                                                                    y=self.model_evaluations[:, i],
                                                                                    basis_indices=basis_indices,
                                                                                    sparsity=self.sparsity,
                                                                                    var_cutoff=self.var_cutoff,
                                                                                    )
                                                                 for i in range(self.n_obs))
            self.pce_list = out
        else:
            for i, y_ in enumerate(self.model_evaluations.T):
                output = self._fit(psi=psi_training, y=y_, basis_indices=basis_indices,
                                   sparsity=self.sparsity, var_cutoff=self.var_cutoff)
                self.pce_list.append(output)

        self.surrogate_output = self.predict_(input_sets=self.training_points)['output']
        self.surrogate_error = np.subtract(self.surrogate_output, self.model_evaluations)

        # Evaluate apce at the training points:
        # for i, y_ in enumerate(self.model_evaluations.T):
        #     self.surrogate_out[:, i] = self.predict(input_samples=self.collocation_points, i=i)[:, 0]

    def _train_with_retrain(self, initial_reg_method=None):
        """
        Train the aPCE with sparsity and, after sparsity, the aPCE is retrained with the remaining coefficients.
        -------
        """
        # Initial training
        if initial_reg_method is None:
            initial_reg_method = self.pce_reg_method

        # generate psi at collocation points:
        basis_indices = self.pce_config.univ_basis_indices
        psi_training = self.generate_psi(max_degree=self.pce_degree, sample=self.training_points)

        if self.parallel and self.n_obs > 1:
            out = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(self._fit)(psi=psi_training,
                                                                                    y=self.model_evaluations[:, i],
                                                                                    basis_indices=basis_indices,
                                                                                    sparsity=self.sparsity,
                                                                                    var_cutoff=self.var_cutoff,
                                                                                    reg_method=initial_reg_method
                                                                                    )
                                                                 for i in range(self.n_obs))
            temp_pce_list = out
            # self.pce_list = out
        else:
            temp_pce_list = []
            for i, y_ in enumerate(self.model_evaluations.T):
                output = self._fit(psi=psi_training, y=y_,
                                   basis_indices=basis_indices,
                                   sparsity=self.sparsity,
                                   var_cutoff=self.var_cutoff,
                                   reg_method=initial_reg_method)
                temp_pce_list.append(output)

        # Second/final optimization
        if self.parallel and self.n_obs > 1:
            out = Parallel(n_jobs=-1, backend='multiprocessing')(
                delayed(self._fit)(psi=temp_pce_list[i]['sparsePsi'],
                                   y=self.model_evaluations[:, i],
                                   basis_indices=temp_pce_list[i]['sparseMulti-Index'],
                                   )
                for i in range(self.n_obs))
            self.pce_list = out
        else:
            for i, y_ in enumerate(self.model_evaluations.T):
                output = self._fit(psi=temp_pce_list[i]['sparsePsi'], y=y_,
                                   basis_indices=temp_pce_list[i]['sparseMulti-Index']
                                   )
                self.pce_list.append(output)

        # Evaluate apce at the training points:
        self.surrogate_output = self.predict_(input_sets=self.training_points)
        # for i, y_ in enumerate(self.model_evaluations.T):
        #     self.surrogate_out[:, i] = self.predict(input_samples=self.collocation_points, i=i)[:, 0]

    def _fit(self, psi, y, basis_indices, reg_method=None, sparsity=True, var_cutoff=0, loo=True):
        """
        Author: Farid Mohammadi, M.Sc.
        Based on MetaModel.fit() function from BayesValidRox lib

        Fit regression using the regression method provided.

        If sparsity is to be added (e.g. if sparsity==True and varcutoff != 0, or if a non-sparse method is being used,
        then we make all coefficients to be removed (ignored) 0, but we do not create "sparse" matrices.

        Parameters
        ----------
        psi : array of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array of shape (n_samples,)
            Target values.
        basis_indices : array of shape (n_terms, n_params)
            Multi-indices of multivariate polynomials.
        reg_method : str, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        return_out_dict : Dict
            Fitted estimator, spareMulti-Index, sparseX and coefficients.

        ToDo: bias_term is taken as True, since no PCA has been implemented.
        """
        if reg_method is None:
            reg_method = self.pce_reg_method

        # bias_term = self.dim_red_method.lower() != 'pca'
        bias_term = True

        # compute_score = True if self.verbose else False
        compute_score = True

        #  inverse of the observed variance of the data
        if np.var(y) != 0:
            Lambda = 1 / np.var(y)
        else:
            Lambda = 1e-6

        # Bayes sparse adaptive aPCE
        if reg_method.lower() == 'ols':
            # ordinary least squares method: equivalent to (scipy.linalg.pinv(np.dot(psi.T, psi)).dot(psi.T)).dot(y)
            clf_poly = lm.LinearRegression(fit_intercept=False)
        if reg_method.lower() == 'nnls':
            clf_poly = lm.LinearRegression(positive=True, fit_intercept=False)
        elif reg_method.lower() == 'rr':
            # ridge regression: L2 regularization
            clf_poly = lm.Ridge(alpha=1)
        elif reg_method.lower() == 'brr':
            clf_poly = lm.BayesianRidge(n_iter=1000, tol=1e-7,
                                        fit_intercept=False,
                                        compute_score=compute_score,
                                        alpha_1=1e-04, alpha_2=1e-04,
                                        lambda_1=Lambda, lambda_2=Lambda)
            clf_poly.converged = True

        elif reg_method.lower() == 'ard':
            clf_poly = lm.ARDRegression(fit_intercept=False,
                                        compute_score=compute_score,
                                        n_iter=1000, tol=0.0001,
                                        alpha_1=1e-3, alpha_2=1e-3,
                                        lambda_1=Lambda, lambda_2=Lambda)

        elif reg_method.lower() == 'fastard':
            clf_poly = RegressionFastARD(fit_intercept=False,
                                         normalize=True,
                                         compute_score=compute_score,
                                         n_iter=300, tol=1e-10)

        elif reg_method.lower() == 'bcs':
            clf_poly = RegressionFastLaplace(fit_intercept=False,
                                             bias_term=bias_term,
                                             n_Kfold=min(self.ndim, 10),
                                             n_iter=1000, tol=1e-7)

        elif reg_method.lower() == 'lars':
            clf_poly = lm.LassoLarsCV(fit_intercept=False)

        elif reg_method.lower() == 'sgdr':
            clf_poly = lm.SGDRegressor(fit_intercept=False,
                                       max_iter=5000, tol=1e-7)

        elif reg_method.lower() == 'omp':
            clf_poly = OrthogonalMatchingPursuit(fit_intercept=False)

        # elif reg_method.lower() == 'vbl':
        #     clf_poly = VBLinearRegression(fit_intercept=False)
        #
        # elif reg_method.lower() == 'ebl':
        #     clf_poly = EBLinearRegression(optimizer='em')

        # Fit
        # psi_new = np.dot(psi.T, psi) + np.diag(np.ones(psi.shape[1])*5)
        # y_new = np.dot(psi.T, y)
        #
        # clf_poly.fit(psi_new, y_new)

        clf_poly.fit(psi, y)

        # Select the nonzero entries of coefficients (make sparse)
        if sparsity:
            if var_cutoff == 0:
                # remove all coefficients which are 0: only valid (will do sth) if a sparse solver is used.
                # Original way done in BayesValidRox
                nnz_idx = np.nonzero(clf_poly.coef_)[0]
            else:
                var_cutoff = np.abs(var_cutoff)
                descending_idx = (-(clf_poly.coef_ ** 2)).argsort()  # index in descending order
                var_ordered = clf_poly.coef_[descending_idx] ** 2    # arrange coefficients in descending order
                var_sum = np.cumsum(var_ordered) / np.sum(var_ordered)  # normalize: sobol indices
                if var_cutoff <= 1:
                    # If cutoff between 0 and 1, it is a variance percentage, so we cut off the coefficients to either
                    # N = cutoff_variance or N = number of TP (whichever is greater), so it is not under-determined.
                    # cutoff = max(np.argmin(np.abs(var_sum - var_cutoff)), psi.shape[0])
                    cutoff = np.argmin(np.abs(var_sum - var_cutoff))
                    nnz_idx = np.sort(descending_idx[0:cutoff+1])
                else:
                    # If cutoff is greater than 1, cutoff to the number of terms = number of TP (determined system)
                    nnz_idx = np.sort(descending_idx[0: psi.shape[0]])
                    max_variance = np.sum(clf_poly.coef_[nnz_idx] ** 2)/np.sum(clf_poly.coef_**2)  # quantify max var

        else:
            nnz_idx = np.arange(clf_poly.coef_.shape[0])

        # This is for the case where all outputs are zero, so all coefficients are zero
        if (y == 0).all():
            nnz_idx = np.insert(np.nonzero(clf_poly.coef_)[0], 0, 0)

        # Extract and save the sparse basis_indices(combinations that have an influence), sparse PSI (corresponding to
        # non-zero columns, for all CP, sparse indices --> save them in the regression object.
        sparse_basis_indices = basis_indices[nnz_idx]
        sparse_psi = psi[:, nnz_idx]
        sparse_coeffs = clf_poly.coef_[nnz_idx]

        original_coeffs = clf_poly.coef_

        sparse_coeffs_with_zeros = np.zeros(clf_poly.coef_.shape)
        sparse_coeffs_with_zeros[nnz_idx] = sparse_coeffs

        # overwrite the coefficients for the sparse coefficients, for evaluations (and also save the new basis_indices)
        if var_cutoff == 0: # no retrain / final training
            clf_poly.coef_ = sparse_coeffs
            if loo:
                # Evaluate fit
                score, LCerror = self.corr_loocv_error(clf=clf_poly, psi=sparse_psi, coeffs=sparse_coeffs, y=y)

        else:
            clf_poly.coef_ = sparse_coeffs_with_zeros
            if loo:
                # Evaluate fir
                score, LCerror = self.corr_loocv_error(clf=clf_poly, psi=psi, coeffs=sparse_coeffs_with_zeros, y=y)

        # Create a dict to pass the needed outputs (always send the sparse-results, even if no sparsity is used)
        return_out_dict = dict()
        return_out_dict['clf_poly'] = clf_poly
        # return_out_dict['Multi-Index'] = basis_indices
        return_out_dict['Multi-Index'] = sparse_basis_indices
        # return_out_dict['coeffs'] = sparse_coeffs_with_zeros
        return_out_dict['coeffs'] = sparse_coeffs

        return_out_dict['sparseMulti-Index'] = sparse_basis_indices
        return_out_dict['sparsePsi'] = sparse_psi

        return_out_dict['sparse_coeffs'] = sparse_coeffs
        return_out_dict['full_coeffs'] = original_coeffs
        return_out_dict['loocv_score'] = score
        return_out_dict['loocv_errors'] = LCerror

        return return_out_dict

    # ----------------------------------------------------------------------------------------------------- #

    def predict_(self, input_sets, get_conf_int=False, i=None):
        """
        Function evaluates the PCE surrogate for a given output loc "i", on a set of input parameter sets.

        Parameters
        ----------
        input_sets: np.array [mc_size, n_params]
            with input data sets
        i: int
            index for output location. Default is None, and the prediction for all output locations is estimated
        get_conf_int: bool
            True to return confidence intervals

        Returns np.array [mc_samples,]
            with surrogate model outputs for each input parameter set

        -------

        """
        if i is None:
            start = 0
            end = self.n_obs
        else:
            start = i
            end = i+1

        sm_predictions = np.zeros((input_sets.shape[0], int(end-start)))
        sm_std = np.zeros((input_sets.shape[0], int(end - start)))

        for Idx in range(start, end):
            # Extract the corresponding trained object
            obj = self.pce_list[Idx]

            # NEW:
            psi_predict = self.generate_psi(max_degree=self.pce_degree, sample=input_sets,
                                            basis_indices=obj['Multi-Index'])
            # Predict for input_samples
            try:
                sm_predictions[:, Idx], sm_std[:, Idx] = obj['clf_poly'].predict(psi_predict, return_std=True)
            except:
                try:
                    sm_predictions[:, Idx] = obj['clf_poly'].predict(psi_predict)
                except:
                    print('An exception occurred, prediction is estimated manually')
                    sm_predictions[:, Idx] = np.dot(psi_predict, obj['coeffs'])
                get_conf_int = False

        output_dict = {'output': sm_predictions,
                       'std': sm_std}
        if get_conf_int:
            output_dict['upper_ci'] = sm_predictions + 2*sm_std
            output_dict['lower_ci'] = sm_predictions + 2 * sm_std
        return output_dict

    def validate_surrogate(self, method='loocv', validation_input=None, validation_output=None):
        """

        Parameters
        ----------
        method: str
            Method used for validation. Options: 'loocv', 'validation'
        validation_input: array[mc_size, n_p]
            With parameter set used for validation
        validation_output: array[mc_size, n_obs]
            With FCM output values for each validation_input parameter set.

        Returns float
            Average error from all output locations, for the given input method
        -------

        Notes
            'loocv' method uses the method proposed by Blatman(2009), using linear superimposition of orthogonal terms
            'validation' method uses a validation set, and compares the model outputs vs surogate outputs.
        """
        if method == 'loocv':  # Uses Blatman et.al. (2009,2011) method implemented in BayesValidRox
            score_ = []
            LCerror_ = []
        elif method == 'validation':
            score_ = []

        for i, obj in enumerate(self.trained_objs):
            if method == 'loocv':
                score, LCerror = self.corr_loocv_error(obj['clf_poly'], obj['sparePsi'], obj['coeffs'],
                                                       validation_output[:, i])
                score_.append(score)
                LCerror_.append(LCerror)
            elif method == 'validation':
                model_predictions = self.predict(input_sets=validation_input, i=i)
                score_.append(self.validation_error(model_predictions, validation_output[:, i]))

        return np.mean(score_)

    @staticmethod
    def corr_loocv_error(clf, psi, coeffs, y):
        """
        Calculates the corrected LOO error for regression on regressor
        matrix `psi` that generated the coefficients based on [1] and [2].

        [1] Blatman, G., 2009. Adaptive sparse polynomial chaos expansions for
            uncertainty propagation and sensitivity analysis (Doctoral
            dissertation, Clermont-Ferrand 2).

        [2] Blatman, G. and Sudret, B., 2011. Adaptive sparse polynomial chaos
            expansion based on least angle regression. Journal of computational
            Physics, 230(6), pp.2345-2367.

        Author: Farid Mohammadi, M.Sc. (BayesValidRox)

        Parameters
        ----------
        clf : object
            Fitted estimator.
        psi : array of shape (n_samples, n_features)
            The multivariate orthogonal polynomials (regressor).
        coeffs : array-like of shape (n_features,)
            Estimated cofficients.
        y : array of shape (n_samples,)
            Target values.

        Returns
        -------
        R_2 : float
            LOOCV Validation score (1-LOOCV erro).
        residual : array of shape (n_samples,)
            Residual values (y - predicted targets).
        """
        psi = np.array(psi, dtype=float)

        # Create PSI_Sparse by removing redundent terms
        nnz_idx = np.nonzero(coeffs)[0]
        if len(nnz_idx) == 0:
            nnz_idx = [0]
        psi_sparse = psi[:, nnz_idx]

        # NrCoeffs of aPCEs
        P = len(nnz_idx)
        # NrEvaluation (Size of experimental design)
        N = psi.shape[0]

        # Build the projection matrix
        PsiTPsi = np.dot(psi_sparse.T, psi_sparse)

        if np.linalg.cond(PsiTPsi) > 1e-12: #and \
           # np.linalg.cond(PsiTPsi) < 1/sys.float_info.epsilon:
            # faster
            M = scipy.linalg.solve(PsiTPsi,
                                   scipy.sparse.eye(PsiTPsi.shape[0]).toarray())
        else:
            # stabler
            M = np.linalg.pinv(PsiTPsi)

        # h factor (the full matrix is not calculated explicitly,
        # only the trace is, to save memory)
        PsiM = np.dot(psi_sparse, M)

        h = np.sum(np.multiply(PsiM, psi_sparse), axis=1, dtype=np.float64)

        # ------ Calculate Error Loocv for each measurement point ----
        # Residuals
        try:
            residual = clf.predict(psi) - y
        except:
            residual = np.dot(psi, coeffs) - y

        # Variance
        var_y = np.var(y)

        if var_y == 0:
            norm_emp_error = 0
            loo_error = 0
            LCerror = np.zeros((y.shape))
            return 1-loo_error, LCerror
        else:
            norm_emp_error = np.mean(residual**2)/var_y

            # LCerror = np.divide(residual, (1-h))
            LCerror = residual / (1-h)
            loo_error = np.mean(np.square(LCerror)) / var_y
            # if there are NaNs, just return an infinite LOO error (this
            # happens, e.g., when a strongly underdetermined problem is solved)
            if np.isnan(loo_error):
                loo_error = np.inf

        # Corrected Error for over-determined system
        tr_M = np.trace(M)
        if tr_M < 0 or abs(tr_M) > 1e6:
            tr_M = np.trace(np.linalg.pinv(np.dot(psi.T, psi)))

        # Over-determined system of Equation
        if N > P:
            T_factor = N/(N-P) * (1 + tr_M)

        # Under-determined system of Equation
        else:
            T_factor = np.inf

        corrected_loo_error = loo_error * T_factor

        R_2 = 1 - corrected_loo_error

        return R_2, LCerror


def validation_error(true_y, sim_y, output_names, n_per_type):
    """
    Estimates different evaluation (validation) criteria for a surrogate model, for each output location. Results for
    each output type are saved under different keys in a dictionary.
    Args:
        true_y: array [mc_valid, n_obs]
            simulator outputs for valid_samples
        sim_y: array [mc_valid, n_obs] or dict{}
            surrogate/emulator's outputs for valid_samples. If a dict is given, it has output and std keys.
        output_names: array [n_types,]
            with strings, with name of each output
        n_per_type: int
            Number of observation per output type

    Returns: float, float or array[n_obs], float or array[n_obs]
        with validation criteria for each output locaiton, and each output type

    ToDo: Like in BayesValidRox, estimate surrogate predictions here, by giving a surrogate object as input (maybe)
    ToDo: add as part of MyGeneralGPR class, and the outputs are a dictionary, with output type as a key.
    """
    criteria_dict = {'rmse': dict(),
                     'mse': dict(),
                     'nse': dict(),
                     'r2': dict(),
                     'mean_error': dict(),
                     'std_error': dict()}

    if isinstance(sim_y, dict):
        if 'upper_ci' in sim_y.keys():
            sm_out = sim_y['output']
            sm_std = sim_y['std']
            upper_ci = sim_y['upper_ci']
            lower_ci = sim_y['lower_ci']

            criteria_dict['norm_error'] = dict()
            criteria_dict['P95'] = dict()
        else:
            sm_out = sim_y['output']

    # RMSE for each output location: not a dictionary (yet). [n_obs, ]
    rmse = sklearn.metrics.mean_squared_error(y_true=true_y, y_pred=sm_out, multioutput='raw_values',
                                              squared=False)

    c = 0
    for i, key in enumerate(output_names):
        # RMSE
        criteria_dict['rmse'][key] = sklearn.metrics.mean_squared_error(y_true=true_y[:, c:c + n_per_type],
                                                                        y_pred=sm_out[:, c:c + n_per_type],
                                                                        multioutput='raw_values', squared=False)
        criteria_dict['mse'][key] = sklearn.metrics.mean_squared_error(y_true=true_y[:, c:c + n_per_type],
                                                                       y_pred=sm_out[:, c:c + n_per_type],
                                                                       multioutput='raw_values', squared=True)
        # NSE
        criteria_dict['nse'][key] = sklearn.metrics.r2_score(y_true=true_y[:, c:c + n_per_type],
                                                             y_pred=sm_out[:, c:c + n_per_type],
                                                             multioutput='raw_values')
        # Mean errors
        criteria_dict['mean_error'][key] = np.abs(
            np.mean(true_y[:, c:c + n_per_type], axis=0) - np.mean(sm_out[:, c:c + n_per_type], axis=0)) / np.mean(
            true_y[:, c:c + n_per_type], axis=0)

        criteria_dict['std_error'][key] = np.abs(
            np.std(true_y[:, c:c + n_per_type], axis=0) - np.std(sm_out[:, c:c + n_per_type], axis=0)) / np.std(
            true_y[:, c:c + n_per_type], axis=0)

        # # Validation error:
        # criteria_dict['valid_error'][key] = criteria_dict['rmse'][key] ** 2 / np.var(true_y[:, c:c+n_per_type],
        #                                                                              ddof=1, axis=0)

        # Norm error: only if std is available
        if isinstance(sim_y, dict) and 'upper_ci' in sim_y.keys():
            # Normalized error
            ind_val = np.divide(np.subtract(sm_out[:, c:c + n_per_type], true_y[:, c:c + n_per_type]),
                                sm_std[:, c:c + n_per_type])
            criteria_dict['norm_error'][key] = np.mean(ind_val ** 2, axis=0)

            # P95
            p95 = np.where((true_y[:, c:c + n_per_type] <= upper_ci[:, c:c + n_per_type]) & (
                    true_y[:, c:c + n_per_type] >= lower_ci[:, c:c + n_per_type]), 1, 0)
            criteria_dict['P95'][key] = np.mean(p95, axis=0)

        # R2
        criteria_dict['r2'][key] = np.zeros(n_per_type)
        for j in range(n_per_type):
            criteria_dict['r2'][key][j] = np.corrcoef(true_y[:, j + c], sm_out[:, j + c])[0, 1]

        c = c + n_per_type

    return rmse, criteria_dict


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
