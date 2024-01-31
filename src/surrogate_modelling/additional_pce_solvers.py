"""
Author: Farid Mohammadi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 19:41:45 2020

@author: farid
"""
import numpy as np
from scipy.linalg import solve_triangular
from numpy.linalg import LinAlgError
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel
import warnings
from sklearn.utils import check_X_y
from scipy.linalg import pinvh

from sklearn.utils import as_float_array
from sklearn.model_selection import KFold


def update_precisions(Q, S, q, s,A, active, tol, n_samples, clf_bias):
    '''
    Selects one feature to be added/recomputed/deleted to model based on
    effect it will have on value of log marginal likelihood.
    '''
    # initialise vector holding changes in log marginal likelihood
    deltaL = np.zeros(Q.shape[0])

    # identify features that can be added , recomputed and deleted in model
    theta        =  q**2 - s
    add          =  (theta > 0) * (active == False)
    recompute    =  (theta > 0) * (active == True)
    delete       = ~(add + recompute)

    # compute sparsity & quality parameters corresponding to features in
    # three groups identified above
    Qadd,Sadd      = Q[add], S[add]
    Qrec,Srec,Arec = Q[recompute], S[recompute], A[recompute]
    Qdel,Sdel,Adel = Q[delete], S[delete], A[delete]

    # compute new alpha's (precision parameters) for features that are
    # currently in model and will be recomputed
    Anew           = s[recompute]**2/ ( theta[recompute] + np.finfo(np.float32).eps)
    delta_alpha    = (1./Anew - 1./Arec)

    # compute change in log marginal likelihood
    deltaL[add]       = ( Qadd**2 - Sadd ) / Sadd + np.log(Sadd/Qadd**2 )
    deltaL[recompute] = Qrec**2 / (Srec + 1. / delta_alpha) - np.log(1 + Srec*delta_alpha)
    deltaL[delete]    = Qdel**2 / (Sdel - Adel) - np.log(1 - Sdel / Adel)
    deltaL            = deltaL  / n_samples

    # find feature which caused largest change in likelihood
    feature_index = np.argmax(deltaL)

    # no deletions or additions
    same_features  = np.sum( theta[~recompute] > 0) == 0

    # changes in precision for features already in model is below threshold
    no_delta       = np.sum( abs( Anew - Arec ) > tol ) == 0
    # if same_features: print(abs( Anew - Arec ))
    # print("same_features = {} no_delta = {}".format(same_features,no_delta))
    # check convergence: if no features to add or delete and small change in
    #                    precision for current features then terminate
    converged = False
    if same_features and no_delta:
        converged = True
        return [A,converged]

    # if not converged update precision parameter of weights and return
    if theta[feature_index] > 0:
        A[feature_index] = s[feature_index]**2 / theta[feature_index]
        if active[feature_index] == False:
            active[feature_index] = True
    else:
        # at least two active features
        if active[feature_index] == True and np.sum(active) >= 2:
            # do not remove bias term in classification
            # (in regression it is factored in through centering)
            if not (feature_index == 0 and clf_bias):
                active[feature_index] = False
                A[feature_index]      = np.PINF

    return [A,converged]


class RegressionFastARD(LinearModel, RegressorMixin):
    '''
    Regression with Automatic Relevance Determination (Fast Version uses
    Sparse Bayesian Learning)
    https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/rvm_ard_models/fast_rvm.py

    Parameters
    ----------
    n_iter: int, optional (DEFAULT = 100)
        Maximum number of iterations

    start: list, optional (DEFAULT = None)
        Initial selected features.

    tol: float, optional (DEFAULT = 1e-3)
        If absolute change in precision parameter for weights is below threshold
        algorithm terminates.

    fit_intercept : boolean, optional (DEFAULT = True)
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    copy_X : boolean, optional (DEFAULT = True)
        If True, X will be copied; else, it may be overwritten.

    compute_score : bool, default=False
        If True, compute the log marginal likelihood at each iteration of the
        optimization.

    verbose : boolean, optional (DEFAULT = FALSE)
        Verbose mode when fitting the model

    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)

    alpha_ : float
       estimated precision of the noise

    active_ : array, dtype = bool, shape = (n_features)
       True for non-zero coefficients, False otherwise

    lambda_ : array, shape = (n_features)
       estimated precisions of the coefficients

    sigma_ : array, shape = (n_features, n_features)
        estimated covariance matrix of the weights, computed only
        for non-zero coefficients

    scores_ : array-like of shape (n_iter_+1,)
        If computed_score is True, value of the log marginal likelihood (to be
        maximized) at each iteration of the optimization.

    References
    ----------
    [1] Fast marginal likelihood maximisation for sparse Bayesian models
    (Tipping & Faul 2003) (http://www.miketipping.com/papers/met-fastsbl.pdf)
    [2] Analysis of sparse Bayesian learning (Tipping & Faul 2001)
        (http://www.miketipping.com/abstracts.htm#Faul:NIPS01)
    '''

    def __init__(self, n_iter=300, start=None, tol=1e-3, fit_intercept=True,
                 normalize=False, copy_X=True, compute_score=False, verbose=False):
        self.n_iter          = n_iter
        self.start           = start
        self.tol             = tol
        self.scores_         = list()
        self.fit_intercept   = fit_intercept
        self.normalize       = normalize
        self.copy_X          = copy_X
        self.compute_score   = compute_score
        self.verbose         = verbose

    def _preprocess_data(self, X, y):
        """Center and scale data.
        Centers data to have mean zero along axis 0. If fit_intercept=False or
        if the X is a sparse matrix, no centering is done, but normalization
        can still be applied. The function returns the statistics necessary to
        reconstruct the input data, which are X_offset, y_offset, X_scale, such
        that the output
            X = (X - X_offset) / X_scale
        X_scale is the L2 norm of X - X_offset.
        """

        if self.copy_X:
            X = X.copy(order='K')

        y = np.asarray(y, dtype=X.dtype)

        if self.fit_intercept:
            X_offset = np.average(X, axis=0)
            X -= X_offset
            if self.normalize:
                X_scale = np.ones(X.shape[1], dtype=X.dtype)
                std = np.sqrt(np.sum(X**2, axis=0)/(len(X)-1))
                X_scale[std != 0] = std[std != 0]
                X /= X_scale
            else:
                X_scale = np.ones(X.shape[1], dtype=X.dtype)
            y_offset = np.mean(y)
            y = y - y_offset
        else:
            X_offset = np.zeros(X.shape[1], dtype=X.dtype)
            X_scale = np.ones(X.shape[1], dtype=X.dtype)
            if y.ndim == 1:
                y_offset = X.dtype.type(0)
            else:
                y_offset = np.zeros(y.shape[1], dtype=X.dtype)

        return X, y, X_offset, y_offset, X_scale

    def fit(self, X, y):
        """
       Fits ARD Regression with Sequential Sparse Bayes Algorithm.

        Parameters
        -----------
        X: {array-like, sparse matrix} of size (n_samples, n_features)
           Training data, matrix of explanatory variables

        y: array-like of size [n_samples, n_features]
           Target values

        Returns
        -------
        self : object
            Returns self.
        """

        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        n_samples, n_features = X.shape

        X, y, X_mean, y_mean, X_std = self._preprocess_data(X, y)
        self._x_mean_ = X_mean
        self._y_mean = y_mean
        self._x_std = X_std

        #  precompute X'*Y , X'*X for faster iterations & allocate memory for
        #  sparsity & quality vectors
        XY = np.dot(X.T, y)
        XX = np.dot(X.T, X)
        XXd = np.diag(XX)

        #  initialise precision of noise & and coefficients
        var_y = np.var(y)

        # check that variance is non zero !!!
        if var_y == 0:
            beta = 1e-2
            self.var_y = True
        else:
            beta = 1. / np.var(y)
            self.var_y = False

        A = np.PINF * np.ones(n_features)
        active = np.zeros(n_features, dtype=bool)

        if self.start is not None and not hasattr(self, 'active_'):
            start = self.start
            # start from a given start basis vector
            proj = XY**2 / XXd
            active[start] = True
            A[start] = XXd[start]/(proj[start] - var_y)

        else:
            # in case of almost perfect multicollinearity between some features
            # start from feature 0
            if np.sum(XXd - X_mean**2 < np.finfo(np.float32).eps) > 0:
                A[0] = np.finfo(np.float16).eps
                active[0] = True

            else:
                # start from a single basis vector with largest projection on
                # targets
                proj = XY**2 / XXd
                start = np.argmax(proj)
                active[start] = True
                A[start] = XXd[start]/(proj[start] - var_y +
                                       np.finfo(np.float32).eps)

        warning_flag = 0
        scores_ = []
        for i in range(self.n_iter):
            # Handle variance zero
            if self.var_y:
                A[0] = y_mean
                active[0] = True
                converged = True
                break

            XXa = XX[active, :][:, active]
            XYa = XY[active]
            Aa = A[active]

            # mean & covariance of posterior distribution
            Mn, Ri, cholesky = self._posterior_dist(Aa, beta, XXa, XYa)
            if cholesky:
                Sdiag = np.sum(Ri**2, 0)
            else:
                Sdiag = np.copy(np.diag(Ri))
                warning_flag += 1

            # raise warning in case cholesky fails
            if warning_flag == 1:
                warnings.warn(("Cholesky decomposition failed! Algorithm uses "
                               "pinvh, which is significantly slower, if you "
                               "use RVR it is advised to change parameters of "
                               "kernel"))

            # compute quality & sparsity parameters
            s, q, S, Q = self._sparsity_quality(XX, XXd, XY, XYa, Aa, Ri,
                                                active, beta, cholesky)

            # update precision parameter for noise distribution
            rss = np.sum((y - np.dot(X[:, active], Mn))**2)

            # if near perfect fit , then terminate
            if (rss / n_samples/var_y) < self.tol:
                warnings.warn('Early termination due to near perfect fit')
                converged = True
                break
            beta = n_samples - np.sum(active) + np.sum(Aa * Sdiag)
            beta /= rss
            # beta /= (rss + np.finfo(np.float32).eps)

            # update precision parameters of coefficients
            A, converged = update_precisions(Q, S, q, s, A, active, self.tol,
                                             n_samples, False)

            if self.compute_score:
                scores_.append(self.log_marginal_like(XXa, XYa, Aa, beta))

            if self.verbose:
                print(('Iteration: {0}, number of features '
                       'in the model: {1}').format(i, np.sum(active)))

            if converged or i == self.n_iter - 1:
                if converged and self.verbose:
                    print('Algorithm converged !')
                break

        # after last update of alpha & beta update parameters
        # of posterior distribution
        XXa, XYa, Aa = XX[active, :][:, active], XY[active], A[active]
        Mn, Sn, cholesky = self._posterior_dist(Aa, beta, XXa, XYa, True)
        self.coef_ = np.zeros(n_features)
        self.coef_[active] = Mn
        self.sigma_ = Sn
        self.active_ = active
        self.lambda_ = A
        self.alpha_ = beta
        self.converged = converged
        if self.compute_score:
            self.scores_ = np.array(scores_)

        # set intercept_
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_std
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_.T)
        else:
            self.intercept_ = 0.
        return self

    def log_marginal_like(self, XXa, XYa, Aa, beta):
        """Computes the log of the marginal likelihood."""
        N, M = XXa.shape
        A = np.diag(Aa)

        Mn, sigma_, cholesky = self._posterior_dist(Aa, beta, XXa, XYa,
                                                    full_covar=True)

        C = sigma_ + np.dot(np.dot(XXa.T, np.linalg.pinv(A)), XXa)

        score = np.dot(np.dot(XYa.T, np.linalg.pinv(C)), XYa) +\
            np.log(np.linalg.det(C)) + N * np.log(2 * np.pi)

        return -0.5 * score

    def predict(self, X, return_std=False):
        '''
        Computes predictive distribution for test set.
        Predictive distribution for each data point is one dimensional
        Gaussian and therefore is characterised by mean and variance based on
        Ref.[1] Section 3.3.2.

        Parameters
        -----------
        X: {array-like, sparse} (n_samples_test, n_features)
           Test data, matrix of explanatory variables

        Returns
        -------
        : list of length two [y_hat, var_hat]

             y_hat: numpy array of size (n_samples_test,)
                    Estimated values of targets on test set (i.e. mean of
                    predictive distribution)

                var_hat: numpy array of size (n_samples_test,)
                    Variance of predictive distribution
        References
        ----------
        [1] Bishop, C. M. (2006). Pattern recognition and machine learning.
        springer.
        '''

        y_hat = np.dot(X, self.coef_) + self.intercept_

        if return_std:
            # Handle the zero variance case
            if self.var_y:
                return y_hat, np.zeros_like(y_hat)

            if self.normalize:
                X -= self._x_mean_[self.active_]
                X /= self._x_std[self.active_]
            var_hat = 1./self.alpha_
            var_hat += np.sum(X.dot(self.sigma_) * X, axis=1)
            std_hat = np.sqrt(var_hat)
            return y_hat, std_hat
        else:
            return y_hat

    def _posterior_dist(self, A, beta, XX, XY, full_covar=False):
        """
        Calculates mean and covariance matrix of posterior distribution
        of coefficients.
        """
        # compute precision matrix for active features
        Sinv = beta * XX
        np.fill_diagonal(Sinv, np.diag(Sinv) + A)
        cholesky = True

        # try cholesky, if it fails go back to pinvh
        try:
            # find posterior mean : R*R.T*mean = beta*X.T*Y
            # solve(R*z = beta*X.T*Y) =>find z=> solve(R.T*mean = z)=>find mean
            R = np.linalg.cholesky(Sinv)
            Z = solve_triangular(R, beta*XY, check_finite=True, lower=True)
            Mn = solve_triangular(R.T, Z, check_finite=True, lower=False)

            # invert lower triangular matrix from cholesky decomposition
            Ri = solve_triangular(R, np.eye(A.shape[0]), check_finite=False,
                                  lower=True)
            if full_covar:
                Sn = np.dot(Ri.T, Ri)
                return Mn, Sn, cholesky
            else:
                return Mn, Ri, cholesky
        except LinAlgError:
            cholesky = False
            Sn = pinvh(Sinv)
            Mn = beta*np.dot(Sinv, XY)
            return Mn, Sn, cholesky

    def _sparsity_quality(self, XX, XXd, XY, XYa, Aa, Ri, active, beta, cholesky):
        '''
        Calculates sparsity and quality parameters for each feature

        Theoretical Note:
        -----------------
        Here we used Woodbury Identity for inverting covariance matrix
        of target distribution
        C    = 1/beta + 1/alpha * X' * X
        C^-1 = beta - beta^2 * X * Sn * X'
        '''
        bxy = beta*XY
        bxx = beta*XXd
        if cholesky:
            # here Ri is inverse of lower triangular matrix obtained from
            # cholesky decomp
            xxr = np.dot(XX[:, active], Ri.T)
            rxy = np.dot(Ri, XYa)
            S = bxx - beta**2 * np.sum(xxr**2, axis=1)
            Q = bxy - beta**2 * np.dot(xxr, rxy)
        else:
            # here Ri is covariance matrix
            XXa = XX[:, active]
            XS = np.dot(XXa, Ri)
            S = bxx - beta**2 * np.sum(XS*XXa, 1)
            Q = bxy - beta**2 * np.dot(XS, XYa)
        # Use following:
        # (EQ 1) q = A*Q/(A - S) ; s = A*S/(A-S)
        # so if A = np.PINF q = Q, s = S
        qi = np.copy(Q)
        si = np.copy(S)
        # If A is not np.PINF, then it should be 'active' feature => use (EQ 1)
        Qa, Sa = Q[active], S[active]
        qi[active] = Aa * Qa / (Aa - Sa)
        si[active] = Aa * Sa / (Aa - Sa)

        return [si, qi, S, Q]

# ------------------------------------------------------------------------


class RegressionFastLaplace:
    '''
    Sparse regression with Bayesian Compressive Sensing as described in Alg. 1
    (Fast Laplace) of Ref.[1], which updated formulas from [2].

    sigma2: noise precision (sigma^2)
    nu fixed to 0

    uqlab/lib/uq_regression/BCS/uq_bsc.m

    Parameters
    ----------
    n_iter: int, optional (DEFAULT = 1000)
        Maximum number of iterations

    tol: float, optional (DEFAULT = 1e-7)
        If absolute change in precision parameter for weights is below
        threshold algorithm terminates.

    fit_intercept : boolean, optional (DEFAULT = True)
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    copy_X : boolean, optional (DEFAULT = True)
        If True, X will be copied; else, it may be overwritten.

    verbose : boolean, optional (DEFAULT = FALSE)
        Verbose mode when fitting the model

    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)

    alpha_ : float
       estimated precision of the noise

    active_ : array, dtype = bool, shape = (n_features)
       True for non-zero coefficients, False otherwise

    lambda_ : array, shape = (n_features)
       estimated precisions of the coefficients

    sigma_ : array, shape = (n_features, n_features)
        estimated covariance matrix of the weights, computed only
        for non-zero coefficients

    References
    ----------
    [1] Babacan, S. D., Molina, R., & Katsaggelos, A. K. (2009). Bayesian
        compressive sensing using Laplace priors. IEEE Transactions on image
        processing, 19(1), 53-63.
    [2] Fast marginal likelihood maximisation for sparse Bayesian models
        (Tipping & Faul 2003).
        (http://www.miketipping.com/papers/met-fastsbl.pdf)
    '''

    def __init__(self, n_iter=1000, n_Kfold=10, tol=1e-7, fit_intercept=False,
                 bias_term=True, copy_X=True, verbose=False):
        self.n_iter = n_iter
        self.n_Kfold = n_Kfold
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.bias_term = bias_term
        self.copy_X = copy_X
        self.verbose = verbose

    def _center_data(self, X, y):
        ''' Centers data'''
        X = as_float_array(X, copy=self.copy_X)

        # normalisation should be done in preprocessing!
        X_std = np.ones(X.shape[1], dtype=X.dtype)
        if self.fit_intercept:
            X_mean = np.average(X, axis=0)
            y_mean = np.average(y, axis=0)
            X -= X_mean
            y -= y_mean
        else:
            X_mean = np.zeros(X.shape[1], dtype=X.dtype)
            y_mean = 0. if y.ndim == 1 else np.zeros(y.shape[1], dtype=X.dtype)
        return X, y, X_mean, y_mean, X_std

    def fit(self, X, y):

        k_fold = KFold(n_splits=self.n_Kfold)

        varY = np.var(y, ddof=1) if np.var(y, ddof=1) != 0 else 1.0
        sigma2s = len(y)*varY*(10**np.linspace(-16, -1, self.n_Kfold))

        errors = np.zeros((len(sigma2s), self.n_Kfold))
        for s, sigma2 in enumerate(sigma2s):
            for k, (train, test) in enumerate(k_fold.split(X, y)):
                self.fit_(X[train], y[train], sigma2)
                errors[s, k] = np.linalg.norm(
                    y[test] - self.predict(X[test])
                    )**2/len(test)

        KfCVerror = np.sum(errors, axis=1)/self.n_Kfold/varY
        i_minCV = np.argmin(KfCVerror)

        self.kfoldCVerror = np.min(KfCVerror)

        return self.fit_(X, y, sigma2s[i_minCV])

    def fit_(self, X, y, sigma2):

        N, P = X.shape
        # n_samples, n_features = X.shape

        X, y, X_mean, y_mean, X_std = self._center_data(X, y)
        self._x_mean_ = X_mean
        self._y_mean = y_mean
        self._x_std = X_std

        # check that variance is non zero !!!
        if np.var(y) == 0:
            self.var_y = True
        else:
            self.var_y = False
        beta = 1./sigma2

        #  precompute X'*Y , X'*X for faster iterations & allocate memory for
        #  sparsity & quality vectors X=Psi
        PsiTY = np.dot(X.T, y)
        PsiTPsi = np.dot(X.T, X)
        XXd = np.diag(PsiTPsi)

        # initialize with constant regressor, or if that one does not exist,
        # with the one that has the largest correlation with Y
        ind_global_to_local = np.zeros(P, dtype=np.int32)

        # identify constant regressors
        constidx = np.where(~np.diff(X, axis=0).all(axis=0))[0]

        if self.bias_term and constidx.size != 0:
            ind_start = constidx[0]
            ind_global_to_local[ind_start] = True
        else:
            # start from a single basis vector with largest projection on
            # targets
            proj = np.divide(np.square(PsiTY), XXd)
            ind_start = np.argmax(proj)
            ind_global_to_local[ind_start] = True

        num_active = 1
        active_indices = [ind_start]
        deleted_indices = []
        bcs_path = [ind_start]
        gamma = np.zeros(P)
        # for the initial value of gamma(ind_start), use the RVM formula
        #   gamma = (q^2 - s) / (s^2)
        # and the fact that initially s = S = beta*Psi_i'*Psi_i and q = Q =
        # beta*Psi_i'*Y
        gamma[ind_start] = np.square(PsiTY[ind_start])
        gamma[ind_start] -= sigma2 * PsiTPsi[ind_start, ind_start]
        gamma[ind_start] /= np.square(PsiTPsi[ind_start, ind_start])

        Sigma = 1. / (beta * PsiTPsi[ind_start, ind_start]
                      + 1./gamma[ind_start])

        mu = Sigma * PsiTY[ind_start] * beta
        tmp1 = beta * PsiTPsi[ind_start]
        S = beta * np.diag(PsiTPsi).T - Sigma * np.square(tmp1)
        Q = beta * PsiTY.T - mu*(tmp1)

        tmp2 = np.ones(P)  # alternative computation for the initial s,q
        q0tilde = PsiTY[ind_start]
        s0tilde = PsiTPsi[ind_start, ind_start]
        tmp2[ind_start] = s0tilde / (q0tilde**2) / beta
        s = np.divide(S, tmp2)
        q = np.divide(Q, tmp2)
        Lambda = 2*(num_active - 1) / np.sum(gamma)

        Delta_L_max = []
        for i in range(self.n_iter):
            # Handle variance zero
            if self.var_y:
                mu = np.mean(y)
                break

            if self.verbose:
                print('    lambda = {0:.6e}\n'.format(Lambda))

            # Calculate the potential updated value of each gamma[i]
            if Lambda == 0.0:  # RVM
                gamma_potential = np.multiply((
                    (q**2 - s) > Lambda),
                    np.divide(q**2 - s, s**2)
                    )
            else:
                a = Lambda * s**2
                b = s**2 + 2*Lambda*s
                c = Lambda + s - q**2
                gamma_potential = np.multiply(
                    (c < 0), np.divide(
                        -b + np.sqrt(b**2 - 4*np.multiply(a, c)), 2*a)
                    )

            l_gamma = - np.log(np.absolute(1 + np.multiply(gamma, s)))
            l_gamma += np.divide(np.multiply(q**2, gamma),
                                 (1 + np.multiply(gamma, s)))
            l_gamma -= Lambda*gamma  # omitted the factor 1/2

            # Contribution of each updated gamma(i) to L(gamma)
            l_gamma_potential = - np.log(
                np.absolute(1 + np.multiply(gamma_potential, s))
                )
            l_gamma_potential += np.divide(
                np.multiply(q**2, gamma_potential),
                (1 + np.multiply(gamma_potential, s))
                )
            # omitted the factor 1/2
            l_gamma_potential -= Lambda*gamma_potential

            # Check how L(gamma) would change if we replaced gamma(i) by the
            # updated gamma_potential(i), for each i separately
            Delta_L_potential = l_gamma_potential - l_gamma

            # deleted indices should not be chosen again
            if len(deleted_indices) != 0:
                values = -np.inf * np.ones(len(deleted_indices))
                Delta_L_potential[deleted_indices] = values

            Delta_L_max.append(np.nanmax(Delta_L_potential))
            ind_L_max = np.nanargmax(Delta_L_potential)

            # in case there is only 1 regressor in the model and it would now
            # be deleted
            if len(active_indices) == 1 and ind_L_max == active_indices[0] \
               and gamma_potential[ind_L_max] == 0.0:
                Delta_L_potential[ind_L_max] = -np.inf
                Delta_L_max[i] = np.max(Delta_L_potential)
                ind_L_max = np.argmax(Delta_L_potential)

            # If L did not change significantly anymore, break
            if Delta_L_max[i] <= 0.0 or\
                    (i > 0 and all(np.absolute(Delta_L_max[i-1:])
                                   < sum(Delta_L_max)*self.tol)) or \
                    (i > 0 and all(np.diff(bcs_path)[i-1:] == 0.0)):
                if self.verbose:
                    print('Increase in L: {0:.6e} (eta = {1:.3e})\
                          -- break\n'.format(Delta_L_max[i], self.tol))
                break

            # Print information
            if self.verbose:
                print('    Delta L = {0:.6e} \n'.format(Delta_L_max[i]))

            what_changed = int(gamma[ind_L_max] == 0.0)
            what_changed -= int(gamma_potential[ind_L_max] == 0.0)

            # Print information
            if self.verbose:
                if what_changed < 0:
                    print(f'{i+1} - Remove regressor #{ind_L_max+1}..\n')
                elif what_changed == 0:
                    print(f'{i+1} - Recompute regressor #{ind_L_max+1}..\n')
                else:
                    print(f'{i+1} - Add regressor #{ind_L_max+1}..\n')

            # --- Update all quantities ----
            if what_changed == 1:
                # adding a regressor

                # update gamma
                gamma[ind_L_max] = gamma_potential[ind_L_max]

                Sigma_ii = 1.0 / (1.0/gamma[ind_L_max] + S[ind_L_max])
                try:
                    x_i = np.matmul(
                        Sigma, PsiTPsi[active_indices, ind_L_max].reshape(-1, 1)
                        )
                except ValueError:
                    x_i = Sigma * PsiTPsi[active_indices, ind_L_max]
                tmp_1 = - (beta * Sigma_ii) * x_i
                Sigma = np.vstack(
                    (np.hstack(((beta**2 * Sigma_ii) * np.dot(x_i, x_i.T)
                                + Sigma, tmp_1)), np.append(tmp_1.T, Sigma_ii))
                    )
                mu_i = Sigma_ii * Q[ind_L_max]
                mu = np.vstack((mu - (beta * mu_i) * x_i, mu_i))

                tmp2_1 = PsiTPsi[:, ind_L_max] - beta * np.squeeze(
                    np.matmul(PsiTPsi[:, active_indices], x_i)
                    )
                if i == 0:
                    tmp2_1[0] /= 2
                tmp2 = beta * tmp2_1.T
                S = S - Sigma_ii * np.square(tmp2)
                Q = Q - mu_i * tmp2

                num_active += 1
                ind_global_to_local[ind_L_max] = num_active
                active_indices.append(ind_L_max)
                bcs_path.append(ind_L_max)

            elif what_changed == 0:
                # recomputation
                # zero if regressor has not been chosen yet
                if not ind_global_to_local[ind_L_max]:
                    raise Exception('cannot recompute index{0} -- not yet\
                                    part of the model!'.format(ind_L_max))
                Sigma = np.atleast_2d(Sigma)
                mu = np.atleast_2d(mu)
                gamma_i_new = gamma_potential[ind_L_max]
                gamma_i_old = gamma[ind_L_max]
                # update gamma
                gamma[ind_L_max] = gamma_potential[ind_L_max]

                # index of regressor in Sigma
                local_ind = ind_global_to_local[ind_L_max]-1

                kappa_i = (1.0/gamma_i_new - 1.0/gamma_i_old)
                kappa_i = 1.0 / kappa_i
                kappa_i += Sigma[local_ind, local_ind]
                kappa_i = 1 / kappa_i
                Sigma_i_col = Sigma[:, local_ind]

                Sigma = Sigma - kappa_i * (Sigma_i_col * Sigma_i_col.T)
                mu_i = mu[local_ind]
                mu = mu - (kappa_i * mu_i) * Sigma_i_col[:, None]

                tmp1 = beta * np.dot(
                    Sigma_i_col.reshape(1, -1), PsiTPsi[active_indices])[0]
                S = S + kappa_i * np.square(tmp1)
                Q = Q + (kappa_i * mu_i) * tmp1

                # no change in active_indices or ind_global_to_local
                bcs_path.append(ind_L_max + 0.1)

            elif what_changed == -1:
                gamma[ind_L_max] = 0

                # index of regressor in Sigma
                local_ind = ind_global_to_local[ind_L_max]-1

                Sigma_ii_inv = 1. / Sigma[local_ind, local_ind]
                Sigma_i_col = Sigma[:, local_ind]

                Sigma = Sigma - Sigma_ii_inv * (Sigma_i_col * Sigma_i_col.T)

                Sigma = np.delete(
                    np.delete(Sigma, local_ind, axis=0), local_ind, axis=1)

                mu = mu - (mu[local_ind] * Sigma_ii_inv) * Sigma_i_col[:, None]
                mu = np.delete(mu, local_ind, axis=0)

                tmp1 = beta * np.dot(Sigma_i_col, PsiTPsi[active_indices])
                S = S + Sigma_ii_inv * np.square(tmp1)
                Q = Q + (mu_i * Sigma_ii_inv) * tmp1

                num_active -= 1
                ind_global_to_local[ind_L_max] = 0.0
                v = ind_global_to_local[ind_global_to_local > local_ind] - 1
                ind_global_to_local[ind_global_to_local > local_ind] = v
                del active_indices[local_ind]
                deleted_indices.append(ind_L_max)
                # and therefore ineligible
                bcs_path.append(-ind_L_max)

            # same for all three cases
            tmp3 = 1 - np.multiply(gamma, S)
            s = np.divide(S, tmp3)
            q = np.divide(Q, tmp3)

            # Update lambda
            Lambda = 2*(num_active - 1) / np.sum(gamma)

        # Prepare the result object
        self.coef_ = np.zeros(P)
        self.coef_[active_indices] = np.squeeze(mu)
        self.sigma_ = Sigma
        self.active_ = active_indices
        self.gamma = gamma
        self.Lambda = Lambda
        self.beta = beta
        self.bcs_path = bcs_path

        # set intercept_
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_std
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_.T)
        else:
            self.intercept_ = 0.

        return self

    def predict(self, X, return_std=False):
        '''
        Computes predictive distribution for test set.
        Predictive distribution for each data point is one dimensional
        Gaussian and therefore is characterised by mean and variance based on
        Ref.[1] Section 3.3.2.

        Parameters
        -----------
        X: {array-like, sparse} (n_samples_test, n_features)
           Test data, matrix of explanatory variables

        Returns
        -------
        : list of length two [y_hat, var_hat]

             y_hat: numpy array of size (n_samples_test,)
                    Estimated values of targets on test set (i.e. mean of
                    predictive distribution)

                var_hat: numpy array of size (n_samples_test,)
                    Variance of predictive distribution

        References
        ----------
        [1] Bishop, C. M. (2006). Pattern recognition and machine learning.
        springer.
        '''
        y_hat = np.dot(X, self.coef_) + self.intercept_

        if return_std:
            # Handle the zero variance case
            if self.var_y:
                return y_hat, np.zeros_like(y_hat)

            var_hat = 1./self.beta
            var_hat += np.sum(X.dot(self.sigma_) * X, axis=1)
            std_hat = np.sqrt(var_hat)
            return y_hat, std_hat
        else:
            return y_hat

# ------------------------------------------------------------------------


def corr(x, y):
    return abs(x.dot(y))/np.sqrt((x**2).sum())


class OrthogonalMatchingPursuit(LinearModel, RegressorMixin):
    '''
    Regression with Orthogonal Matching Pursuit [1].

    Parameters
    ----------
    fit_intercept : boolean, optional (DEFAULT = True)
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    copy_X : boolean, optional (DEFAULT = True)
        If True, X will be copied; else, it may be overwritten.

    verbose : boolean, optional (DEFAULT = FALSE)
        Verbose mode when fitting the model

    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)

    active_ : array, dtype = bool, shape = (n_features)
       True for non-zero coefficients, False otherwise

    References
    ----------
    [1] Pati, Y., Rezaiifar, R., Krishnaprasad, P. (1993). Orthogonal matching
        pursuit: recursive function approximation with application to wavelet
        decomposition. Proceedings of 27th Asilomar Conference on Signals,
        Systems and Computers, 40-44.
    '''

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 verbose=False):
        self.fit_intercept   = fit_intercept
        self.normalize       = normalize
        self.copy_X          = copy_X
        self.verbose         = verbose

    def _preprocess_data(self, X, y):
        """Center and scale data.
        Centers data to have mean zero along axis 0. If fit_intercept=False or
        if the X is a sparse matrix, no centering is done, but normalization
        can still be applied. The function returns the statistics necessary to
        reconstruct the input data, which are X_offset, y_offset, X_scale, such
        that the output
            X = (X - X_offset) / X_scale
        X_scale is the L2 norm of X - X_offset.
        """

        if self.copy_X:
            X = X.copy(order='K')

        y = np.asarray(y, dtype=X.dtype)

        if self.fit_intercept:
            X_offset = np.average(X, axis=0)
            X -= X_offset
            if self.normalize:
                X_scale = np.ones(X.shape[1], dtype=X.dtype)
                std = np.sqrt(np.sum(X**2, axis=0)/(len(X)-1))
                X_scale[std != 0] = std[std != 0]
                X /= X_scale
            else:
                X_scale = np.ones(X.shape[1], dtype=X.dtype)
            y_offset = np.mean(y)
            y = y - y_offset
        else:
            X_offset = np.zeros(X.shape[1], dtype=X.dtype)
            X_scale = np.ones(X.shape[1], dtype=X.dtype)
            if y.ndim == 1:
                y_offset = X.dtype.type(0)
            else:
                y_offset = np.zeros(y.shape[1], dtype=X.dtype)

        return X, y, X_offset, y_offset, X_scale

    def fit(self, X, y):
        '''
        Fits Regression with Orthogonal Matching Pursuit Algorithm.

        Parameters
        -----------
        X: {array-like, sparse matrix} of size (n_samples, n_features)
           Training data, matrix of explanatory variables

        y: array-like of size [n_samples, n_features]
           Target values

        Returns
        -------
        self : object
            Returns self.
        '''
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        n_samples, n_features = X.shape

        X, y, X_mean, y_mean, X_std = self._preprocess_data(X, y)
        self._x_mean_ = X_mean
        self._y_mean = y_mean
        self._x_std = X_std

        # Normalize columns of Psi, so that each column has norm = 1
        norm_X = np.linalg.norm(X, axis=0)
        X_norm = X/norm_X

        # Initialize residual vector to full model response and normalize
        R = y
        norm_y = np.sqrt(np.dot(y, y))
        r = y/norm_y

        # Check for constant regressors
        const_indices = np.where(~np.diff(X, axis=0).any(axis=0))[0]
        bool_const = not const_indices

        # Start regression using OPM algorithm
        precision = 0        # Set precision criterion to precision of program
        early_stop = True
        cond_early = True    # Initialize condition for early stop
        ind = []
        iindx = []           # index of selected columns
        indtot = np.arange(n_features)  # Full index set for remaining columns
        kmax = min(n_samples, n_features)  # Maximum number of iterations
        LOO = np.PINF * np.ones(kmax)  # Store LOO error at each iteration
        LOOmin = np.PINF               # Initialize minimum value of LOO
        coeff = np.zeros((n_features, kmax))
        count = 0
        k = 0.1                # Percentage of iteration history for early stop

        # Begin iteration over regressors set (Matrix X)
        while (np.linalg.norm(R) > precision) and (count <= kmax-1) and \
              ((cond_early or early_stop) ^ ~cond_early):

            # Update index set of columns yet to select
            if count != 0:
                indtot = np.delete(indtot, iindx)

            # Find column of X that is most correlated with residual
            h = abs(np.dot(r, X_norm))
            iindx = np.argmax(h[indtot])
            indx = indtot[iindx]

            # initialize with the constant regressor, if it exists in the basis
            if (count == 0) and bool_const:
                # overwrite values for iindx and indx
                iindx = const_indices[0]
                indx = indtot[iindx]

            # Invert the information matrix at the first iteration, later only
            # update its value on the basis of the previously inverted one,
            if count == 0:
                M = 1 / np.dot(X[:, indx], X[:, indx])
            else:
                x = np.dot(X[:, ind].T, X[:, indx])
                r = np.dot(X[:, indx], X[:, indx])
                M = self.blockwise_inverse(M, x, x.T, r)

            # Add newly found index to the selected indexes set
            ind.append(indx)

            # Select regressors subset (Projection subspace)
            Xpro = X[:, ind]

            # Obtain coefficient by performing OLS
            TT = np.dot(y, Xpro)
            beta = np.dot(M, TT)
            coeff[ind, count] = beta

            # Compute LOO error
            LOO[count] = self.loo_error(Xpro, M, y, beta)

            # Compute new residual due to new projection
            R = y - np.dot(Xpro, beta)

            # Normalize residual
            norm_R = np.sqrt(np.dot(R, R))
            r = R / norm_R

            # Update counters and early-stop criterions
            countinf = max(0, int(count-k*kmax))
            LOOmin = min(LOOmin, LOO[count])

            if count == 0:
                cond_early = (LOO[0] <= LOOmin)
            else:
                cond_early = (min(LOO[countinf:count+1]) <= LOOmin)

            if self.verbose:
                print(f'Iteration: {count+1}, mod. LOOCV error : '
                      f'{LOO[count]:.2e}')

            # Update counter
            count += 1

        # Select projection with smallest cross-validation error
        countmin = np.argmin(LOO[:-1])
        self.coef_ = coeff[:, countmin]
        self.active = coeff[:, countmin] != 0.0

        # set intercept_
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_std
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_.T)
        else:
            self.intercept_ = 0.

        return self

    def predict(self, X):
        '''
        Computes predictive distribution for test set.

        Parameters
        -----------
        X: {array-like, sparse} (n_samples_test, n_features)
           Test data, matrix of explanatory variables

        Returns
        -------
        y_hat: numpy array of size (n_samples_test,)
               Estimated values of targets on test set (i.e. mean of
               predictive distribution)
        '''

        y_hat = np.dot(X, self.coef_) + self.intercept_

        return y_hat

    def loo_error(self, psi, inv_inf_matrix, y, coeffs):
        """
        Calculates the corrected LOO error for regression on regressor
        matrix `psi` that generated the coefficients based on [1] and [2].

        [1] Blatman, G., 2009. Adaptive sparse polynomial chaos expansions for
            uncertainty propagation and sensitivity analysis (Doctoral
            dissertation, Clermont-Ferrand 2).

        [2] Blatman, G. and Sudret, B., 2011. Adaptive sparse polynomial chaos
            expansion based on least angle regression. Journal of computational
            Physics, 230(6), pp.2345-2367.

        Parameters
        ----------
        psi : array of shape (n_samples, n_feature)
            Orthogonal bases evaluated at the samples.
        inv_inf_matrix : array
            Inverse of the information matrix.
        y : array of shape (n_samples, )
            Targets.
        coeffs : array
            Computed regresssor cofficients.

        Returns
        -------
        loo_error : float
            Modified LOOCV error.

        """

        # NrEvaluation (Size of experimental design)
        N, P = psi.shape

        # h factor (the full matrix is not calculated explicitly,
        # only the trace is, to save memory)
        PsiM = np.dot(psi, inv_inf_matrix)

        h = np.sum(np.multiply(PsiM, psi), axis=1, dtype=np.float64)

        # ------ Calculate Error Loocv for each measurement point ----
        # Residuals
        residual = np.dot(psi, coeffs) - y

        # Variance
        varY = np.var(y)

        if varY == 0:
            norm_emp_error = 0
            loo_error = 0
        else:
            norm_emp_error = np.mean(residual**2)/varY

            loo_error = np.mean(np.square(residual / (1-h))) / varY

            # if there are NaNs, just return an infinite LOO error (this
            # happens, e.g., when a strongly underdetermined problem is solved)
            if np.isnan(loo_error):
                loo_error = np.inf

        # Corrected Error for over-determined system
        tr_M = np.trace(np.atleast_2d(inv_inf_matrix))
        if tr_M < 0 or abs(tr_M) > 1e6:
            tr_M = np.trace(np.linalg.pinv(np.dot(psi.T, psi)))

        # Over-determined system of Equation
        if N > P:
            T_factor = N/(N-P) * (1 + tr_M)

        # Under-determined system of Equation
        else:
            T_factor = np.inf

        loo_error *= T_factor

        return loo_error

    def blockwise_inverse(self, Ainv, B, C, D):
        """
        non-singular square matrix M defined as M = [[A B]; [C D]] .
        B, C and D can have any dimension, provided their combination defines
        a square matrix M.

        Parameters
        ----------
        Ainv : float or array
            inverse of the square-submatrix A.
        B : float or array
            Information matrix with all new regressor.
        C : float or array
            Transpose of B.
        D : float or array
            Information matrix with all selected regressors.

        Returns
        -------
        M : array
            Inverse of the information matrix.

        """
        if np.isscalar(D):
            # Inverse of D
            Dinv = 1/D
            # Schur complement
            SCinv = 1/(D - np.dot(C, np.dot(Ainv, B[:, None])))[0]
        else:
            # Inverse of D
            Dinv = np.linalg.solve(D, np.eye(D.shape))
            # Schur complement
            SCinv = np.linalg.solve((D - C*Ainv*B), np.eye(D.shape))

        T1 = np.dot(Ainv, np.dot(B[:, None], SCinv))
        T2 = np.dot(C, Ainv)

        # Assemble the inverse matrix
        M = np.vstack((
            np.hstack((Ainv+T1*T2, -T1)),
            np.hstack((-(SCinv)*T2, SCinv))
            ))
        return M
