#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import distance


class Exploration:
    """
    Created based on the Surrogate Modeling Toolbox (SUMO) [1]. Modeled after code in BayesValidRox [2]

    [1] Gorissen, D., Couckuyt, I., Demeester, P., Dhaene, T. and Crombecq, K.,
        2010. A surrogate modeling and adaptive sampling toolbox for computer
        based design. Journal of machine learning research.-Cambridge, Mass.,
        11, pp.2051-2055. sumo@sumo.intec.ugent.be - http://sumo.intec.ugent.be
    [2] Mohammadi, F. https://pypi.org/project/bayesvalidrox/

    Class generates samples from the prior distribution, using the ExpDesign class attributes and functions. There are
    2 possible ways: Voronoi sampling and normal MC sampling.
        Voronoi: t.b.d.

        mc_samples: normal MC sampling using random samples, sobol sampling or latin_hypercube sampling. It can also
        receive previously-sampled candidates, in which case no sampling is done, and the scores for each sample will
        be estimated

    For the selected candidates, a score will be estimated in regard to their distance to the existing training
    points. This guarantees that we explore the entire domain. The scores are normalized, and threfore range from 0-1.
    The highest values correspond to the highest/best scores.

    Attributes
    ----------
    exp_design : obj
        ExpDesign object, needed to sample from the prior distribution
    n_candidate : int
        Number of candidate samples.
    mc_criterion : str
        Selection criterion. The default is `'mc-intersite-proj-th'`. Another
        option is `'mc-intersite-proj'`.
    w : int
        Number of random points in the domain for each sample of the
        training set.
    """
    def __init__(self, n_candidate,
                 old_tp,
                 exp_design=None,
                 mc_criterion='mc-intersite-proj-th',
                 w=100):

        self.n_candidate = n_candidate
        self.mc_criterion = mc_criterion
        self.w = w

        self.old_ED_X = old_tp
        self.ExpDesign = exp_design

    def get_exploration_samples(self, prior_candidates=None):
        """
        This function generates candidates to be selected as new design and
        their associated exploration scores.

        Returns
        -------
        all_candidates : array of shape (n_candidate, n_params)
            A list of samples.
        exploration_scores: arrays of shape (n_candidate)
            Exploration scores.
        """
        explore_method = self.ExpDesign.explore_method

        # print("\n")
        # print(f' The {explore_method}-Method is selected as the exploration method.')
        # print("\n")

        if explore_method == 'Voronoi':
            # Generate samples using the Voronoi method
            all_candidates, exploration_scores = self.get_vornoi_samples()
        else:
            # Generate samples using the MC method
            all_candidates, exploration_scores = self.get_mc_samples(all_candidates=prior_candidates)

        return all_candidates, exploration_scores

    # -------------------------------------------------------------------------
    def get_vornoi_samples(self):
        """
        This function generates samples based on voronoi cells and their
        corresponding scores

        Returns
        -------
        new_samples : array of shape (n_candidate, n_params)
            A list of samples.
        exploration_scores: arrays of shape (n_candidate)
            Exploration scores.
        """

        mc_criterion = self.mc_criterion
        n_candidate = self.n_candidate

        # Get the Old ExpDesign #samples
        # old_ED_X = self.MetaModel.ExpDesign.X
        old_ED_X = self.old_ED_X
        ndim = old_ED_X.shape[1]

        # calculate error #averageErrors
        error_voronoi, all_candidates = self.approximate_voronoi(
            self.w, old_ED_X
        )

        # Pick the best candidate point in the voronoi cell
        # for each best sample
        selected_samples = np.empty((0, ndim))
        bad_samples = []

        for index in range(len(error_voronoi)):

            # get candidate new samples from voronoi tesselation
            candidates = self.closest_points[index]

            # get total number of candidates
            n_new_samples = candidates.shape[0]

            # still no candidate samples around this one, skip it!
            if n_new_samples == 0:
                print('The following sample has been skipped because there '
                      'were no candidate samples around it...')
                print(old_ED_X[index])
                bad_samples.append(index)
                continue

            # find candidate that is farthest away from any existing sample
            max_min_distance = 0
            best_candidate = 0
            min_intersite_dist = np.zeros((n_new_samples))
            min_projected_dist = np.zeros((n_new_samples))

            for j in range(n_new_samples):

                new_samples = np.vstack((old_ED_X, selected_samples))

                # find min distorted distance from all other samples
                euclidean_dist = self._build_dist_matrix_point(
                    new_samples, candidates[j], do_sqrt=True)
                min_euclidean_dist = np.min(euclidean_dist)
                min_intersite_dist[j] = min_euclidean_dist

                # Check if this is the maximum minimum distance from all other
                # samples
                if min_euclidean_dist >= max_min_distance:
                    max_min_distance = min_euclidean_dist
                    best_candidate = j

                # Projected distance
                projected_dist = distance.cdist(
                    new_samples, [candidates[j]], 'chebyshev')
                min_projected_dist[j] = np.min(projected_dist)

            if mc_criterion == 'mc-intersite-proj':
                weight_euclidean_dist = 0.5 * ((n_new_samples + 1) ** (1 / ndim) - 1)
                weight_projected_dist = 0.5 * (n_new_samples + 1)
                total_dist_scores = weight_euclidean_dist * min_intersite_dist
                total_dist_scores += weight_projected_dist * min_projected_dist

            elif mc_criterion == 'mc-intersite-proj-th':
                alpha = 0.5  # chosen (tradeoff)
                d_min = 2 * alpha / n_new_samples
                if any(min_projected_dist < d_min):
                    candidates = np.delete(
                        candidates, [min_projected_dist < d_min], axis=0
                    )
                    total_dist_scores = np.delete(
                        min_intersite_dist, [min_projected_dist < d_min],
                        axis=0
                    )
                else:
                    total_dist_scores = min_intersite_dist
            else:
                raise NameError(
                    'The MC-Criterion you requested is not available.'
                )

            # Add the best candidate to the list of new samples
            best_candidate = np.argsort(total_dist_scores)[::-1][:n_candidate]
            selected_samples = np.vstack(
                (selected_samples, candidates[best_candidate])
            )

        self.new_samples = selected_samples
        self.exploration_scores = np.delete(error_voronoi, bad_samples, axis=0)

        return self.new_samples, self.exploration_scores

    # -------------------------------------------------------------------------
    def get_mc_samples(self, all_candidates=None):
        """
        This function generates random samples based on Global Monte Carlo
        methods and their corresponding scores, based on [1].

        [1] Crombecq, K., Laermans, E. and Dhaene, T., 2011. Efficient
            space-filling and non-collapsing sequential design strategies for
            simulation-based modeling. European Journal of Operational Research
            , 214(3), pp.683-696.
            DOI: https://doi.org/10.1016/j.ejor.2011.05.032

        Implemented methods to compute scores:
            1) mc-intersite-proj
            2) mc-intersite-proj-th

        Arguments
        ---------
        all_candidates : array, optional
            Samples to compute the scores for. The default is `None`. In this
            case, samples will be generated by defined model input marginals.

        Returns
        -------
        new_samples : array of shape (n_candidate, n_params)
            A list of samples.
        exploration_scores: arrays of shape (n_candidate)
            Exploration scores.
        """
        # MetaModel = self.MetaModel
        # explore_method = MetaModel.ExpDesign.explore_method
        ExpDesign = self.ExpDesign
        explore_method = ExpDesign.explore_method
        mc_criterion = self.mc_criterion

        # Get number of candidates to evaluate
        if all_candidates is None:
            n_candidate = self.n_candidate
        else:
            n_candidate = all_candidates.shape[0]

        # Get the Old ExpDesign #samples
        old_ED_X = self.old_ED_X
        ndim = old_ED_X.shape[1]     # Number of parameters in each parameter set

        # Sample from prior-----------------------------------------
        if all_candidates is None:
            # Generate MC Samples
            all_candidates = ExpDesign.generate_samples(
                self.n_candidate, explore_method)
        self.all_candidates = all_candidates

        # initialization
        min_intersite_dist = np.zeros((n_candidate))
        min_projected_dist = np.zeros((n_candidate))

        for i, candidate in enumerate(all_candidates):

            # find candidate that is farthest away from any existing sample
            maxMinDistance = 0

            # find min distorted distance from all other samples
            euclidean_dist = self._build_dist_matrix_point(
                old_ED_X, candidate, do_sqrt=True)
            min_euclidean_dist = np.min(euclidean_dist)
            min_intersite_dist[i] = min_euclidean_dist

            # Check if this is the maximum minimum distance from all other
            # samples
            if min_euclidean_dist >= maxMinDistance:
                maxMinDistance = min_euclidean_dist

            # Projected distance
            projected_dist = self._build_dist_matrix_point(
                old_ED_X, candidate, 'chebyshev')
            min_projected_dist[i] = np.min(projected_dist)

        if mc_criterion == 'mc-intersite-proj':
            weight_euclidean_dist = ((n_candidate + 1) ** (1 / ndim) - 1) * 0.5
            weight_projected_dist = (n_candidate + 1) * 0.5
            total_dist_scores = weight_euclidean_dist * min_intersite_dist
            total_dist_scores += weight_projected_dist * min_projected_dist

        elif mc_criterion == 'mc-intersite-proj-th':
            alpha = 0.5  # chosen (tradeoff)
            d_min = 2 * alpha / n_candidate
            if any(min_projected_dist < d_min):
                all_candidates = np.delete(
                    all_candidates, [min_projected_dist < d_min], axis=0
                )
                total_dist_scores = np.delete(
                    min_intersite_dist, [min_projected_dist < d_min], axis=0
                )
            else:
                total_dist_scores = min_intersite_dist
        else:
            raise NameError('The MC-Criterion you requested is not available.')

        self.new_samples = all_candidates
        self.exploration_scores = total_dist_scores
        self.exploration_scores /= np.nansum(total_dist_scores)

        return self.new_samples, self.exploration_scores

    # -------------------------------------------------------------------------
    def approximate_voronoi(self, w, samples):
        """
        An approximate (monte carlo) version of Matlab's voronoi command.

        Arguments
        ---------
        samples : array
            Old experimental design to be used as center points for voronoi
            cells.

        Returns
        -------
        areas : array
            An approximation of the voronoi cells' areas.
        all_candidates: list of arrays
            A list of samples in each voronoi cell.
        """
        # MetaModel = self.MetaModel
        ExpDesign = self.ExpDesign

        n_samples = samples.shape[0]
        ndim = samples.shape[1]

        # Compute the number of random points
        n_points = w * samples.shape[0]
        # Generate w random points in the domain for each sample
        points = ExpDesign.generate_samples(n_points, 'random')
        self.all_candidates = points

        # Calculate the nearest sample to each point
        self.areas = np.zeros((n_samples))
        self.closest_points = [np.empty((0, ndim)) for i in range(n_samples)]

        # Compute the minimum distance from all the samples of old_ED_X for
        # each test point
        for idx in range(n_points):
            # calculate the minimum distance
            distances = self._build_dist_matrix_point(
                samples, points[idx], do_sqrt=True
            )
            closest_sample = np.argmin(distances)

            # Add to the voronoi list of the closest sample
            self.areas[closest_sample] = self.areas[closest_sample] + 1
            prev_closest_points = self.closest_points[closest_sample]
            self.closest_points[closest_sample] = np.vstack(
                (prev_closest_points, points[idx])
            )

        # Divide by the amount of points to get the estimated volume of each
        # voronoi cell
        self.areas /= n_points

        self.perc = np.max(self.areas * 100)

        self.errors = self.areas

        return self.areas, self.all_candidates

    # -------------------------------------------------------------------------
    def _build_dist_matrix_point(self, samples, point, method='euclidean',
                                 do_sqrt=False):
        """
        Calculates the intersite distance of all points in samples from point.

        Parameters
        ----------
        samples : array of shape (n_samples, n_params)
            The old experimental design.
        point : array
            A candidate point.
        method : str
            Distance method.
        do_sqrt : bool, optional
            Whether to return distances or squared distances. The default is
            `False`.

        Returns
        -------
        distances : array
            Distances.

        """
        distances = distance.cdist(samples, np.array([point]), method)

        # do square root?
        if do_sqrt:
            return distances
        else:
            return distances ** 2

