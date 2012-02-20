import numpy as np
from sklearn.linear_model.base import LinearModel, LinearRegression
from sklearn.utils import safe_asarray as safe_asarray
from scipy.optimize import nnls
import scipy.sparse as sp
from scipy import linalg


VERBOSE = 1

import time
from sklearn.cross_validation import check_cv
from sklearn.base import clone


def svd_ridge_regression(X, y, alphas):
    # import pdb
    # pdb.set_trace()
    U, s, V = linalg.svd(X, full_matrices=False)
    col_s = s[:, np.newaxis]

    UTy = np.dot(U.T, y)

    sUTy = UTy / (col_s + alphas / col_s)

    return np.dot(V.T, sUTy)


class SVDRidge(LinearModel):
    """Implements Ridge regression using thin svd with different alphas
    for each y"""

    def __init__(self, alphas=None, alpha=1.,
                 fit_intercept=True, normalize=False, copy_X=True):
        self.alphas = alphas
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X

    # def fit(self, X, y):
    #     X = safe_asarray(X)
    #     y = np.asarray(y)

    #     X, y, X_mean, y_mean, X_std = self._center_data(X, y,
    #             self.fit_intercept, self.normalize, self.copy_X)

    #     U, s, V = linalg.svd(X, full_matrices=False)

    #     if self.alphas is None:
    #         alphas = self.alpha * np.ones([1, y.shape[1]])
    #     else:
    #         alphas = self.alphas.reshape(1, -1)
    #         assert(alphas.shape[1] == y.shape[1])     # otherwise won't work

    #     col_s = s[:, np.newaxis]

    #     UTy = np.dot(U.T, y)

    #     sUTy = UTy / (col_s + alphas / col_s)

    #     self.coef_ = np.dot(V.T, sUTy)
    #     self._set_intercept(X_mean, y_mean, X_std)
    #     return self

    def fit(self, X, y):
        X = safe_asarray(X, dtype=np.float)
        y = np.asarray(y, dtype=np.float)

        X, y, X_mean, y_mean, X_std = \
           self._center_data(X, y, self.fit_intercept,
                   self.normalize, self.copy_X)

        if self.alphas is None:
            alphas = self.alpha * np.ones([1, y.shape[1]])
        else:
            alphas = self.alphas.reshape(1, -1)
            assert(alphas.shape[1] == y.shape[1])     # otherwise won't work

        self.coef_ = svd_ridge_regression(X, y, alphas).T
        self._set_intercept(X_mean, y_mean, X_std)
        return self


def crazy_svd_ridge_regression(X, y, alphas):
    """does svd ridge with several alphas per y"""
    if VERBOSE > 0:
        print "Starting multiple alpha ridge regression with alphas %s"\
            % str(alphas)

    t0 = time.time()
    U, s, V = linalg.svd(X, full_matrices=False)

    t = time.time()
    if VERBOSE:
        print "SVD done in %f seconds" % (t - t0)

    crazy_col_s = s[np.newaxis, :, np.newaxis]

    t0 = time.time()
    UTy = np.dot(U.T, y)[np.newaxis, :]
    t = time.time()
    if VERBOSE:
        print "U.T * y multiplication took %f seconds" % (t - t0)

    crazy_alphas = alphas[:, np.newaxis, :]

    t0 = time.time()
    sUTy = UTy / (crazy_col_s + crazy_alphas / crazy_col_s)
    t = time.time()
    if VERBOSE:
        print "Singular value and penalty adding took %f seconds" % (t - t0)

    t0 = time.time()
    VT = V.T  # .copy()
    raw_result = np.empty([alphas.shape[0], VT.shape[0], sUTy.shape[2]])
    # raw_result = np.array([np.dot(VT, sUTy[i].T.copy().T)
    #                        for i in range(alphas.shape[0])])
    for i in range(alphas.shape[0]):
        raw_result[i] = np.dot(VT, sUTy[i])

    t1 = time.time()
    if VERBOSE:
        print "Separate matrix multiplications took %f seconds" % (t1 - t0)
    # raw_result = np.dot(V.T, sUTy)
    # t = time.time()
    # if VERBOSE:
    #     print "the stupid V.T * multiplication took %f seconds" % (t - t1)

    t0 = time.time()
    result = raw_result.transpose((0, 2, 1))
    t = time.time()

    if VERBOSE:
        print "transposing took %f seconds" % (t - t0)
    return result


def mass_r2_score(y_true, y_pred):
    """Does r2 scoring along axis 1==-2, leaving the other axes as they are"""

    # if y_true_.ndim == 2:
    #     y_true = y_true_[np.newaxis, :]
    # else:
    #     y_true = y_true_

    numerator = ((y_pred - y_true[np.newaxis, :]) ** 2).sum(1)

    denominator = ((y_true - y_true.mean(0)) ** 2).sum(0)

    zero_denominators = denominator == 0.0

    score = -100.0 * np.ones([y_pred.shape[0], y_pred.shape[2]])

    score[:, np.logical_not(zero_denominators)] =\
        1. - numerator / denominator[np.newaxis, :]

    zero_denominators_and_numerators = numerator[:, zero_denominators] == 0

    score[:, zero_denominators][:, zero_denominators_and_numerators] = 1.

    return score


class cSVDRidge(LinearModel):
    """Implements Ridge regression using thin svd with different alphas
    for each y"""

    def __init__(self, alpha=1.,
                 fit_intercept=True, normalize=False, copy_X=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X

    # def _get_params(self, *args, **kwargs):
    #     """Gives back the important parameters to build this estimator"""
    #     raise Exception("Not yet implemented!")

    def fit(self, X, y):
        X = safe_asarray(X, dtype=np.float)
        y = np.asarray(y, dtype=np.float)

        X, y, X_mean, y_mean, X_std = \
           self._center_data(X, y, self.fit_intercept,
                   self.normalize, self.copy_X)

        if not isinstance(self.alpha, np.ndarray):
            alphas = np.array([[self.alpha]])
        elif self.alpha.ndim == 1:
            # one dimensional array. If length corresponds to number of ys
            # then treat alphas as corresponding to each ys.
            # Otherwise use every alpha for every y
            if len(self.alpha) == y.shape[1]:
                alphas = self.alpha.reshape(1, -1)
            else:
                alphas = self.alpha.reshape(-1, 1)
        elif self.alpha.ndim == 2:
            # if the array is two dimensional,
            # it must either be n x ys.shape[1] or n x 1

            if (self.alpha.shape[1] != 1) and \
                (self.alpha.shape[1] != y.shape[1]):
                raise Exception("got 2d array of alphas of wrong size. Don't\
know what to do")
            else:
                alphas = self.alpha

        self.coef_ = crazy_svd_ridge_regression(X, y, alphas)
        # TODO : SET INTERCEPT is missing!
        return self

    def predict_(self, X):
        """Applies the regression coefficients to the given data.
        Will return one prediction per different alpha and thus
        does not correspond to the standard response of this function."""
        # return np.dot(X, self.coef_)
        predictions = np.empty([self.coef_.shape[0], X.shape[0],
                                self.coef_.shape[1]])
        # return np.array([np.dot(X, self.coef_.transpose((0, 2, 1))[i])
        #                  for i in range(self.coef_.shape[0])])
        for i in range(self.coef_.shape[0]):
            predictions[i] = np.dot(X, self.coef_.transpose((0, 2, 1))[i])
        return predictions

    def predict(self, X):
        p = self.predict_(X)
        return p.squeeze()

    def score_(self, X, y_true):
        """r2 scores the whole batch"""

        y_pred = self.predict_(X)

        return mass_r2_score(y_true, y_pred)

    def best_score(self, X, y_true):
        score = self.score_(X, y_true)
        return (score.max(0), score.argmax(0))


def make_grids(alpha_mins, alpha_maxs, num_steps,
               end_included=True, start_included=True):
    """Makes all the requested grids - independent for each y vector"""

    # first resolve some broadcasting issues:
    # this way, both arrays have the same size

    alpha_maxs = alpha_mins + (alpha_maxs - alpha_mins)
    alpha_mins = alpha_maxs - (alpha_maxs - alpha_mins)

    log_alpha_mins = np.log10(alpha_mins)
    log_alpha_maxs = np.log10(alpha_maxs)

    steps = (log_alpha_maxs - log_alpha_mins) /\
        np.float64(num_steps + 1 - end_included - start_included)

    grids = np.exp(np.log(10) * np.vstack([log_alpha_mins +
                                           (i + (1 - start_included)) * steps
                                           for i in range(num_steps)]))
    # if not start_included:
    #     grids = grids[1:]   # remove first line

    return grids


def find_new_intervals(current_grid, best_score_indices):

    lower_bounds = np.zeros(best_score_indices.shape[0:1])
    upper_bounds = np.zeros(best_score_indices.shape[0:1])

    lower_uncritical = best_score_indices > 0
    lower_critical = np.logical_not(lower_uncritical)
    upper_uncritical = best_score_indices < len(current_grid) - 1
    upper_critical = np.logical_not(upper_uncritical)

    lower_bounds[lower_uncritical] = current_grid[
        best_score_indices[lower_uncritical] - 1,
        lower_uncritical.nonzero()]

    lower_bounds[lower_critical] =\
        current_grid[0, lower_critical] ** 2 /\
        current_grid[-1, lower_critical]

    upper_bounds[upper_uncritical] = current_grid[
        best_score_indices[upper_uncritical] + 1,
        upper_uncritical.nonzero()]

    upper_bounds[upper_critical] =\
        current_grid[-1, upper_critical] ** 2 /\
        current_grid[0, upper_critical]

    upper_bounds[lower_critical] = np.sqrt(current_grid[0, lower_critical] *
                                           current_grid[-1, lower_critical])

    lower_bounds[upper_critical] = np.sqrt(current_grid[0, upper_critical] *
                                           current_grid[-1, upper_critical])


    if VERBOSE > 0:
        if lower_critical.any():
            print "lower critical points: %s" % str(lower_critical.nonzero())
        if upper_critical.any():
            print "upper critical points: %s" % str(upper_critical.nonzero())

    return (lower_bounds, upper_bounds)


def _cross_val_helper(estimator_, X, y, train, test, keep_estimators=False):

    # make sure estimator is clean
    estimator = clone(estimator_)

    estimator.fit(X[train], y[train])
    scores = estimator.score_(X[test], y[test])

    if keep_estimators:
        return (scores, estimator)
    else:
        return scores


def do_cross_val(estimator, X, y, cv, keep_estimators=False):
    """Lean version of cross_val_score that gives back all I need"""
    cv = check_cv(cv, X, y)

    # fitted_estimators = [clone(estimator).fit(X[train], y[train])
    #                     for train, test in cv]

    # tests = [test for _, test in cv]
    # scores = np.array([fitted_estimator.score_(X[test], y[test])
    #                   for fitted_estimator, test
    #                   in zip(fitted_estimators, tests)])

    results = [_cross_val_helper(clone(estimator),
                                X, y, train, test, keep_estimators)
                                for train, test in cv]

    if keep_estimators:
        transposed_results = zip(*results)
        return (np.array(transposed_results[0]), transposed_results[1])
    else:
        return np.array(results)

    # import pdb
    # pdb.set_trace()

    # if keep_estimators:
    #     return (scores, fitted_estimators)
    # else:
    #     return scores, results


class SVDRidgeCV(LinearModel):
    """Does CV with svd ridge"""

    def __init__(self,
                 alpha_min=1e-3,
                 alpha_max=1e7,
                 num_grid_points=5,
                 num_refinements=5,
                 cv=5):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.num_grid_points = num_grid_points
        self.num_refinements = num_refinements
        self.cv = cv

    def _get_scores(self, X, y, alphas, cv):
        clf = cSVDRidge(alphas)
        scores = do_cross_val(clf, X, y, cv, keep_estimators=False)
        return scores

    def _best_params(self, X, y, alphas, cv):
        scores = self._get_scores(X, y, cv, keep_estimators=False)

        mean_scores = scores.mean(0)

        return mean_scores.argmax(0)

    def fit(self, X, y, keep_best_estimators=False, keep_scores=True):
        """Fits ridge estimators using a grid refinement scheme for the
        penalties"""
        t_fit0 = time.time()
        if VERBOSE > 0:
            print "Fitting SVDRidgeCV to data of shape X=%s, y=%s" %\
                (str(X.shape), str(y.shape))

        if isinstance(self.alpha_min, np.ndarray):
            if self.alpha_min.shape != (1, y.shape[1]):
                raise Exception("alpha_min is of shape %s. I don't understand"\
                                % str(self.alpha_min.shape))
            alpha_mins = self.alpha_min
        else:
            # in this case self.alpha_min is one number
            alpha_mins = self.alpha_min * np.ones((1, y.shape[1]))

        if isinstance(self.alpha_max, np.ndarray):
            if self.alpha_max.shape != (1, y.shape[1]):
                raise Exception("alpha_max is of shape %s. I don't understand"\
                                % str(self.alpha_max.shape))
            alpha_maxs = self.alpha_max
        else:
            # same here
            alpha_maxs = self.alpha_max * np.ones((1, y.shape[1]))

        grids = make_grids(alpha_mins, alpha_maxs, self.num_grid_points)

        cv = check_cv(self.cv, X, y)

        # initialise structure for keeping all grid points and scores if needed
        if keep_scores:
            self.all_grids = np.empty([grids.shape[0] * self.num_refinements,
                                       grids.shape[1]])

            self.all_scores = np.empty((len(cv),) + self.all_grids.shape)

        # Do all refinements except the last one
        for i in range(self.num_refinements - 1):
            if VERBOSE > 0:
                print "Effectuating cross_val grid refinement %d" % i

            t0 = time.time()
            scores = self._get_scores(X, y, grids, cv)
            best_params = scores.mean(0).argmax(0)
            t1 = time.time()

            if VERBOSE > 0:
                print "This took %f seconds" % (t1 - t0)

            if keep_scores:
                self.all_grids[i * grids.shape[0]:\
                               (i + 1) * grids.shape[0], :] = grids
                self.all_scores[:, i * grids.shape[0]:\
                                (i + 1) * grids.shape[0], :] = scores

            lower_bounds, upper_bounds = find_new_intervals(grids, best_params)
            grids = make_grids(lower_bounds, upper_bounds,
                               self.num_grid_points, False, False)

        # Here goes the last one, potentially keeping the estimators
        clf = cSVDRidge(grids)
        if VERBOSE > 0:
            print "Effectuating cross_val grid refinement %d" %\
                (self.num_refinements - 1)
        t0 = time.time()
        res = do_cross_val(clf, X, y, self.cv, keep_best_estimators)
        t1 = time.time()
        if VERBOSE > 0:
            print "This took %f seconds" % (t1 - t0)

        if keep_best_estimators:
            scores, self.estimators = res
        else:
            scores = res
            self.estimators = None

        best_params_arg = scores.mean(0).argmax(0)
        self.best_params = grids[best_params_arg,
                            np.arange(y.shape[1])].reshape(1, -1)

        if keep_scores:
            self.all_grids[-grids.shape[0]:, :] = grids
            self.all_scores[:, -grids.shape[0]:, :] = scores

        clf = cSVDRidge(self.best_params)
        clf.fit(X, y)

        self.clf = clf

        t_fit1 = time.time()
        if VERBOSE > 0:
            print "The whole fitting process took %f seconds" %\
                (t_fit1 - t_fit0)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def score(self, X, y_true):
        if VERBOSE > 0:
            print "in SVDRidgeCV.score !!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        return self.clf.score_(X, y_true).squeeze()
