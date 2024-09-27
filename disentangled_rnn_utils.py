import os
import time
import datetime
import logging
import torch
import torch.nn as nn
from sklearn.metrics import mutual_info_score
from torch import vmap
from torch.func import jacrev
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset
from disentangled_rnn import RNN


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        # We trust the dict to init itself better than we can.
        dict.__init__(self, *args, **kwargs)
        # Because of that, we do duplicate work, but it's worth it.
        for k, v in self.items():
            self.__setitem__(k, v)

    def __getattr__(self, k):
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            # Maintain consistent syntactical behaviour.
            raise AttributeError(
                "'DotDict' object has no attribute '" + str(k) + "'"
            )

    def __setitem__(self, k, v):
        dict.__setitem__(self, k, DotDict.__convert(v))

    __setattr__ = __setitem__

    def __delattr__(self, k):
        try:
            dict.__delitem__(self, k)
        except KeyError:
            raise AttributeError(
                "'DotDict' object has no attribute '" + str(k) + "'"
            )

    @staticmethod
    def __convert(o):
        """
        Recursively convert `dict` objects in `dict`, `list`, `set`, and
        `tuple` objects to `attrdict` objects.
        """
        if isinstance(o, dict):
            o = DotDict(o)
        elif isinstance(o, list):
            o = list(DotDict.__convert(v) for v in o)
        elif isinstance(o, set):
            o = set(DotDict.__convert(v) for v in o)
        elif isinstance(o, tuple):
            o = tuple(DotDict.__convert(v) for v in o)
        return o

    @staticmethod
    def to_dict(data):
        """
        Recursively transforms a dotted dictionary into a dict
        """
        if isinstance(data, dict):
            data_new = {}
            for k, v in data.items():
                data_new[k] = DotDict.to_dict(v)
            return data_new
        elif isinstance(data, list):
            return [DotDict.to_dict(i) for i in data]
        elif isinstance(data, set):
            return [DotDict.to_dict(i) for i in data]
        elif isinstance(data, tuple):
            return [DotDict.to_dict(i) for i in data]
        else:
            return data


# mutual information stuff

def normalized_multiinformation(s):
    """
    s: np.ndarray, shape=(n_samples, n_sources)
    """
    assert s.ndim == 2
    marginal_entropies = [entropy(s[:, i]) for i in range(s.shape[1])]
    joint_entropy = entropy(s)
    nmi = (sum(marginal_entropies) - joint_entropy) / sum(marginal_entropies)
    nmi = max(0, nmi)
    nmi = min(1, nmi)
    return nmi


def discretize_binning(z, bins):
    ret = np.zeros_like(z, dtype=np.int32)
    for i in range(z.shape[1]):
        ret[:, i] = np.digitize(z[:, i], np.histogram(z[:, i], bins=bins)[1][:-1])
    return ret


def discrete_mutual_info(mus, ys):
    """Compute discrete mutual information."""
    num_codes = mus.shape[0]
    num_factors = ys.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = mutual_info_score(ys[j, :], mus[i, :])
    return m


def discrete_entropy(ys):
    """Compute discrete mutual information."""
    num_factors = ys.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = mutual_info_score(ys[j, :], ys[j, :])
    return h


def histogram_discretize(target, num_bins=20):
    """Discretization based on histograms."""
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(
            target[i, :], num_bins)[1][:-1])
    return discretized


def mig(m):
    sorted_m = np.sort(m, axis=0)[::-1]
    return np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :]))


def mir(m):
    score = np.max(m, axis=1) / np.sum(m, axis=1)
    min_mir = 1.0 / m.shape[1]
    return (np.mean(score) - min_mir) / (1 - min_mir)


def clean_mi(m, val=10):
    return m[m.sum(axis=1) > m.sum(axis=1).max() / val, :]


def compute_mir(factors, mus, num_bins=20, clean_val=20):
    average_firing = np.mean(np.abs(mus), axis=1)
    cell_to_keep = average_firing > average_firing.max() / 20
    mus = mus[cell_to_keep]

    discretized_mus = histogram_discretize(mus, num_bins=num_bins)
    discretized_factors = histogram_discretize(factors, num_bins=num_bins)
    m = discrete_mutual_info(discretized_mus, discretized_factors)  # num_cells x num_factors
    entropy = discrete_entropy(discretized_factors)
    m = m / entropy[None, ...]
    m_cleaned = clean_mi(m, val=clean_val)  # only consider cells that have sizable mutual info
    return mir(m_cleaned), m, cell_to_keep


# import ipdb
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import log_loss, mean_squared_error


def entropy(x):
    _, counts = np.unique(x, return_counts=True, axis=0)
    p_x = counts / x.shape[0]
    return -np.sum(p_x * np.log(p_x))


def conditional_entropy(x, y):
    """H(X|Y) = H(X, Y) - H(Y)"""
    H_xy = entropy(np.concatenate([x, y], axis=1))
    H_y = entropy(y)
    return H_xy - H_y


def mutual_information(x, y):
    """I(X;Y) = H(X) + H(Y) - H(X, Y)"""
    assert x.ndim == y.ndim == 2
    H_x = entropy(x)
    H_y = entropy(y)
    H_xy = entropy(np.concatenate([x, y], axis=1))
    return H_x + H_y - H_xy


def conditional_mutual_information(x, y, z):
    """I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X, Y|Z) = I(X;Y;Z) - I(X;Z)"""
    assert x.ndim == y.ndim == z.ndim == 2
    I_x_yz = mutual_information(x, np.concatenate([y, z], axis=1))
    I_x_z = mutual_information(x, z)
    return I_x_yz - I_x_z


def compute_ncmi(s, z, s_type, z_type, n_neighbors):
    """
    Normalized conditional mutual information between sources s and latents z.
    The [i, j] element is the conditional mutual information between the i-th source and the j-th latent
    given the rest of the sources, normalized by the conditional entropy of the i-th source given the rest
    of the sources.
    I(z_j; s_i | s_{-i}) = I(z_j; s_i, s_{-i}) - I(z_j; s_{-i}) = I(z_j; s) - I(z_j; s_{-i})
    NCMI[i, j] = I(z_j; s_i | s_{-i}) / H(s_i | s_{-i})
    """

    ds = s.shape[1]
    dz = z.shape[1]

    if s_type == 'discrete' and z_type == 'continuous':
        I_z_j_and_s_joint = np.empty(shape=(dz,))
        s_joint = LabelEncoder().fit_transform([str(s_sample) for s_sample in s])
        for j in range(dz):
            I_z_j_and_s_joint[j] = mutual_info_classif(
                z[:, j].reshape(-1, 1),
                s_joint,
                discrete_features=False,
                n_neighbors=n_neighbors
            ).squeeze()
        # print('I(x;y,z)', I_z_j_and_s_joint)
        ncmi = np.empty(shape=(ds, dz))
        for i in range(ds):
            s_rest = s[:, np.arange(ds) != i]
            s_rest_str = LabelEncoder().fit_transform([str(s_sample) for s_sample in s_rest])
            for j in range(dz):
                I_z_j_and_s_rest = mutual_info_classif(
                    z[:, j].reshape(-1, 1), s_rest_str, discrete_features=False, n_neighbors=n_neighbors
                ).squeeze()
                ncmi[i, j] = I_z_j_and_s_joint[j] - I_z_j_and_s_rest
            H_s_i_given_s_rest = conditional_entropy(s[:, i].reshape(-1, 1), s_rest)
            ncmi[i, :] /= H_s_i_given_s_rest
    elif s_type == 'discrete' and z_type == 'discrete':
        ncmi = np.empty(shape=(ds, dz))
        for i in range(ds):
            for j in range(dz):
                ncmi[i, j] = conditional_mutual_information(
                    z[:, j].reshape(-1, 1), s[:, i].reshape(-1, 1), s[:, np.arange(ds) != i]
                )
            H_s_i_given_s_rest = conditional_entropy(s[:, i].reshape(-1, 1), s[:, np.arange(ds) != i])
            ncmi[i, :] /= H_s_i_given_s_rest
    else:
        raise NotImplementedError

    ncmi[ncmi < 0] = 0
    ncmi[ncmi > 1] = 1
    return ncmi


def compute_linear_nmi(s, z, type):
    ds = s.shape[1]
    dz = z.shape[1]

    if type == 'continuous':
        s = StandardScaler().fit_transform(s)
        z = StandardScaler().fit_transform(z)
        nmi = np.empty(shape=(ds, dz))
        for i in range(ds):
            for j in range(dz):
                null = np.zeros_like(z[:, j].reshape(-1, 1))
                pH_s_i = linear_regression(null, s[:, i])
                pH_s_i_given_z_j = linear_regression(z[:, j].reshape(-1, 1), s[:, i])
                nmi[i, j] = (pH_s_i - pH_s_i_given_z_j) / pH_s_i
    else:
        raise NotImplementedError

    nmi[nmi < 0] = 0
    nmi[nmi > 1] = 1
    return nmi


def compute_linear_ncmi(s, z, type):
    ds = s.shape[1]
    dz = z.shape[1]

    if type == 'continuous':
        s = StandardScaler().fit_transform(s)
        z = StandardScaler().fit_transform(z)

        I_z_j_to_s_joint = np.empty(shape=(dz,))
        # I_z_j_to_s_joint_alt = np.empty(shape=(dz,))
        for j in range(dz):
            null = np.zeros_like(z[:, j].reshape(-1, 1))
            pH_s_joint_given_z_j = linear_regression(z[:, j].reshape(-1, 1), s)
            pH_s_joint = linear_regression(null, s)
            I_z_j_to_s_joint[j] = pH_s_joint - pH_s_joint_given_z_j
            # I_z_j_to_s_joint_alt[j] = LinearRegression().fit(
            #     z[:, j].reshape(-1, 1),
            #     s
            # ).score(
            #     z[:, j].reshape(-1, 1),
            #     s
            # ) * np.trace(np.cov(s.T))
        ncmi = np.empty(shape=(ds, dz))
        for i in range(ds):
            s_rest = s[:, np.arange(ds) != i]
            for j in range(dz):
                null = np.zeros_like(z[:, j].reshape(-1, 1))
                pH_s_rest_given_z_j = linear_regression(z[:, j].reshape(-1, 1), s_rest)
                pH_s_rest = linear_regression(null, s_rest)
                I_z_j_to_s_rest = pH_s_rest - pH_s_rest_given_z_j
                # I_z_j_to_s_rest = LinearRegression().fit(z[:, j].reshape(-1, 1), s_rest).score(z[:, j].reshape(-1,
                #                                                                                                1),
                #                                                                                s_rest) * np.trace(np.cov(s_rest.T))
                ncmi[i, j] = I_z_j_to_s_joint[j] - I_z_j_to_s_rest
            pH_s_i_given_s_rest = linear_regression(s_rest, s[:, i])
            ncmi[i, :] /= pH_s_i_given_s_rest
    elif type == 'discrete':
        I_z_j_to_s_joint = np.empty(shape=(dz,))
        s_joint = LabelEncoder().fit_transform([str(s_sample) for s_sample in s])
        for j in range(dz):
            null = np.zeros_like(z[:, j].reshape(-1, 1))
            z_j = OneHotEncoder().fit_transform(z[:, j].reshape(-1, 1))
            pH_s_joint_given_z_j = logistic_regression(z_j, s_joint)
            pH_s_joint = logistic_regression(null, s_joint)
            I_z_j_to_s_joint[j] = pH_s_joint - pH_s_joint_given_z_j
        ncmi = np.empty(shape=(ds, dz))
        for i in range(ds):
            s_rest = s[:, np.arange(ds) != i]
            s_rest_str = LabelEncoder().fit_transform([str(s_sample) for s_sample in s_rest])
            for j in range(dz):
                null = np.zeros_like(z[:, j].reshape(-1, 1))
                z_j = OneHotEncoder().fit_transform(z[:, j].reshape(-1, 1))
                pH_s_rest_given_z_j = logistic_regression(z_j, s_rest_str)
                pH_s_rest = logistic_regression(null, s_rest_str)
                I_z_j_to_s_rest = pH_s_rest - pH_s_rest_given_z_j
                ncmi[i, j] = I_z_j_to_s_joint[j] - I_z_j_to_s_rest
            s_i = LabelEncoder().fit_transform(s[:, i])
            s_rest_features = OneHotEncoder().fit_transform(s_rest)
            pH_s_i_given_s_rest = logistic_regression(s_rest_features, s_i)
            ncmi[i, :] /= pH_s_i_given_s_rest

    ncmi[ncmi < 0] = 0
    ncmi[ncmi > 1] = 1
    return ncmi


def compute_nmi(s, z, s_type, z_type, n_neighbors):
    ds = s.shape[1]
    dz = z.shape[1]
    nmi = np.empty(shape=(ds, dz))

    if s_type == 'discrete' and z_type == 'continuous':
        for i in range(ds):
            for j in range(dz):
                nmi[i, j] = mutual_info_classif(
                    z[:, j].reshape(-1, 1),
                    s[:, i],
                    discrete_features=False,
                    n_neighbors=n_neighbors
                )
            H_s_i = entropy(s[:, i])
            nmi[i, :] /= H_s_i
    elif s_type == 'discrete' and z_type == 'discrete':
        for i in range(ds):
            for j in range(dz):
                nmi[i, j] = mutual_information(s[:, i].reshape(-1, 1), z[:, j].reshape(-1, 1))
            H_s_i = entropy(s[:, i])
            nmi[i, :] /= H_s_i
    else:
        raise NotImplementedError
    nmi[nmi < 0] = 0
    nmi[nmi > 1] = 1
    return nmi


def compute_modularity_compactness(mi, z_active):
    ds = mi.shape[0]
    dz = np.sum(z_active)
    if dz == 0:
        return 0, 0
    pruned_mi = mi[:, z_active]
    modularity = (
            (np.mean(np.max(pruned_mi, axis=0) / np.sum(pruned_mi, axis=0)) - 1 / ds)
            / (1 - 1 / ds)
    )
    compactness = (
            (np.mean(np.max(pruned_mi, axis=1) / np.sum(pruned_mi, axis=1)) - 1 / dz)
            / (1 - 1 / dz)
    )
    return modularity, compactness


def logistic_regression(X, y):
    assert X.shape[0] == y.shape[0]
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.dtype in [np.float32, np.float64, np.int32, np.int64]
    assert y.dtype in [np.int32, np.int64]

    model = LogisticRegression(
        penalty=None,
        dual=False,
        tol=1e-4,
        fit_intercept=True,
        class_weight='balanced',
        solver='lbfgs',
        max_iter=10000,
        multi_class='multinomial',
        n_jobs=1,
        verbose=0,
    )

    model.fit(X, y)
    y_pred = model.predict_proba(X)
    return log_loss(y, y_pred)


def linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    return np.sum((y - y_pred) ** 2)


def compute_infoe(s, z, z_type):
    null = np.zeros_like(z)
    if z_type == 'discrete':
        z = OneHotEncoder().fit_transform(z)
    elif z_type == 'continuous':
        z = StandardScaler().fit_transform(z)
    else:
        raise NotImplementedError

    ds = s.shape[1]
    npi = np.empty(shape=(ds,))
    for i in range(ds):
        pH_s_i_given_z = logistic_regression(z, s[:, i])
        pH_s_i = logistic_regression(null, s[:, i])
        npi[i] = (pH_s_i - pH_s_i_given_z) / pH_s_i
    return np.mean(npi)


def process_discrete(x):
    if x.dtype in [np.float32, np.float64]:
        x_processed = []
        for i in range(x.shape[1]):
            x_processed.append(LabelEncoder().fit_transform(x[:, i]))
        x = np.stack(x_processed, axis=1)
    return x


def compute_metrics(s, z, s_type='discrete', z_type='continuous', n_neighbors=2, n_duplicate=2, z_noise=1e-2):
    """
    s: np.ndarray, shape=(n_samples, n_sources)
    z: np.ndarray, shape=(n_samples, n_latents)
    s_type: str, 'discrete' only
    z_type: str, 'continuous' or 'discrete'
    n_neighbors: int, number of neighbors for mutual information estimation
    n_duplicate: int, number of times to duplicate the data for KSG estimator
    z_noise: float, noise level to add to continuous latents
    """

    assert s.shape[0] == z.shape[0]
    assert z.dtype in [np.float32, np.float64]
    assert s_type == 'discrete'
    assert z_type in ['continuous', 'discrete']

    z_ranges = np.max(z, axis=0) - np.min(z, axis=0)
    max_z_range = np.max(z_ranges)
    z_active = z_ranges > 0.1 * max_z_range

    if z_type == 'continuous':
        # wacky hack so that discrete-continuous KSG estimator doesn't give bogus results
        s = np.tile(s, (n_duplicate, 1))
        z = np.tile(z, (n_duplicate, 1))

        z = StandardScaler().fit_transform(z)
        z = z + np.random.normal(0, z_noise, size=z.shape)
    else:
        z = process_discrete(z)
    s = process_discrete(s)

    nmi = compute_nmi(s, z, s_type, z_type, n_neighbors)
    ncmi = compute_ncmi(s, z, s_type, z_type, n_neighbors)

    ret = {}
    ret['infom'], ret['infoc'] = compute_modularity_compactness(nmi, z_active)
    ret['cinfom'], ret['cinfoc'] = compute_modularity_compactness(ncmi, z_active)
    ret['infoe'] = compute_infoe(s, z, z_type)
    ret['nmi'] = nmi
    ret['ncmi'] = ncmi
    ret['z_active'] = z_active
    return ret


def compute_linear_metrics(s, z, type):
    z_ranges = np.max(z, axis=0) - np.min(z, axis=0)
    max_z_range = np.max(z_ranges)
    z_active = z_ranges > 0.1 * max_z_range
    linear_ncmi = compute_linear_ncmi(s, z, type)
    linear_nmi = compute_linear_nmi(s, z, type)
    ret = {}
    ret['linear_cinfom'], ret['linear_cinfoc'] = compute_modularity_compactness(linear_ncmi, z_active)
    ret['linear_ncmi'] = linear_ncmi
    ret['linear_infom'], ret['linear_infoc'] = compute_modularity_compactness(linear_nmi, z_active)
    ret['linear_nmi'] = linear_nmi
    ret['z_active'] = z_active
    return ret


def get_inputs(specs, batch_size, T, input_size, every_n=1, resample_input=False, resample_output=False,
               intercept_x=0.0, intercept_y=1.0, gradient=1, rnns=None):
    x = np.random.uniform(low=specs.low, high=specs.high, size=(batch_size, T, input_size))

    # cut off corners of input
    num_resamples, prop_out_ = 0, 0
    if resample_input or resample_output:
        random = np.random.rand(batch_size, T)
    while resample_input:
        num_resamples += 1
        x, resample_input, prop_out = check_and_resample_input(x, specs, intercept_x=intercept_x,
                                                               intercept_y=intercept_y, gradient=gradient,
                                                               random=random)
        # print(num_resamples, prop_out)
        if num_resamples == 1:
            prop_out_ = prop_out

    # only input every n
    for t in range(T):
        if t % every_n != 0:
            x[:, t, :] = 0.0

    # cut off corners of output
    while resample_output and rnns is not None:
        num_resamples += 1
        x, resample_output, prop_out = check_and_resample_rnn_input(rnns, x, specs, every_n,
                                                                    intercept_x=intercept_x, intercept_y=intercept_y,
                                                                    gradient=gradient, random=random)
        # print(num_resamples, prop_out)
        if num_resamples == 1:
            prop_out_ = prop_out
    # if num_resamples>0:
    #    print('number of resamples:', num_resamples, prop_out_)
    return x, prop_out_


def check_and_resample_input(x, specs, intercept_x=0.0, intercept_y=1.0, gradient=1, random=None):
    condition_type = specs.condition_type

    # reimagine as if square was in [0,1] box to get intercept for [low,high] box
    if condition_type == 'corner_cut':
        meets_condition = corner_cut(x[..., 0], x[..., 1], intercept_x, intercept_y, gradient)
    elif condition_type == 'corner_cut_both':
        meets_condition = np.logical_or(corner_cut(x[..., 0], x[..., 1], intercept_x, intercept_y, gradient),
                                        corner_cut(x[..., 0], x[..., 1], -intercept_x, -intercept_y, gradient,
                                                   opp=True))
    elif condition_type == 'oversample_diagonal':
        meets_condition = oversample_diagonal(x[..., 0], x[..., 1], random, temp=0.9)
    elif condition_type == 'oversample_diagonal_2':
        meets_condition = np.logical_or(corner_cut(x[..., 0], x[..., 1], intercept_x, intercept_y, gradient),
                                        corner_cut(x[..., 0], x[..., 1], -intercept_x, -intercept_y, gradient,
                                                   opp=True))
        batch_to_change = random < specs.prop_batch_oversample
        meets_condition = np.logical_and(meets_condition, batch_to_change)
    elif condition_type == 'oversample_diagonals':
        meets_condition_1 = np.logical_or(corner_cut(x[..., 0], x[..., 1], intercept_x, intercept_y, gradient),
                                          corner_cut(x[..., 0], x[..., 1], -intercept_x, -intercept_y, gradient,
                                                     opp=True))
        meets_condition_2 = np.logical_or(
            corner_cut(x[..., 0], x[..., 1], intercept_x, intercept_y, -gradient, opp=True),
            corner_cut(x[..., 0], x[..., 1], -intercept_x, -intercept_y, -gradient))
        meets_condition = np.logical_not(
            np.logical_or(np.logical_not(meets_condition_1), np.logical_not(meets_condition_2)))
        batch_to_change = random < specs.prop_batch_oversample
        meets_condition = np.logical_and(meets_condition, batch_to_change)
    elif condition_type == 'oversample_edges':
        meets_condition = oversample_edges(x[..., 0], x[..., 1])
    elif condition_type == 'oversample_region':
        meets_condition = oversample_region(x[..., 0], x[..., 1], random, temp=0.8, xl=specs.xl, xh=specs.xh,
                                            yl=specs.yl, yh=specs.yh)
    elif condition_type == 'None':
        meets_condition = np.zeros_like(x[..., 0]).astype(bool)
    else:
        raise ValueError('Condition not specified')
    ys, xs = np.where(meets_condition)
    prop_out = np.mean(meets_condition)
    if len(xs) == 0:
        return x, False, prop_out
    x_new, _ = get_inputs(specs, 1, len(xs), x.shape[2], every_n=1)
    x[ys, xs, :] = x_new
    return x, True, prop_out


def check_and_resample_rnn_input(rnns, x, specs, every_n, intercept_x=0.0, intercept_y=1.0, gradient=1,
                                 random=None):
    condition_type = specs.condition_type

    batch_size, T, input_size = x.shape
    targets_1, _ = rnns[0](torch.from_numpy(x[..., :input_size // 2]).type(torch.float32))
    targets_2, _ = rnns[1](torch.from_numpy(x[..., input_size // 2:]).type(torch.float32))
    targets = torch.concatenate([targets_1, targets_2], axis=2).detach().numpy()

    if condition_type == 'corner_cut':
        meets_condition = corner_cut(targets[..., 0], targets[..., 1], intercept_x, intercept_y, gradient)
    elif condition_type == 'oversample_diagonal':
        meets_condition = oversample_diagonal(targets[..., 0], targets[..., 1], random, temp=0.4)
    elif condition_type == 'oversample_region':
        meets_condition = oversample_region(targets[..., 0], targets[..., 1], random, temp=0.2, xl=specs.xl,
                                            xh=specs.xh, yl=specs.yl, yh=specs.yh)
    elif condition_type == 'None':
        meets_condition = np.zeros_like(targets[..., 0]).astype(bool)
    else:
        raise ValueError('Condition not specified')

    batches = np.where(np.any(meets_condition, axis=1))
    prop_out = np.mean(np.any(meets_condition, axis=1))

    # need to remove whole batch as it's a sequence
    if len(batches[0]) == 0:
        return x, False, prop_out
    x_new, _ = get_inputs(specs, len(batches[0]), T, input_size, every_n=every_n)
    x[batches[0], :] = x_new
    return x, True, prop_out


def oversample_diagonal(x1, x2, random, temp=0.1):
    # doesn't work well for rnn outputs
    return random > np.exp(-temp * (x1 - x2) ** 2)


def oversample_edges(x1, x2, lim=0.9):
    return np.logical_not(np.logical_or.reduce((x1 > lim, x1 < -lim, x2 > lim, x2 < -lim)))


def oversample_region(x1, x2, random, temp=0.1, xl=-0.5, xh=0.5, yl=-0.5, yh=0.5):
    a = np.logical_or(x1 < xl, x1 > xh)
    b = np.logical_or(x2 < yl, x2 > yh)
    return np.logical_and(random < temp, np.logical_or(a, b))


def corner_cut(x1, x2, intercept_x, intercept_y, gradient, opp=False):
    # y - iny = m (x - inx)
    if opp:
        return x2 - intercept_y <= gradient * (x1 - intercept_x)
    else:
        return x2 - intercept_y > gradient * (x1 - intercept_x)


def make_directories(base_path='../Summaries_disrnn/', params=None):
    """
    Creates directories for storing data during a model training run
    """

    if params is not None:
        try:
            org_rule = [(x.split('.')[0], x.split('.')[-1]) for x in params.misc.org_rule]
            name = [str(params[a][b]) for (a, b) in org_rule]
            for i, n in enumerate(name):
                n = n.replace(',', '')
                n = n.replace('.', '')
                n = n.replace(' ', '')
                if n == 'loop':
                    n = n + '_' + params['data']['behaviour_type']
                name[i] = n
            name = ' ' + ' '.join(name)
        except KeyError:
            name = ''
    else:
        name = ''

    # Get current date for saving folder
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    # Initialise the run and dir_check to create a new run folder within the current date
    run = 0
    dir_check = True
    # Initialise all paths
    train_path, model_path, save_path, script_path, run_path = None, None, None, None, None
    # Find the current run: the first run that doesn't exist yet
    while dir_check:
        # Construct new paths (allowed a max of 10000 runs per day)
        run_name = date + name + '/run' + ('000' + str(run))[-4:]
        run_path = base_path + run_name + '/'
        train_path = run_path + 'train'
        model_path = run_path + 'model'
        save_path = run_path + 'save'
        script_path = run_path + 'script'
        run += 1
        # And once a path doesn't exist yet: create new folders
        if not os.path.exists(train_path) and not os.path.exists(model_path) and not os.path.exists(save_path):
            try:
                os.makedirs(train_path)
                os.makedirs(model_path)
                os.makedirs(save_path)
                os.makedirs(script_path)
                dir_check = False
            except FileExistsError:
                # often multiple jobs run at same time and get fudged here, so add this catch statement
                pass
    if run > 10000:
        raise ValueError("While loop for making directory was going on forever")

    # Return folders to new path
    return run_path, train_path, model_path, save_path, script_path, run_name


def make_logger(run_path, name):
    """
    Creates logger so output during training can be stored to file in a consistent way
    """

    # Create new logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Remove anly existing handlers so you don't output to old files, or to new files twice
    # - important when resuming training existing model
    logger.handlers = []
    # Create a file handler, but only if the handler does
    handler = logging.FileHandler(run_path + name + '.log')
    handler.setLevel(logging.INFO)
    # Create a logging format
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    handler.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(handler)
    # Return the logger object
    return logger


class MyDataset(Dataset):
    def __init__(self, inputs, targets, h_1, h_2, batch_size=16, shuffle=True):
        permutation_indices = np.random.permutation(len(inputs))
        self.inputs = inputs[permutation_indices] if shuffle else inputs
        self.targets = targets[permutation_indices] if shuffle else targets
        self.h_1 = h_1[permutation_indices] if shuffle else inputs
        self.h_2 = h_2[permutation_indices] if shuffle else targets
        self.iteration = 0
        self.num_epochs = 0
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.targets[index]
        h1 = self.h_1[index]
        h2 = self.h_2[index]
        return x, y

    def __len__(self):
        return len(self.inputs)

    def next(self):
        start, stop = self.iteration * self.batch_size, (self.iteration + 1) * self.batch_size
        if stop > len(self.inputs):
            self.iteration = 0
            self.num_epochs += 1
            if self.shuffle:
                permutation_indices = np.random.permutation(len(self.inputs))
                self.inputs = self.inputs[permutation_indices]
                self.targets = self.targets[permutation_indices]
                self.h_1 = self.h_1[permutation_indices]
                self.h_2 = self.h_2[permutation_indices]
            start, stop = self.iteration * self.batch_size, (self.iteration + 1) * self.batch_size

        x = self.inputs[start:stop]
        y = self.targets[start:stop]
        h1 = self.h_1[start:stop]
        h2 = self.h_2[start:stop]
        self.iteration += 1
        return x, y, h1, h2


class GECO:
    def __init__(self, geco_pars, beta):
        super(GECO, self).__init__()

        self.threshold = geco_pars['threshold']
        self.alpha = geco_pars['alpha']
        self.gamma = geco_pars['gamma']
        self.moving_average = 0.0
        self.batch_index = 0
        self.beta = beta

    def update(self, loss):
        constraint = loss - self.threshold
        if self.batch_index == 0:
            self.moving_average = constraint
        else:
            self.moving_average = self.alpha * self.moving_average + (1 - self.alpha) * constraint
        constraint = constraint + (self.moving_average - constraint).detach()

        constraint = constraint.detach()

        # update beta params
        self.beta = self.beta * torch.exp(self.gamma * constraint)
        # self.beta = torch.clamp(self.beta, min=0.0, max=10.0)
        self.batch_index += 1

        return self.beta


def get_data(specs, n_data=None):
    hidden_size = specs.hidden_size
    output_size = specs.output_size
    input_size = specs.input_size
    teacher_rnn_type = specs.teacher_rnn_type
    every_n = specs.every_n
    intercept_x = specs.intercept_x
    intercept_y = specs.intercept_y
    gradient = specs.gradient
    resample_input = specs.resample_input
    resample_output = specs.resample_output
    high = specs.high
    low = specs.low
    targets_scale = specs.targets_scale
    T = specs.T
    if n_data is None:
        n_data = specs.n_data

    def out_in(inps, tars, h_):
        return tars - teacher_rnn.w_out(teacher_rnn.rnn_step(h_, inps))

    def hidden_in(inps, hids, h_):
        return hids - teacher_rnn.rnn_step(h_, inps)

    match_hidden = True
    if match_hidden:
        f = hidden_in
        dim = hidden_size
        dim_i = hidden_size
    else:
        f = out_in
        dim = output_size
        dim_i = input_size
    st = time.time()
    # show example data
    teacher_rnn = RNN(hidden_size, dim_i, 2 * output_size, activation_type=teacher_rnn_type, orthog=True)
    inputs, prop_out = get_inputs(specs, n_data, T, dim_i, every_n=every_n, intercept_x=intercept_x,
                                  intercept_y=intercept_y, gradient=gradient, resample_input=resample_input,
                                  resample_output=resample_output)
    targets_og, _ = get_inputs(specs, n_data, T, dim, every_n=every_n, intercept_x=intercept_x,
                               intercept_y=intercept_y, gradient=gradient, resample_input=resample_input,
                               resample_output=resample_output)

    with torch.no_grad():
        inputs_torch = torch.from_numpy(inputs).type(torch.float32)
        targets = torch.from_numpy(targets_og).type(torch.float32) * targets_scale

        # for rnn1
        h = torch.zeros(n_data, teacher_rnn.hidden_size)
        diffs = []
        for t in range(20):
            inps_t = torch.zeros_like(inputs_torch[:, t, ...]) + 1e-6
            tars_t = targets[:, t, ...]

            minimise_inputs = True
            if teacher_rnn_type == 'tanh':
                finv = torch.atanh
            else:
                finv = nn.Identity()
            if minimise_inputs:
                # across all batch, do linear sum alignment so only have to add small input on...
                p = teacher_rnn.rnn_step(h, inps_t)
                # do it in batches otherwise memory explodes...
                chunk = 196
                for i in range(np.ceil(n_data / chunk).astype(int)):
                    if (i + 1) * chunk >= n_data:
                        end = None
                    else:
                        end = (i + 1) * chunk
                    start = i * chunk

                    cost = torch.sum((finv(p[start:end, None, ...]) - finv(tars_t[None, start:end, ...])) ** 2,
                                     axis=2).detach().numpy()
                    row_ind, col_ind = linear_sum_assignment(cost)

                    tars_t[start:end] = tars_t[start:end][col_ind, ...]
                targets[:, t, ...] = tars_t

            for i in range(6):
                a = vmap(jacrev(f, argnums=0))(inps_t, tars_t, h)
                update = torch.matmul(torch.linalg.inv(a), f(inps_t, tars_t, h)[..., None])[:, :, 0]
                inps_t = inps_t - torch.clamp(update, min=-2, max=2)

            diffs.append(torch.mean(f(inps_t, tars_t, h)).detach().numpy())
            h = teacher_rnn.rnn_step(h, inps_t)
            inputs_torch[:, t, ...] = inps_t

        print('diffs', np.mean(diffs))

    print('time:', time.time() - st)

    targets_hat, h = teacher_rnn(inputs_torch)
    targets_hat = targets_hat.detach().numpy() / (targets_scale * (high - low))
    inputs_hat = inputs_torch.detach().numpy()
    h = h.detach().numpy()

    print('input norm', np.mean(inputs_hat ** 2))

    # Calculate Pearson Correlation Coefficient and the p-value
    print('corr input', pearsonr(inputs_hat.reshape(-1, input_size)[:, 0], inputs_hat.reshape(-1, input_size)[:, 1])[0])
    print('corr target',
          pearsonr(targets_hat.reshape(-1, 2 * output_size)[:, 0], targets_hat.reshape(-1, 2 * output_size)[:, 1])[0])
    # Calculate Mutual Information
    id1 = histogram_discretize(inputs_hat.reshape(-1, input_size)[:, :1].T)
    id2 = histogram_discretize(inputs_hat.reshape(-1, input_size)[:, 1:].T)
    print('mi input', mutual_info_score(id1[0, :], id2[0, :]))  # WILL LOOK ARTIFICIALLY HIGHER DUE TO EVERY_N
    id1 = histogram_discretize(targets_hat.reshape(-1, 2 * output_size)[:, :1].T)
    id2 = histogram_discretize(targets_hat.reshape(-1, 2 * output_size)[:, 1:].T)
    print('mi target', mutual_info_score(id1[0, :], id2[0, :]))

    # corrs/mi on input / target
    sources = h.reshape(-1, hidden_size)
    print('corr hidden', pearsonr(sources[:, 0], sources[:, 1])[0])
    sources = discretize_binning(sources, bins='auto')
    nmi = normalized_multiinformation(sources)
    print('mi hidden', nmi)

    return inputs_hat, targets_hat, h, h, teacher_rnn, teacher_rnn
