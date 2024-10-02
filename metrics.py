import ipdb
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import log_loss, mean_squared_error


def entropy(x):
    _, counts = np.unique(x, return_counts=True, axis=0)
    p_x = counts / x.shape[0]
    return -np.sum(p_x * np.log(p_x))


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

def compute_linear_nmi(s, z, s_type, z_type):
    ds = s.shape[1]
    dz = z.shape[1]

    if s_type == 'continuous' and z_type == 'continuous':
        s = StandardScaler().fit_transform(s)
        z = StandardScaler().fit_transform(z)
        nmi = np.empty(shape=(ds, dz))
        for i in range(ds):
            for j in range(dz):
                null = np.zeros_like(z[:, j].reshape(-1, 1))
                pH_s_i = linear_regression(null, s[:, i])
                pH_s_i_given_z_j = linear_regression(z[:, j].reshape(-1, 1), s[:, i])
                nmi[i, j] = (pH_s_i - pH_s_i_given_z_j) / pH_s_i
    elif s_type == 'discrete' and z_type == 'continuous':
        z = StandardScaler().fit_transform(z)
        nmi = np.empty(shape=(ds, dz))
        for i in range(ds):
            for j in range(dz):
                null = np.zeros_like(z[:, j].reshape(-1, 1))
                s_i = LabelEncoder().fit_transform(s[:, i])
                pH_s_i = logistic_regression(null, s_i)
                pH_s_i_given_z_j = logistic_regression(z[:, j].reshape(-1, 1), s_i)
                nmi[i, j] = (pH_s_i - pH_s_i_given_z_j) / pH_s_i
    else:
        raise NotImplementedError

    nmi[nmi < 0] = 0
    nmi[nmi > 1] = 1
    return nmi

def compute_linear_ncmi(s, z, s_type, z_type):
    ds = s.shape[1]
    dz = z.shape[1]

    if s_type == 'continuous' and z_type == 'continuous':
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
    elif s_type == 'discrete' and z_type == 'continuous':
        z = StandardScaler().fit_transform(z)
        I_z_j_to_s_joint = np.empty(shape=(dz,))
        s_joint = LabelEncoder().fit_transform([str(s_sample) for s_sample in s])
        for j in range(dz):
            null = np.zeros_like(z[:, j].reshape(-1, 1))
            pH_s_joint_given_z_j = logistic_regression(z[:, j].reshape(-1, 1), s_joint)
            pH_s_joint = logistic_regression(null, s_joint)
            I_z_j_to_s_joint[j] = pH_s_joint - pH_s_joint_given_z_j
        ncmi = np.empty(shape=(ds, dz))
        for i in range(ds):
            s_rest = s[:, np.arange(ds) != i]
            s_rest_str = LabelEncoder().fit_transform([str(s_sample) for s_sample in s_rest])
            for j in range(dz):
                null = np.zeros_like(z[:, j].reshape(-1, 1))
                pH_s_rest_given_z_j = logistic_regression(z[:, j].reshape(-1, 1), s_rest_str)
                pH_s_rest = logistic_regression(null, s_rest_str)
                I_z_j_to_s_rest = pH_s_rest - pH_s_rest_given_z_j
                ncmi[i, j] = I_z_j_to_s_joint[j] - I_z_j_to_s_rest
            s_i = LabelEncoder().fit_transform(s[:, i])
            s_rest_features = OneHotEncoder().fit_transform(s_rest)
            pH_s_i_given_s_rest = logistic_regression(s_rest_features, s_i)
            ncmi[i, :] /= pH_s_i_given_s_rest
    elif s_type == 'discrete' and z_type == 'discrete':
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
        max_iter=3000,
        multi_class='multinomial',
        n_jobs=-1,
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


def compute_linear_metrics(s, z, s_type, z_type):
    z_ranges = np.max(z, axis=0) - np.min(z, axis=0)
    max_z_range = np.max(z_ranges)
    z_active = z_ranges > 0.1 * max_z_range
    linear_ncmi = compute_linear_ncmi(s, z, s_type, z_type)
    linear_nmi = compute_linear_nmi(s, z, s_type, z_type)
    ret = {}
    ret['linear_cinfom'], ret['linear_cinfoc'] = compute_modularity_compactness(linear_ncmi, z_active)
    ret['linear_ncmi'] = linear_ncmi
    ret['linear_infom'], ret['linear_infoc'] = compute_modularity_compactness(linear_nmi, z_active)
    ret['linear_nmi'] = linear_nmi
    ret['z_active'] = z_active
    return ret

