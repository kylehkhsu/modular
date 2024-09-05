import numpy as np
import matplotlib.pyplot as plt

def plot_shape_position_variance(mean_variances, filename=None):
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.imshow(mean_variances, cmap='Purples')
    ax.set_xticks(range(mean_variances.shape[1]))
    ax.set_yticks(range(mean_variances.shape[0]))
    ax.set_xticklabels(range(mean_variances.shape[1]))
    ax.set_yticklabels(['Shape', 'Position'])
    ax.set_xlabel('Neurons')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_MI(contributions, variables, type=None):
    variables = [int(str(variable)[:2]) for variable in variables]
    n_models = len(contributions[0])
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Purples(np.linspace(0, 1, n_models))
    for i in range(n_models):
        ax.errorbar(variables, np.array(contributions)[:, i], fmt='x', color=colors[i])
    if type == 'dropout':
        ax.set_xlabel('Dropout')
    elif type == 'correlation_strengths':
        ax.set_xlabel('Correlation Strength')
    ax.set_ylabel('Modularity Index')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_MI_vs_mi(contributions, mutual_information, type=None):
    if type == 'dropout':
        mi = [m[0] for m in mutual_information]
    elif type == 'correlation_strengths':
        mi = mutual_information
    n_models = len(contributions[0])
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Purples(np.linspace(0, 1, n_models))
    for i in range(n_models):
        ax.errorbar(mi, np.array(contributions)[:, i], fmt='x', color=colors[i])
    ax.set_ylabel('Modularity Index')
    ax.set_xlabel('Mutual Information')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_ncmi(metrics):
    ncmi = metrics['ncmi']
    [n_sources, n_latents] = ncmi.shape
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(ncmi, cmap='Purples')
    ax.set_xticks(range(n_latents))
    ax.set_yticks(range(n_sources))
    ax.set_xticks(range(0, n_latents, 2))
    ax.set_yticklabels(['Shape', 'Position'])
    plt.show()



import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


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
    assert X.dtype in [np.float32, np.float64, np.int32]
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


def compute_infoe(s, z):
    ds = s.shape[1]
    npi = np.empty(shape=(ds,))
    null = np.zeros_like(z)
    for i in range(ds):
        pH_s_i_given_z = logistic_regression(z, s[:, i])
        pH_s_i = logistic_regression(null, s[:, i])
        npi[i] = (pH_s_i - pH_s_i_given_z) / pH_s_i
    return np.mean(npi)


def compute_metrics(s, z, s_type='discrete', z_type='continuous', n_neighbors=3, n_duplicate=5, z_noise=1e-2):
    """
    s: np.ndarray, shape=(n_samples, n_sources)
    z: np.ndarray, shape=(n_samples, n_latents)
    s_type: str, 'discrete' only
    z_type: str, 'continuous' only
    n_neighbors: int, number of neighbors for mutual information estimation
    n_duplicate: int, number of times to duplicate the data for KSG estimator
    z_noise: float, noise level to add to continuous latents
    """

    assert s.shape[0] == z.shape[0]
    assert z.dtype in [np.float32, np.float64]
    assert s_type == 'discrete'
    assert z_type == 'continuous'

    # wacky hack so that discrete-continuous KSG estimator doesn't give bogus results
    s = np.tile(s, (n_duplicate, 1))
    z = np.tile(z, (n_duplicate, 1))

    z = StandardScaler().fit_transform(z)
    z = z + np.random.normal(0, z_noise, size=z.shape)

    if s.dtype in [np.float32, np.float64]:
        s_processed = []
        for i in range(s.shape[1]):
            s_processed.append(LabelEncoder().fit_transform(s[:, i]))
        s = np.stack(s_processed, axis=1)

    nmi = compute_nmi(s, z, s_type, z_type, n_neighbors)
    ncmi = compute_ncmi(s, z, s_type, z_type, n_neighbors)

    z_ranges = np.max(z, axis=0) - np.min(z, axis=0)
    max_z_range = np.max(z_ranges)
    z_active = z_ranges > 0.1 * max_z_range

    ret = {}
    ret['infom'], ret['infoc'] = compute_modularity_compactness(nmi, z_active)
    ret['cinfom'], ret['cinfoc'] = compute_modularity_compactness(ncmi, z_active)
    ret['infoe'] = compute_infoe(s, z)
    ret['nmi'] = nmi
    ret['ncmi'] = ncmi
    ret['z_active'] = z_active
    return ret
