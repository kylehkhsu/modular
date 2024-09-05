import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import log_loss

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
    contributions = np.array(contributions)  # Convert list to numpy array for easier manipulation
    mean_contributions = np.mean(contributions, axis=1)  # Mean across m
    std_contributions = np.std(contributions, axis=1)  # Standard deviation across m

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(variables, mean_contributions, yerr=std_contributions, fmt='x', color='Purple')
    if type == 'dropout':
        ax.set_xlabel('Dropout')
    elif type == 'correlation':
        ax.set_xlabel('Correlation Strength')
    ax.set_ylabel('Mixed-Selectivity Index')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_MI_vs_mi(contributions, mutual_information, type=None):
    contributions = np.array(contributions)  # Convert list to numpy array for easier manipulation
    mean_contributions = np.mean(contributions, axis=1)  # Mean across m
    std_contributions = np.std(contributions, axis=1)  # Standard deviation across m

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(mutual_information, mean_contributions, yerr=std_contributions, fmt='x', color='Purple')

    ax.set_ylabel('Mixed-Selectivity Index')
    ax.set_xlabel('Mutual Information')
    plt.tight_layout()
    plt.show()

def plot_all_MI_vs_mi(contributions, mutual_informations):
    corner_mi, diag_mi, corr_mi = mutual_informations
    corner_cont, diag_cont, corr_cont = contributions

    # Convert lists to numpy arrays
    corner_cont = np.array(corner_cont)
    diag_cont = np.array(diag_cont)
    corr_cont = np.array(corr_cont)

    # Compute means and standard deviations
    mean_corner_cont = np.mean(corner_cont, axis=1)
    std_corner_cont = np.std(corner_cont, axis=1)
    mean_diag_cont = np.mean(diag_cont, axis=1)
    std_diag_cont = np.std(diag_cont, axis=1)
    mean_corr_cont = np.mean(corr_cont, axis=1)
    std_corr_cont = np.std(corr_cont, axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each set of contributions with error areas
    ax.plot(corner_mi, mean_corner_cont, color='royalblue', label='Corner Dropout')
    ax.fill_between(corner_mi, mean_corner_cont - std_corner_cont, mean_corner_cont + std_corner_cont, color='royalblue', alpha=0.2)

    ax.plot(diag_mi, mean_diag_cont, color='forestgreen', label='Diagonal Dropout')
    ax.fill_between(diag_mi, mean_diag_cont - std_diag_cont, mean_diag_cont + std_diag_cont, color='forestgreen', alpha=0.2)

    ax.plot(corr_mi, mean_corr_cont, color='firebrick', label='Correlations')
    ax.fill_between(corr_mi, mean_corr_cont - std_corr_cont, mean_corr_cont + std_corr_cont, color='firebrick', alpha=0.2)

    ax.set_ylabel('Mixed-Selectivity Index')
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

def plot_lpmi(metrics, ss_mi):
    linear_infom = [[m['linear_infom'] for m in metric] for metric in metrics]
    linear_cinfom = [[m['linear_cinfom'] for m in metric] for metric in metrics]

    mean_infom = np.mean(linear_infom, axis=1)
    std_infom = np.std(linear_infom, axis=1)

    mean_cinfom = np.mean(linear_cinfom, axis=1)
    std_cinfom = np.std(linear_cinfom, axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ss_mi, mean_cinfom, color='royalblue', label='Linear NCMI')
    ax.fill_between(ss_mi, mean_cinfom - std_cinfom, mean_cinfom + std_cinfom, color='royalblue', alpha=0.2)
    ax.plot(ss_mi, mean_infom, color='forestgreen', label='Linear NMI')
    ax.fill_between(ss_mi, mean_infom - std_infom, mean_infom + std_infom, color='forestgreen', alpha=0.2)

    ax.set_ylabel('Linear Mutual Information')
    ax.set_xlabel('Mutual Information')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_all_lpmi(lpmis, mutual_informations, type=None):
    corner_mi, diag_mi, corr_mi = mutual_informations
    corner_lpmis, diag_lpmis, corr_lpmis = lpmis

    corner_infom = [[m['linear_infom'] for m in metric] for metric in corner_lpmis]
    corner_cinfom = [[m['linear_cinfom'] for m in metric] for metric in corner_lpmis]
    diag_infom = [[m['linear_infom'] for m in metric] for metric in diag_lpmis]
    diag_cinfom = [[m['linear_cinfom'] for m in metric] for metric in diag_lpmis]
    corr_infom = [[m['linear_infom'] for m in metric] for metric in corr_lpmis]
    corr_cinfom = [[m['linear_cinfom'] for m in metric] for metric in corr_lpmis]

    # Compute means and standard deviations
    mean_corner_infom = np.mean(corner_infom, axis=1)
    std_corner_infom = np.std(corner_infom, axis=1)
    mean_corner_cinfom = np.mean(corner_cinfom, axis=1)
    std_corner_cinfom = np.std(corner_cinfom, axis=1)
    mean_diag_infom = np.mean(diag_infom, axis=1)
    std_diag_infom = np.std(diag_infom, axis=1)
    mean_diag_cinfom = np.mean(diag_cinfom, axis=1)
    std_diag_cinfom = np.std(diag_cinfom, axis=1)
    mean_corr_infom = np.mean(corr_infom, axis=1)
    std_corr_infom = np.std(corr_infom, axis=1)
    mean_corr_cinfom = np.mean(corr_cinfom, axis=1)
    std_corr_cinfom = np.std(corr_cinfom, axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))

    if type == 'cinfom':
        ax.plot(corner_mi, mean_corner_cinfom, color='royalblue', label='Corner Dropout')
        ax.fill_between(corner_mi, mean_corner_cinfom - std_corner_cinfom, mean_corner_cinfom + std_corner_cinfom, color='royalblue', alpha=0.2)

        ax.plot(diag_mi, mean_diag_cinfom, color='forestgreen', label='Diagonal Dropout')
        ax.fill_between(diag_mi, mean_diag_cinfom - std_diag_cinfom, mean_diag_cinfom + std_diag_cinfom, color='forestgreen', alpha=0.2)

        ax.plot(corr_mi, mean_corr_cinfom, color='firebrick', label='Correlations')
        ax.fill_between(corr_mi, mean_corr_cinfom - std_corr_cinfom, mean_corr_cinfom + std_corr_cinfom, color='firebrick', alpha=0.2)

        ax.set_ylabel('Conditional Linear Predictive Information')
    elif type == 'infom':
        ax.plot(corner_mi, mean_corner_infom, color='royalblue', label='Corner Dropout')
        ax.fill_between(corner_mi, mean_corner_infom - std_corner_infom, mean_corner_infom + std_corner_infom, color='royalblue', alpha=0.2)

        ax.plot(diag_mi, mean_diag_infom, color='forestgreen', label='Diagonal Dropout')
        ax.fill_between(diag_mi, mean_diag_infom - std_diag_infom, mean_diag_cinfom + std_diag_infom, color='forestgreen', alpha=0.2)

        ax.plot(corr_mi, mean_corr_infom, color='firebrick', label='Correlations')
        ax.fill_between(corr_mi, mean_corr_infom - std_corr_infom, mean_corr_cinfom + std_corr_infom, color='firebrick', alpha=0.2)

        ax.set_ylabel('Linear Predictive Information')
    ax.set_xlabel('Normalised Mutual Information')
    ax.legend()
    plt.tight_layout()
    plt.show()


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
