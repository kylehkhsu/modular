# import infomec
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif


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
    """I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X, Y|Z)"""
    assert x.ndim == y.ndim == z.ndim == 2

    # H_x_given_z = conditional_entropy(x, z)
    # H_y_given_z = conditional_entropy(y, z)
    # H_xy_given_z = conditional_entropy(np.concatenate([x, y], axis=1), z)
    # return H_x_given_z + H_y_given_z - H_xy_given_z

    I_x_yz = mutual_information(x, np.concatenate([y, z], axis=1))
    I_x_z = mutual_information(x, z)
    return I_x_yz - I_x_z

def compute_ncmi(s, z, s_type, z_type, n_neighbors=3):
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
        s = np.tile(s, (5, 1))
        z = np.tile(z, (5, 1))
        z_std = np.std(z, axis=0)
        z = z + np.random.normal(0, z_std * 0.01, size=z.shape)
        I_z_j_and_s_joint = np.empty(shape=(dz,))
        s_joint = LabelEncoder().fit_transform([str(s_sample) for s_sample in s])
        for j in range(dz):
            I_z_j_and_s_joint[j] = mutual_info_classif(z[:, j].reshape(-1, 1), s_joint, discrete_features=False,
                                                       n_neighbors=n_neighbors).squeeze()
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
                # print('I(x;z)', I_z_j_and_s_rest)
            H_s_i_given_s_rest = conditional_entropy(s[:, i].reshape(-1, 1), s_rest)
            # print('entropy', H_s_i_given_s_rest)
            ncmi[i, :] /= H_s_i_given_s_rest
            # H_s_i = entropy(s[:, i].reshape(-1, 1))
            # print('entropy', H_s_i)
            # ncmi[i, :] /= H_s_i
        return ncmi
    elif s_type == 'discrete' and z_type == 'discrete':
        ncmi = np.empty(shape=(ds, dz))
        for i in range(ds):
            for j in range(dz):
                ncmi[i, j] = conditional_mutual_information(
                    z[:, j].reshape(-1, 1), s[:, i].reshape(-1, 1), s[:, np.arange(ds) != i]
                )
            H_s_i_given_s_rest = conditional_entropy(s[:, i].reshape(-1, 1), s[:, np.arange(ds) != i])
            ncmi[i, :] /= H_s_i_given_s_rest
        return ncmi
    else:
        raise NotImplementedError
def compute_cinfomec(s, z, s_type, z_type):
    """
    Compute CInfoM, CInfoC, and InfoE between discrete sources s and continuous latents z.
    s: np.ndarray, shape=(n_samples, n_sources)
    z: np.ndarray, shape=(n_samples, n_latents)

    Inactive latents are heuristically identified by theeir range being less than a proportion of the range of the
    most active latent. Inactive latents are ignored in the computation of CInfoM and CInfoC.
    """
    ncmi = compute_ncmi(s, z, s_type, z_type)
    ncmi[ncmi < 0] = 0

    z_ranges = np.max(z, axis=0) - np.min(z, axis=0)
    max_z_range = np.max(z_ranges)
    z_active = z_ranges > 0.1 * max_z_range

    ds = s.shape[1]
    ncmi_active = ncmi[:, z_active]
    dz_active = np.sum(z_active)
    if dz_active == 0:
        return {
            'cinfom':   0,
            'cinfoc':   0,
            # 'InfoE': 0,
            'ncmi':     ncmi,
            'z_active': z_active
        }
    cinfom = (np.mean(np.max(ncmi_active, axis=0) / np.sum(ncmi_active, axis=0)) - 1 / ds) / (
        1 - 1 / ds)
    cinfoc = (np.mean(np.max(ncmi_active, axis=1) / np.sum(ncmi_active, axis=1)) - 1 / dz_active) / (
        1 - 1 / dz_active)
    # infoe = infomec.compute_infoe(s, z, 'discrete', 'continuous')
    return {
        'cinfom':   cinfom,
        'cinfoc':   cinfoc,
        # 'InfoE': infoe,
        'ncmi':     ncmi,
        'z_active': z_active
    }