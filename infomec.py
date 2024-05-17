import ipdb
import numpy as np
from sklearn import preprocessing, feature_selection, metrics, linear_model

def process_individually(xs, xs_type):
    processed_xs = []
    if xs_type == 'discrete':
        for i in range(xs.shape[1]):
            processed_xs.append(preprocessing.LabelEncoder().fit_transform(xs[:, i]))
        processed_xs = np.stack(processed_xs, axis=1)
    elif xs_type == 'continuous':
        for i in range(xs.shape[1]):
            processed_xs.append(preprocessing.StandardScaler().fit_transform(xs[:, i][:, None]))
        processed_xs = np.concatenate(processed_xs, axis=1)
    else:
        raise ValueError(f'Unknown xs type: {xs_type}')
    return processed_xs



def compute_nmi(sources, latents, sources_type, latents_type):

    processed_sources = process_individually(sources, sources_type)
    processed_latents = process_individually(latents, latents_type)

    ret = np.empty(shape=(processed_sources.shape[1], processed_latents.shape[1]))
    for i in range(processed_sources.shape[1]):
        for j in range(processed_latents.shape[1]):
            if sources_type == 'discrete' and latents_type == 'discrete':
                ret[i, j] = metrics.mutual_info_score(processed_sources[:, i], processed_latents[:, j])
            elif sources_type == 'discrete' and latents_type == 'continuous':
                ret[i, j] = feature_selection.mutual_info_classif(
                    processed_latents[:, j][:, None], processed_sources[:, i], discrete_features=False,n_neighbors=10
                )
            elif sources_type == 'continuous' and latents_type == 'continuous':
                ret[i, j] = feature_selection.mutual_info_regression(
                    processed_latents[:, j][:, None], processed_sources[:, i], discrete_features=False, n_neighbors=10
                )
            else:
                raise ValueError(f'Unknown combination of sources and latents types: {sources_type} and {latents_type}')
        if sources_type == 'discrete':
            entropy = metrics.mutual_info_score(processed_sources[:, i], processed_sources[:, i])
        elif sources_type == 'continuous':
            entropy = feature_selection.mutual_info_regression(
                processed_sources[:, i][:, None], processed_sources[:, i], discrete_features=False, n_neighbors=10
            )
        else:
            raise ValueError(f'Unknown sources type: {sources_type}')
        ret[i, :] /= entropy
    return ret


def compute_infoe(sources, latents, sources_type, latents_type):
    normalized_predictive_information_per_source = []
    processed_sources = process_individually(sources, sources_type)

    if latents_type == 'discrete':
        processed_latents = preprocessing.OneHotEncoder().fit_transform(latents)
    elif latents_type == 'continuous':
        processed_latents = preprocessing.StandardScaler().fit_transform(latents)
    else:
        raise ValueError(f'Unknown latents type: {latents_type}')

    for i in range(processed_sources.shape[1]):
        if sources_type == 'discrete':
            predictive_conditional_entropy = logistic_regression(processed_latents, processed_sources[:, i])
            null = np.zeros_like(latents)
            marginal_source_entropy = logistic_regression(null, processed_sources[:, i])
            normalized_predictive_information_per_source.append(
                (marginal_source_entropy - predictive_conditional_entropy) / marginal_source_entropy
            )
        elif sources_type == 'continuous':
            coefficient_of_determination = linear_model.LinearRegression().fit(
                processed_latents, processed_sources[:, i]
            ).score(processed_latents, processed_sources[:, i])
            normalized_predictive_information_per_source.append(coefficient_of_determination)
        else:
            raise ValueError(f'Unknown sources type: {sources_type}')

    return np.mean(normalized_predictive_information_per_source)


def logistic_regression(X, y):
    assert X.shape[0] == y.shape[0]
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.dtype in [np.float32, np.float64]
    assert y.dtype in [np.int32, np.int64]

    model = linear_model.LogisticRegression(
        penalty=None,
        dual=False,
        tol=1e-4,
        fit_intercept=True,
        class_weight='balanced',
        solver='lbfgs',
        max_iter=100,
        multi_class='multinomial',
        n_jobs=1,
        verbose=0,
    )

    model.fit(X, y)
    y_pred = model.predict_proba(X)
    return metrics.log_loss(y, y_pred)


def normalized_gini(x):
    assert x.ndim == 1
    gini = np.sum(np.abs(x[:, None] - x[None, :])) / (2 * x.shape[0] ** 2) / np.mean(x)
    perfect_gini = 1 - 1 / x.shape[0]
    return gini / perfect_gini


def compute_infomec(sources, latents, sources_type, latents_type):
    nmi = compute_nmi(sources, latents, sources_type, latents_type)

    latent_ranges = np.max(latents, axis=0) - np.min(latents, axis=0)
    max_latent_range = np.max(latent_ranges)
    active_latents = latent_ranges > 0.1 * max_latent_range

    num_sources = sources.shape[1]
    num_active_latents = np.sum(active_latents)
    pruned_nmi = nmi[:, active_latents]
    if num_active_latents == 0:
        return {
            'infom':          0,
            'infoc':          0,
            'infoe':          0,
            'nmi':            nmi,
            'active_latents': active_latents,
        }

    infom = (np.mean(np.max(pruned_nmi, axis=0) / np.sum(pruned_nmi, axis=0)) - 1 / num_sources) / (
                1 - 1 / num_sources)
    infoc = (np.mean(np.max(pruned_nmi, axis=1) / np.sum(pruned_nmi, axis=1)) - 1 / num_active_latents) / (
                1 - 1 / num_active_latents)

    infoe = compute_infoe(sources, latents, sources_type, latents_type)

    return {
        'infom': infom,
        'infoc': infoc,
        'infoe': infoe,
        'nmi': nmi,
        'active_latents': active_latents,
    }


if __name__ == '__main__':
    # sources = np.random.randint(0, 10, size=(1000, 6))
    sources = np.random.random(size=(1000, 2))
    # latents = sources
    # latents = sources + np.random.normal(size=sources.shape)
    latents = np.tanh(sources)
    # sources = np.random.random(size=(1000, 6))
    # compute_infomec(sources, latents, discrete_latents=False)
    # sources = np.linspace(0, 1, 100).reshape(-1, 1)
    # latents = np.linspace(0, 1, 100).reshape(-1, 1)
    infomec = compute_infomec(sources, latents, 'continuous', 'continuous')
    import pprint
    pprint.pprint(infomec)