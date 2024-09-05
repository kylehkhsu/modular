import numpy as np
import matplotlib.pyplot as plt
import utilities as utils

dropouts = [0, 1]
data_distributions_3 = [np.load(f'ForKyle/data_distribution_dropout_{dropout}.npy') for dropout
                        in dropouts]
activity_data_3 = [np.load(f'ForKyle/responses_dropout_{dropout}.npy', allow_pickle=True) for dropout in dropouts]

all_contributions = []
all_variances = []
for d, dropout in enumerate(dropouts):
    num_shapes = num_positions = 9
    variances = {
        'shape' : [],
        'position' : []
    }
    for i, response in enumerate(activity_data_3[d]):
        print(f'Dropout: {dropout}, Model: {i+1}')
        activations = np.array(response['activations']).reshape(-1, 25)
        inputs_info = response['input']

        organised_activations = np.zeros((num_shapes, num_positions, activations.shape[1]))

        for idx, info in enumerate(inputs_info):
            shape_idx = info['what_index']
            position_idx = info['where_index']
            if data_distributions_3[d][shape_idx][position_idx] == 1:
                organised_activations[shape_idx, position_idx] = activations[idx]
            else:
                organised_activations[shape_idx, position_idx] = np.nan
                
        var_shape = np.nanvar(organised_activations, axis=1)
        mean_var_shape = np.mean(var_shape, axis=0)

        var_position = np.nanvar(organised_activations, axis=0)
        mean_var_position = np.mean(var_position, axis=0)

        mean_variances = np.vstack((mean_var_shape, mean_var_position))
        utils.plot_shape_position_variance(mean_variances)

        variances['shape'].append(mean_var_shape)
        variances['position'].append(mean_var_position)
    all_variances.append(variances)

    contributions = []
    for shape_var, position_var in zip(variances['shape'], variances['position']):
        total_var = shape_var + position_var
        min_var = np.minimum(shape_var, position_var)
        contribution = np.sum(min_var) / np.sum(total_var)
        contributions.append(contribution)
    all_contributions.append(contributions)
utils.plot_MI(all_contributions, dropouts)

all_metrics = []
for i, dropout in enumerate(dropouts):
    dropout_metrics = []
    for j, response in enumerate(activity_data_3[i]):
        print(f'Dropout: {dropout}, Model: {j+1}')
        activations = np.array(response['activations']).reshape(-1, 25)
        inputs_info = response['input']

        sources = np.zeros((81, 2))
        latents = np.zeros((81, 25))

        for idx, info in enumerate(inputs_info):
            sources[idx, 0] = info['what_index']
            sources[idx, 1] = info['where_index']
            if data_distributions_3[d][shape_idx][position_idx] == 1:
                latents[idx] = activations[idx]
            else:
                latents[idx] = np.nan
        sources = sources[~np.isnan(latents).any(axis=1)]
        latents = latents[~np.isnan(latents).any(axis=1)]
        metrics = utils.compute_metrics(sources, latents, s_type='discrete', z_type='continuous', n_neighbors=3, n_duplicate=5, z_noise=1e-2)
        utils.plot_ncmi(metrics)
        dropout_metrics.append(metrics)
    all_metrics.append(dropout_metrics)