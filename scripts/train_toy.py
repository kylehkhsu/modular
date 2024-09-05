import matplotlib

matplotlib.use('Agg')

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import optax
import einops
import numpy as np
import math
import tensorflow as tf
import tensorflow_datasets as tfds

import ipdb
import tqdm
import pprint
import wandb
import omegaconf
import hydra
import pathlib
import os
import contextlib
import pandas as pd
import timeit
import functools
import sys

from metrics import compute_metrics, normalized_multiinformation
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


def generate_square_data(d, n_values_per_source, slice, correlated_sampling_factor, covariance, *, key):
    assert not (slice > 0 and correlated_sampling_factor > 0)
    index = ','.join([f'0:1:{n_values_per_source}j' for _ in range(d)])
    s = eval(f'np.mgrid[{index}]')
    s = s.reshape(d, -1).T
    if slice > 0:
        s = s[np.sum(s, axis=1) >= slice]

    elif correlated_sampling_factor > 0:
        mean = np.zeros((d,))
        cov = covariance * np.ones((d, d)) + (1 - covariance) * np.eye(d)
        s_correlated = jax.random.multivariate_normal(key, mean, cov, (int(correlated_sampling_factor * s.shape[0]),))
        quantiles = np.quantile(s_correlated, np.array([0.005, 0.995]), axis=0)
        quantile_range = np.mean(quantiles[1] - quantiles[0])
        s_correlated = s_correlated / quantile_range + 0.5
        # find nearest neighbors of s_correlated in s
        s_correlated_quantized = np.round(s_correlated * (n_values_per_source - 1)) / (n_values_per_source - 1)
        # remove entries where any value is outside [0, 1]
        s_correlated_quantized = s_correlated_quantized[
            np.all((s_correlated_quantized >= 0) & (s_correlated_quantized <= 1), axis=1)
        ]
        s = np.concatenate([s, s_correlated_quantized], axis=0)
    x = s
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    return {
        'x': x,
        's': s,
    }


def generate_square_mlp_data(n_samples, d_source, d_data, mlp_activation, n_mlp_layers, d_mlp_hidden, slice, *, key):
    keys = iter(jax.random.split(key, 100))
    s = jax.random.uniform(next(keys), shape=(n_samples, d_source), minval=0, maxval=1)
    # filter out s below s1 = -s0 + slice
    s = s[s[:, 1] > -s[:, 0] + slice]
    g = eqx.nn.MLP(d_source, d_data, d_mlp_hidden, n_mlp_layers, get_activation(mlp_activation), key=next(keys))
    x = jax.vmap(g)(s)
    x = (x - jnp.mean(x, axis=0)) / jnp.std(x, axis=0)
    return {
        'x': x,
        's': s,
    }


def get_activation(name):
    if name == 'relu':
        return jax.nn.relu
    elif name == 'tanh':
        return jnp.tanh
    elif name == 'sigmoid':
        return jax.nn.sigmoid
    elif name == 'identity':
        return lambda x: x
    else:
        raise ValueError(f'Unknown activation: {name}')


class MLPAutoencoder(eqx.Module):
    encoder: eqx.nn.MLP
    decoder: eqx.nn.MLP
    lambdas: dict
    n_weights: int
    n_biases: int
    negativity_loss: str
    regularize_biases: bool
    normalize_weight_energy: bool

    def __init__(
        self, d_input, d_latent, d_hidden, n_layers, mlp_activation, latent_activation, lambdas,
        negativity_loss, regularize_biases, normalize_weight_energy, *, key
        ):
        keys = iter(jax.random.split(key, 100))
        mlp_activation = get_activation(mlp_activation)
        latent_activation = get_activation(latent_activation)
        self.encoder = eqx.nn.MLP(
            d_input, d_latent, d_hidden, n_layers, mlp_activation, latent_activation, key=next(keys)
        )
        self.decoder = eqx.nn.MLP(
            d_latent, d_input, d_hidden, n_layers, mlp_activation, key=next(keys)
        )
        self.lambdas = lambdas
        self.negativity_loss = negativity_loss
        self.regularize_biases = regularize_biases
        self.normalize_weight_energy = normalize_weight_energy
        self.n_weights = sum(x.size for x in jtu.tree_leaves(eqx.filter(self, eqx.is_array)) if x.ndim > 1)
        self.n_biases = sum(x.size for x in jtu.tree_leaves(eqx.filter(self, eqx.is_array)) if x.ndim == 1)

    @eqx.filter_jit
    def loss(self, model, data, *, key=None):
        x = data['x']
        z = jax.vmap(model.encoder)(x)
        x_hat = jax.vmap(model.decoder)(z)
        losses = {}
        eps = 1e-5

        # losses['reconstruction'] = (jnp.sqrt(einops.reduce((x - x_hat) ** 2, 'n d -> n', 'sum') + eps) /
        #                             jnp.sqrt(x.shape[1]))
        # losses['activation_energy'] = jnp.sqrt(einops.reduce(z ** 2, 'n d -> n', 'sum') + eps) / jnp.sqrt(z.shape[1])
        # losses['activation_negativity'] = jnp.sqrt(einops.reduce((0.5 * (z - jnp.abs(z))) ** 2, 'n d -> n',
        #                                                          'sum') + eps) / jnp.sqrt(z.shape[1])
        # losses['weight_energy'] = model.l2()
        losses['reconstruction'] = einops.reduce((x - x_hat) ** 2, 'n d -> n', 'mean')
        losses['activation_energy'] = einops.reduce(z ** 2, 'n d -> n', 'mean')
        if model.negativity_loss == 'relu':
            losses['activation_negativity'] = einops.reduce(jnp.maximum(-z, 0), 'n d -> n', 'mean')
        elif model.negativity_loss == 'squared':
            losses['activation_negativity'] = einops.reduce((0.5 * (z - jnp.abs(z))) ** 2, 'n d -> n', 'mean')
        else:
            raise ValueError
        if model.regularize_biases:
            losses['weight_energy'] = model.l2()
            if model.normalize_weight_energy:
                losses['weight_energy'] /= (model.n_weights + model.n_biases)
        else:
            losses['weight_energy'] = model.l2_excluding_biases()
            if model.normalize_weight_energy:
                losses['weight_energy'] /= model.n_weights

        losses['weight_energy'] = model.l2()
        losses['total'] = sum(losses[k] * model.lambdas[k] for k in losses.keys())
        losses['sum'] = sum(losses.values())
        log = {f'loss/{k}': v for k, v in losses.items()}
        aux = {
            'z':     z,
            'x_hat': x_hat,
            'log':   log
        }
        return jnp.mean(losses['total']), aux

    def l2(self):
        sum_squares = sum(jnp.sum(x ** 2) for x in jtu.tree_leaves(eqx.filter(self, eqx.is_array)))
        return sum_squares

    def l2_excluding_biases(self):
        sum_squares = sum(jnp.sum(x ** 2) for x in jtu.tree_leaves(eqx.filter(self, eqx.is_array)) if x.ndim > 1)
        return sum_squares


@eqx.filter_jit
def train_step(model, optimizer_state, optimizer, data, *, key):
    (_, aux), grad = eqx.filter_value_and_grad(model.loss, has_aux=True)(model, data, key=key)
    update, optimizer_state = optimizer.update(grad, optimizer_state, model)
    model = eqx.apply_updates(model, update)
    return model, optimizer_state, aux['log']


def evaluate(model, val_set, config, step, *, key):
    log = {}
    log.update({
        'weight_norm/all': model.l2().item(),
        'weight_norm/excluding_biases': model.l2_excluding_biases().item()
    })
    auxs = []
    for i, batch in enumerate(val_set):
        key, sub_key = jax.random.split(key)
        _, aux = model.loss(model, batch, key=sub_key)
        aux['x'] = batch['x']
        aux['s'] = batch['s']
        auxs.append(aux)

    auxs = jax.tree_map(lambda *leaves: jnp.concatenate(leaves) if leaves[0].ndim > 0 else jnp.stack(leaves), *auxs)
    log.update({f'{k}/val': v.mean().item() for k, v in auxs['log'].items()})
    
    if step == 0:
        ds = min(config.data.d_source, 10)
        s = auxs['s']
        fig, axes = plt.subplots(ds, ds, figsize=(2 * ds, 2 * ds))
        for i in range(ds):
            for j in range(ds):
                ax = axes[i][j]
                sns.histplot(
                    ax=ax,
                    x=s[:, j],
                    y=s[:, i],
                    rasterized=True,
                    bins=config.data.n_values_per_source
                )
                ax.set_xlabel(rf'$s_{{{j}}}$')
                ax.set_ylabel(rf'$s_{{{i}}}$')
        fig.tight_layout()
        log.update({'pairwise_sources': wandb.Image(fig)}, step=0)
        plt.close()
        dx = min(auxs['x'].shape[1], 10)
        x = auxs['x']
        fig, axes = plt.subplots(dx, dx, figsize=(2 * dx, 2 * dx))
        for i in range(dx):
            for j in range(dx):
                ax = axes[i][j]
                sns.histplot(
                    ax=ax,
                    x=x[:, j],
                    y=x[:, i],
                    rasterized=True,
                    bins=config.data.n_values_per_source
                )
                ax.set_xlabel(rf'$x_{{{j}}}$')
                ax.set_ylabel(rf'$x_{{{i}}}$')
        fig.tight_layout()
        log.update({'pairwise_data': wandb.Image(fig)}, step=0)
        plt.close()

    # if step == 9999:
    #     # write s and z to disk
    #     np.savez('/iris/u/kylehsu/code/modular/s_and_z_uniform.npz', s=auxs['s'], z=auxs['z'])
    #     sys.exit(0)

    start = timeit.default_timer()
    metrics = compute_metrics(auxs['s'], auxs['z'], 'discrete', 'continuous')
    print(f'metrics: {timeit.default_timer() - start:.1f} s')

    log.update({f'metrics/{k}': v for k, v in metrics.items() if k in ['cinfom', 'cinfoc', 'infom', 'infoe', 'infoc']})


    def plot_mi_heatmap(k):
        mi = metrics[k]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(
            mi, ax=ax, annot=True, fmt='.2f', square=True, vmin=0, vmax=1, cbar=False,
            annot_kws={'fontsize': 8},
            xticklabels=[rf'$\mathbf{{z}}_{{{i}}}$' for i in range(mi.shape[1])],
            yticklabels=[rf'$\mathbf{{s}}_{{{i}}}$' for i in range(mi.shape[0])],
            rasterized=True
        )
        for i, label in enumerate(ax.get_xticklabels()):
            if metrics['z_active'][i] == 0:
                label.set_color('red')
        fig.tight_layout()
        log.update({f'{k}_heatmap': wandb.Image(fig)})
        plt.close()
    plot_mi_heatmap('nmi')
    plot_mi_heatmap('ncmi')

    z = auxs['z']
    nz = z.shape[1]
    s = auxs['s']
    ns = s.shape[1]

    data = pd.DataFrame(z, columns=[f'z{i}' for i in range(z.shape[1])])
    data['id'] = data.index
    data = data.melt(id_vars='id', var_name='z', value_name='value')
    fig, ax = plt.subplots(figsize=(z.shape[1] ** 0.8, 3))
    sns.violinplot(data=data, ax=ax, x='z', y='value', density_norm='width', cut=0)
    fig.tight_layout()
    log.update({f'latent_densities': wandb.Image(fig)})
    plt.close()

    fig, axes = plt.subplots(nz, nz, figsize=(2 * nz, 2 * nz))
    for i in range(nz):
        for j in range(nz):
            ax = axes[i][j]
            sns.histplot(
                ax=ax,
                x=z[:, j],
                y=z[:, i],
                rasterized=True,
            )
            ax.set_xlabel(rf'$z_{{{j}}}$')
            ax.set_ylabel(rf'$z_{{{i}}}$')
    fig.tight_layout()
    log.update({'pairwise_latents': wandb.Image(fig)})
    plt.close()

    fig, axes = plt.subplots(ns, nz, figsize=(2 * nz, 2 * ns))
    for i in range(ns):
        for j in range(nz):
            ax = axes[i][j]
            sns.histplot(
                ax=ax,
                x=z[:, j],
                y=s[:, i],
                rasterized=True
            )
            ax.set_xlabel(rf'$z_{{{j}}}$')
            ax.set_ylabel(rf'$s_{{{i}}}$')
    fig.tight_layout()
    log.update({'sources_latents': wandb.Image(fig)})
    plt.close()
    return log


def small_init(weight: jax.Array, scale: float, key: jax.random.PRNGKey) -> jax.Array:
    out, in_ = weight.shape
    stddev = scale * math.sqrt(1 / in_)
    return stddev * jax.random.normal(key, shape=(out, in_))


def init_linear_weight(model, init_fn, key):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [x.weight
                             for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                             if is_linear(x)]
    weights = get_weights(model)
    new_weights = [init_fn(weight, key=subkey)
                   for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model


def save(path, model, optimizer_state):
    path.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path / 'model.eqx', model)
    eqx.tree_serialise_leaves(path / 'optimizer_state.eqx', optimizer_state)
    print(f'saved model and optimizer state to {path}')


def main(config):
    api = wandb.Api()
    config_dict = omegaconf.OmegaConf.to_container(config, resolve=True)
    pprint.pprint(config_dict)
    run = wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
        config=config_dict,
        save_code=True,
        group=config.wandb.group,
        job_type=config.wandb.job_type,
        name=config.wandb.name,
        mode='disabled' if config.debug else 'online'
    )
    wandb.run.log_code(hydra.utils.get_original_cwd())
    wandb.config.update({'wandb_run_dir': wandb.run.dir})
    wandb.config.update({'hydra_run_dir': os.getcwd()})
    checkpoints_path = pathlib.Path(run.dir) / 'checkpoints'

    keys = iter(jax.random.split(jax.random.PRNGKey(config.experiment.seed), 100))
    model = hydra.utils.instantiate(config.model)(key=next(keys))
    if config.optim.small_init:
        small_init_with_scale = functools.partial(small_init, scale=config.optim.small_init_scale)
        model = init_linear_weight(model, small_init_with_scale, key=next(keys))

    optimizer = optax.adamw(learning_rate=config.optim.learning_rate, weight_decay=config.optim.weight_decay)
    # optimizer = optax.sgd(learning_rate=config.optim.learning_rate)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))

    data_key = jax.random.PRNGKey(config.data.seed)
    match config.data.name:
        case 'square':
            dataset = generate_square_data(
                config.data.d_source,
                config.data.n_values_per_source,
                config.data.slice,
                config.data.correlated_sampling_factor,
                config.data.covariance,
                key=data_key
            )
        case 'square_mlp':
            dataset = generate_square_mlp_data(
                config.data.n_train + config.data.n_val,
                config.data.d_source,
                config.data.d_data,
                config.data.mlp_activation,
                config.data.n_mlp_layers,
                config.data.d_mlp_hidden,
                config.data.slice,
                key=data_key
            )

        case _:
            raise ValueError
    wandb.log({'source_normalized_multiinformation': normalized_multiinformation(dataset['s'])}, step=0)
    train_set = val_set = tf.data.Dataset.from_tensor_slices(dataset)
    train_set = train_set.shuffle(
        train_set.cardinality(),
        seed=config.data.seed,
        reshuffle_each_iteration=True) \
        .repeat() \
        .batch(config.data.batch_size) \
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    val_set = val_set.batch(config.data.batch_size, drop_remainder=False) \
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    train_set, val_set = tfds.as_numpy(train_set), tfds.as_numpy(val_set)

    pbar = tqdm.tqdm(train_set, total=int(config.optim.n_steps))
    # pbar = tqdm.tqdm(range(config.optim.n_steps))
    context = jax.disable_jit if config.debug else contextlib.nullcontext
    train_key = next(keys)
    eval_key = next(keys)
    with context():
        for step, batch in enumerate(pbar):
            if step >= config.optim.n_steps:
                break

            if (step + 1) % config.checkpoint.period == 0:
                path = checkpoints_path / f'step={step}'
                save(path, model, optimizer_state)
                wandb.save(str(path / '*'), base_path=run.dir)

            if (step == 0 and not config.debug) \
                or (step + 1) % config.eval.period == 0:
                # ((step + 1 < config.eval.period) and (step + 1) % (config.eval.period // 5) == 0):
                model = eqx.nn.inference_mode(model, True)
                log = evaluate(model, val_set, config, step, key=eval_key)
                wandb.log(log, step=step)
                model = eqx.nn.inference_mode(model, False)

            train_key, sub_key = jax.random.split(train_key)
            model, optimizer_state, losses = train_step(model, optimizer_state, optimizer, batch, key=sub_key)
            wandb.log({f'{k}/train': v.mean().item() for k, v in losses.items()}, step=step)
    wandb.finish()
