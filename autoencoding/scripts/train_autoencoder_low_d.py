import equinox as eqx
import ipdb
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import einops
import optax

import omegaconf
import wandb
import pprint
import hydra
import os
import pathlib
import math
import functools
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
import contextlib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import metrics
import pandas as pd


class MLPAutoencoder(eqx.Module):
    encoder: eqx.nn.MLP
    decoder: eqx.nn.MLP
    lambdas: dict

    def __init__(
        self,
        d_input,
        d_latent,
        d_hidden,
        n_mlp_layers,
        lambdas,
        *,
        key
    ):
        keys = iter(jax.random.split(key, 100))
        self.encoder = eqx.nn.MLP(
            d_input, d_latent, d_hidden, n_mlp_layers, jax.nn.relu, lambda x: x, key=next(keys)
        )
        self.decoder = eqx.nn.MLP(
            d_latent, d_input, d_hidden, n_mlp_layers, jax.nn.relu, lambda x: x, key=next(keys)
        )
        self.lambdas = lambdas

    def loss(self, model, data, *, key=None):
        x = data
        z = jax.vmap(model.encoder)(x)
        x_hat = jax.vmap(model.decoder)(z)
        losses = {}
        losses['reconstruction'] = einops.reduce((x - x_hat) ** 2, 'n d -> n', 'mean')
        losses['activation_energy'] = einops.reduce(z ** 2, 'n d -> n', 'mean')
        losses['activation_negativity'] = einops.reduce(jnp.maximum(-z, 0), 'n d -> n', 'mean')
        losses['weight_energy'] = model.l2_excluding_biases()

        losses['combined'] = sum(losses[k] * model.lambdas[k] for k in losses.keys())
        losses['sum'] = sum(losses.values())
        log = {f'loss/{k}': v for k, v in losses.items()}
        aux = {
            'x_hat': x_hat,
            'z':     z,
            'log':   log
        }
        return jnp.mean(losses['combined']), aux

    def l2_excluding_biases(self):
        sum_squares = sum(jnp.sum(x ** 2) for x in jtu.tree_leaves(eqx.filter(self, eqx.is_array)) if x.ndim > 1)
        return sum_squares


def evaluate(model, val_sets, config, step, *, key):
    log = {}
    log.update({
        'weight_norm': model.l2_excluding_biases().item()
    })
    for k_val, val_set in val_sets.items():
        auxs = []
        for i, batch in enumerate(val_set):
            key, sub_key = jax.random.split(key)
            _, aux = model.loss(model, batch, key=sub_key)
            aux['s'] = batch
            auxs.append(aux)
        auxs = jax.tree_map(lambda *leaves: jnp.concatenate(leaves) if leaves[0].ndim > 0 else jnp.stack(leaves), *auxs)
        log.update({f'{k}/{k_val}': v.mean().item() for k, v in auxs['log'].items()})

        metrics_ = metrics.compute_metrics(
            np.array(auxs['s']), np.array(auxs['z']), 'discrete', 'continuous', z_noise=3e-2
        )
        log.update({
            f'metrics/{k}/{k_val}': v
            for k, v in metrics_.items() if k in ['cinfom', 'cinfoc', 'infom', 'infoe', 'infoc']
        })

        def plot_mi_heatmap(k):
            mi = metrics_[k]
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(
                mi, ax=ax, annot=True, fmt='.2f', square=True, vmin=0, vmax=1, cbar=False,
                annot_kws={'fontsize': 8},
                xticklabels=[rf'$\mathbf{{z}}_{{{i}}}$' for i in range(mi.shape[1])],
                yticklabels=[rf'$\mathbf{{s}}_{{{i}}}$' for i in range(mi.shape[0])],
                rasterized=True
            )
            for i, label in enumerate(ax.get_xticklabels()):
                if metrics_['z_active'][i] == 0:
                    label.set_color('red')
            fig.tight_layout()
            log.update({f'{k}_heatmap/{k_val}': wandb.Image(fig)})
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
        log.update({f'latent_densities/{k_val}': wandb.Image(fig)})
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
        log.update({f'pairwise_latents/{k_val}': wandb.Image(fig)})
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
        log.update({f'sources_latents/{k_val}': wandb.Image(fig)})
        plt.close()

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

        def three_dimensional_scatterplot(s):
            assert s.shape[1] == 3
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(s[:, 0], s[:, 1], s[:, 2], alpha=0.1)
            ax.set_xlabel('dim 0')
            ax.set_ylabel('dim 1')
            ax.set_zlabel('dim 2')
        three_dimensional_scatterplot(auxs['s'])
        log.update({'sources': wandb.Image(fig)}, step=0)
        plt.close()

    return log


@eqx.filter_jit
def train_step(model, optimizer_state, optimizer, data, *, key):
    (_, aux), grad = eqx.filter_value_and_grad(model.loss, has_aux=True)(model, data, key=key)
    update, optimizer_state = optimizer.update(grad, optimizer_state, model)
    model = eqx.apply_updates(model, update)
    return model, optimizer_state, aux['log']


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
    small_init_with_scale = functools.partial(small_init, scale=config.optim.small_init_scale)
    model = init_linear_weight(model, small_init_with_scale, key=next(keys))

    optimizer = optax.adamw(config.optim.learning_rate, config.optim.weight_decay)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    datasets = np.load(config.data.path)
    train_set = val_set_iid = datasets[config.data.name]
    val_set_independent = datasets['independent']
    val_sets = {'iid': val_set_iid, 'independent': val_set_independent}
    train_set = tf.data.Dataset.from_tensor_slices(train_set)
    train_set = train_set \
        .shuffle(train_set.cardinality(), seed=config.data.seed, reshuffle_each_iteration=True) \
        .repeat() \
        .batch(config.optim.batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    train_set = tfds.as_numpy(train_set)
    for k, v in val_sets.items():
        val_set = tf.data.Dataset.from_tensor_slices(v)
        val_set = val_set \
            .batch(config.optim.batch_size, drop_remainder=False) \
            .prefetch(tf.data.experimental.AUTOTUNE)
        val_sets[k] = tfds.as_numpy(val_set)

    pbar = tqdm.tqdm(train_set, total=int(config.optim.n_steps))
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

            if (step == 0 and not config.debug) or (step + 1) % config.eval.period == 0:
                model = eqx.nn.inference_mode(model, True)
                log = evaluate(model, val_sets, config, step, key=eval_key)
                wandb.log(log, step=step)
                model = eqx.nn.inference_mode(model, False)

            train_key, sub_key = jax.random.split(train_key)
            model, optimizer_state, losses = train_step(model, optimizer_state, optimizer, batch, key=sub_key)
            wandb.log({f'{k}/train': v.mean().item() for k, v in losses.items()}, step=step)
    wandb.finish()
