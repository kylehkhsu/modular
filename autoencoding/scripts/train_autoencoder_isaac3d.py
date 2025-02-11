import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import einops
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

import pathlib
import collections
import ipdb
import tqdm
import omegaconf
import contextlib
import wandb
import hydra
import os
import pprint
import timeit
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import disentangle
import metrics


def load_data(name):
    dataset = np.load('/REDACTED/u/REDACTED/code/modular/isaac3d_subset.npz')
    s = dataset['s']
    x = dataset['x']
    if name == 'independent':
        pass
    else:
        group, value = name.split(':')
        value = float(value)
        if group == 'corner':
            slice = value
            keep = np.sum(s, axis=1) >= slice
            s = s[keep]
            x = x[keep]
        elif group == 'diagonal_remove':
            remove = value
            diagonal = (s[:, 1] == s[:, 2]) & (s[:, 1] <= remove)
            s = s[~diagonal]
            x = x[~diagonal]
        elif group == 'diagonal_strengthen':
            diag = (s[:, 1] == s[:, 2]) & (s[:, 2] == s[:, 4]) & (s[:, 4] == s[:, 5])
            n_add = int(value)
            np.random.seed(42)
            selected_diag = np.random.choice(diag.nonzero()[0], n_add, replace=True)
            s = np.concatenate([s, s[selected_diag]])
            x = np.concatenate([x, x[selected_diag]])
    print(f'Loaded dataset {name} with length {x.shape[0]} and source nmi '
          f'{metrics.multiinformation(s)["normalized_mi"]:.4f}')
    return {
        's': s,
        'x': x,
    }


@eqx.filter_jit
def train_step(model, optimizer_state, optimizer, batch, *, key):
    (_, outs), grads = eqx.filter_value_and_grad(model.batched_loss, has_aux=True)(model, batch, key=key)
    model, optimizer_state = disentangle.utils.optax_step(optimizer, model, grads, optimizer_state)
    return model, optimizer_state, outs['log']


def batched_x_to_image(x):
    x = (x + 1) / 2
    x = einops.rearrange(x, '... c h w -> ... h w c')
    x = jnp.clip(x, 0, 1)
    image = jnp.asarray(255 * x, dtype=jnp.uint8)
    return image


def rows_to_grid_image(rows, num_image_rows, num_image_cols):
    rows = jnp.stack([v for v in rows.values()], axis=0)  # (r, b, c, h, w)
    image = batched_x_to_image(rows)  # (r, b, h, w, c)
    image = einops.rearrange(image, 'r b h w c -> (r h) (b w) c')
    image = einops.rearrange(
        image, 'h (rows cols w) c -> (rows h) (cols w) c', rows=num_image_rows, cols=num_image_cols
    )
    return image


def save(path, model, optimizer_state):
    path.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path / 'model.eqx', model)
    eqx.tree_serialise_leaves(path / 'optimizer_state.eqx', optimizer_state)
    print(f'saved model and optimizer state to {path}')


def evaluate(model, val_sets, config, step, *, key):
    log = {}
    log.update({
        'weight_norm/decoder': disentangle.utils.weight_norm(model.decoder).item(),
        'weight_norm/encoder': disentangle.utils.weight_norm(model.encoder).item(),
    })
    for k_val, val_set in val_sets.items():
        outs_list = []
        for i, batch in tqdm.tqdm(enumerate(val_set)):
            key, sub_key = jax.random.split(key)
            _, outs = model.batched_loss(model, batch, key=sub_key)
            outs['x_pred'] = np.array(jax.nn.sigmoid(outs['decoder']['x_logits']) * 2 - 1)
            outs.pop('decoder')
            outs['log']['metrics/psnr'] = np.array(jax.vmap(disentangle.metrics.peak_signal_to_noise_ratio)(
                batched_x_to_image(outs['x_pred']),
                batched_x_to_image(batch['x'])
            ))
            outs['x'] = batch['x']
            outs['s'] = batch['s']
            outs['z_c'] = np.array(outs['z_c'])
            outs['z_q'] = np.array(outs['z_q'])
            if i > 2:
                for k in [k for k in outs.keys() if k.startswith('x')]:
                    outs[k] = np.zeros((0, *outs[k].shape[1:]), dtype=outs[k].dtype)
            outs_list.append(outs)

        with jax.default_device(jax.devices('cpu')[0]):
            outs = jax.tree_map(lambda *leaves: jnp.concatenate(leaves) if leaves[0].ndim > 0 else jnp.stack(leaves), *outs_list)
        log.update({f'{k}/val': v.mean().item() for k, v in outs['log'].items()})

        if model.quantize_latents:
            z = outs['z_q']
        else:
            z = outs['z_c']
        s = outs['s']
        z = np.array(z)
        s = np.array(s)

        start = timeit.default_timer()
        metrics_ = metrics.compute_metrics(s, z, 'discrete', 'discrete')
        print(f'metrics: {timeit.default_timer() - start:.1f} s')
        log.update({f'metrics/{k}/{k_val}': v for k, v in metrics_.items()
                    if k in ['cinfom', 'cinfoc', 'infom', 'infoe', 'infoc']})

        def plot_mi_heatmap(results, k):
            mi = results[k]
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(
                mi, ax=ax, annot=True, fmt='.2f', square=True, vmin=0, vmax=1, cbar=False,
                annot_kws={'fontsize': 8},
                xticklabels=[rf'$\mathbf{{z}}_{{{i}}}$' for i in range(mi.shape[1])],
                yticklabels=[rf'$\mathbf{{s}}_{{{i}}}$' for i in range(mi.shape[0])],
                rasterized=True
            )
            for i, label in enumerate(ax.get_xticklabels()):
                if results['z_active'][i] == 0:
                    label.set_color('red')
            fig.tight_layout()
            log.update({f'{k}_heatmap/{k_val}': wandb.Image(fig)})
            plt.close()
        plot_mi_heatmap(metrics_, 'nmi')
        plot_mi_heatmap(metrics_, 'ncmi')

        # start = timeit.default_timer()
        # linear_metrics = metrics.compute_linear_metrics(s, z, 'discrete', 'discrete')
        # print(f'linear_metrics: {timeit.default_timer() - start:.1f} s')
        # log.update({f'metrics/{k}/{k_val}': v for k, v in linear_metrics.items()
        #             if k in ['linear_cinfom', 'linear_cinfoc', 'linear_infom', 'linear_infoc']})
        # plot_mi_heatmap(linear_metrics, 'linear_nmi')
        # plot_mi_heatmap(linear_metrics, 'linear_ncmi')

        nz = z.shape[1]
        ns = s.shape[1]

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

        start = timeit.default_timer()
        indices = jax.random.randint(key, (1000,), 0, s.shape[0])
        fig, axes = plt.subplots(ns, nz, figsize=(2 * nz, 2 * ns))
        for i in range(ns):
            for j in range(nz):
                ax = axes[i][j]
                sns.histplot(
                    ax=ax,
                    x=z[indices, j],
                    y=s[indices, i],
                    rasterized=True
                )
                ax.set_xlabel(rf'$z_{{{j}}}$')
                ax.set_ylabel(rf'$s_{{{i}}}$')
        fig.tight_layout()
        log.update({f'sources_latents/{k_val}': wandb.Image(fig)})
        plt.close()
        print(f'sources_latents: {timeit.default_timer() - start:.1f} s')

        start = timeit.default_timer()
        # val generations
        num_samples = config.eval.num_vis_rows.val * config.eval.num_vis_cols
        rows = {k: outs[k][:num_samples] for k in ['x', 'x_pred']}
        rows['x_diff'] = jnp.abs(rows['x'] - rows['x_pred']) - 1
        log.update({f'vis/{k_val}': wandb.Image(np.array(
            rows_to_grid_image(rows, config.eval.num_vis_rows.val, config.eval.num_vis_cols)
        ))})
        print(f'val_generations: {timeit.default_timer() - start:.1f} s')

        # latent densities
        start = timeit.default_timer()
        data = pd.DataFrame(z, columns=[f'z{i}' for i in range(z.shape[1])])
        data['id'] = data.index
        data = data.melt(id_vars='id', var_name='z', value_name='value')
        fig, ax = plt.subplots(figsize=(z.shape[1] ** 0.8, 3))
        sns.violinplot(data=data, ax=ax, x='z', y='value', density_norm='width', cut=0)
        fig.tight_layout()
        log.update({f'latent_densities/{k_val}': wandb.Image(fig)})
        plt.close()
        print(f'latent_densities: {timeit.default_timer() - start:.1f} s')

        if k_val == 'iid':
            # decoded latent interventions
            start = timeit.default_timer()
            z_max = z.max(axis=0)
            z_min = z.min(axis=0)
            z_sample = z[:config.eval.num_intervene_cols]    # (b, z)
            for i_latent in range(z.shape[1]):
                values = jnp.linspace(z_min[i_latent], z_max[i_latent], config.eval.num_intervene_values)
                z_intervene = einops.repeat(jnp.copy(z_sample), 'b z -> b v z', v=config.eval.num_intervene_values)
                z_intervene = z_intervene.at[:, :, i_latent].set(values)
                x_intervene_logits = jax.vmap(jax.vmap(model.decoder))(z_intervene)['x_logits']
                x_intervene = jax.nn.sigmoid(x_intervene_logits) * 2 - 1  # (b, v, c, h, w)
                image = batched_x_to_image(x_intervene)
                image = einops.rearrange(image, 'b v h w c -> (b h) (v w) c')
                log.update({f'decode_latent_interventions/{i_latent}': wandb.Image(np.array(image))})
            print(f'latent_interventions: {timeit.default_timer() - start:.1f} s')

    if step == 0:
        ds = s.shape[1]
        fig, axes = plt.subplots(ds, ds, figsize=(2 * ds, 2 * ds))
        for i in range(ds):
            for j in range(ds):
                ax = axes[i][j]
                sns.histplot(
                    ax=ax,
                    x=s[:, j],
                    y=s[:, i],
                    rasterized=True,
                )
                ax.set_xlabel(rf'$s_{{{j}}}$')
                ax.set_ylabel(rf'$s_{{{i}}}$')
        fig.tight_layout()
        log.update({'pairwise_sources': wandb.Image(fig)}, step=0)
        plt.close()

    return log


def prepare_data(data):
    x = data['x']
    x = einops.rearrange(x, 'h w c -> c h w')
    x = tf.cast(x, tf.float32) / 255.
    x = x * 2 - 1
    data['x'] = x
    return data

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
    train_set = val_set_iid = load_data(config.data.name)
    val_set_independent = load_data('independent')
    config.model.latent_size = 2 * train_set['s'].shape[1]
    wandb.config.update({'model.latent_size': config.model.latent_size})
    wandb.log({'source_normalized_multiinformation': metrics.multiinformation(train_set['s'])['normalized_mi']}, step=0)
    val_sets = {'iid': val_set_iid, 'independent': val_set_independent}
    train_set = tf.data.Dataset.from_tensor_slices(train_set)
    train_set = train_set \
        .shuffle(train_set.cardinality(), seed=config.data.seed, reshuffle_each_iteration=True) \
        .repeat() \
        .map(prepare_data, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(config.optim.batch_size) \
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    train_set = tfds.as_numpy(train_set)
    for k, v in val_sets.items():
        val_set = tf.data.Dataset.from_tensor_slices(v)
        val_set = val_set \
            .map(prepare_data, num_parallel_calls=tf.data.AUTOTUNE) \
            .batch(config.optim.batch_size, drop_remainder=False) \
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        val_sets[k] = tfds.as_numpy(val_set)

    model = hydra.utils.instantiate(config.model)(key=next(keys))
    optimizer, optimizer_state = model.construct_optimizer(config)

    pbar = tqdm.tqdm(train_set, total=int(config.optim.num_steps))
    context = jax.disable_jit if config.debug else contextlib.nullcontext
    train_key = next(keys)
    eval_key = next(keys)
    with context():
        for step, batch in enumerate(pbar):
            if step >= config.optim.num_steps:
                break

            if (step + 1) % config.checkpoint.period == 0:
                path = checkpoints_path / f'step={step}'
                save(path, model, optimizer_state)
                wandb.save(str(path / '*'), base_path=run.dir)

            if (step == 0 and not config.debug) or \
                (step + 1) % config.eval.period == 0:
                # ((step + 1 < config.eval.period) and (step + 1) % (config.eval.period // 5) == 0):
                # model = eqx.nn.inference_mode(model, True)
                log = evaluate(model, val_sets, config, step,  key=eval_key)
                wandb.log(log, step=step)
                # model = eqx.nn.inference_mode(model, False)

            train_key, sub_key = jax.random.split(train_key)
            model, optimizer_state, log = train_step(model, optimizer_state, optimizer, batch, key=sub_key)
            wandb.log({f'{k}/train': v.mean().item() for k, v in log.items()}, step=step)
    wandb.finish()
