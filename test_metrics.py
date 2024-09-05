import ipdb

import metrics
from metrics import compute_linear_metrics, compute_metrics
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import omegaconf
import pathlib
import hydra
import jax
import jax.numpy as jnp
import equinox as eqx
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
from scripts.train_toy import MLPAutoencoder, evaluate, generate_square_data
from scripts.train_autoencoder import batched_x_to_image
import disentangle
sns.set_theme()
api = wandb.Api()
run_path = 'iris_viscam/modular/81k5w7gi'
step = 9999
run = api.run(run_path)
config = omegaconf.OmegaConf.create(run.config)

dataset_info, train_set, val_set = disentangle.datasets.load(config)
config.model.latent_size = 2 * dataset_info['num_sources']
keys = iter(jax.random.split(jax.random.PRNGKey(config.experiment.seed), 100))
model = hydra.utils.instantiate(config.model)(key=next(keys))
model_file = wandb.restore(f'checkpoints/step={step}/model.eqx', run_path=run_path)
model = eqx.tree_deserialise_leaves(model_file.name, model)

key = next(keys)
outs_list = []
for i, batch in tqdm.tqdm(enumerate(val_set), total=config.data.num_val // config.data.batch_size):
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
    outs = jax.tree.map(lambda *leaves: jnp.concatenate(leaves) if leaves[0].ndim > 0 else jnp.stack(leaves),
                        *outs_list)

    if model.quantize_latents:
        z = outs['z_q']
    else:
        z = outs['z_c']
    s = outs['s']

metrics = compute_metrics(s, z, 'discrete', 'discrete')
linear_metrics = compute_linear_metrics(s, z, 'discrete', 'discrete')
ipdb.set_trace()