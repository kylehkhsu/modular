import ipdb
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import einops
import numpy as np
from typing import *

import disentangle.nn
import disentangle.losses
import disentangle.utils


class Autoencoder(eqx.Module):
    encoder: eqx.Module
    latent: eqx.Module
    decoder: eqx.Module
    lambdas: Dict[str, float]
    reconstruction_loss_fn: str
    regularized_attributes: List[str]
    quantize_latents: bool

    def __init__(
        self,
        encoder,
        decoder,
        latent_size,
        lambdas,
        reconstruction_loss_fn,
        regularized_attributes,
        quantize_latents,
        num_quantized_values,
        *,
        key
    ):
        keys = iter(jax.random.split(key, 100))
        self.encoder = encoder(latent_size=latent_size, key=next(keys))
        if quantize_latents:
            latent = disentangle.nn.Quantizer(num_latents=latent_size, num_values_per_latent=num_quantized_values,
                                              key=next(keys))
        else:
            latent = eqx.nn.Lambda(lambda z: {'z_c': z})
        self.latent = latent
        self.decoder = decoder(latent_size=latent_size, key=next(keys))
        self.lambdas = lambdas
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.regularized_attributes = regularized_attributes
        self.quantize_latents = quantize_latents

    @eqx.filter_jit
    def batched_loss(self, model, batch, *, key):
        outs = {}
        outs.update(jax.vmap(model.latent)(jax.vmap(model.encoder)(batch['x'])))

        if model.quantize_latents:
            z = outs['z_q']
        else:
            z = outs['z_c']

        outs['decoder'] = jax.vmap(model.decoder)(z)
        losses = {}

        if model.reconstruction_loss_fn == 'binary_cross_entropy':
            losses['reconstruct'] = jax.vmap(disentangle.losses.binary_cross_entropy)(
                x_pred_logits=outs['decoder']['x_logits'],
                x_true_probs=(batch['x'] + 1) / 2
            )
        else:
            raise ValueError(f'Unknown reconstruction loss function: {model.reconstruction_loss_fn}')

        losses['total'] = sum(model.lambdas[k] * losses[k] for k in model.lambdas.keys())
        outs['log'] = {f'loss/{k}': v for k, v in losses.items()}

        return jnp.mean(losses['total']), outs

    def construct_optimizer(self, config):
        weight_decay = config.optim.weight_decay
        optimizer = optax.multi_transform(
            {
                'regularized': optax.chain(
                    optax.clip(config.optim.clip),
                    optax.adamw(
                        learning_rate=config.optim.learning_rate,
                        weight_decay=weight_decay
                    )
                ),
                'unregularized': optax.chain(
                    optax.clip(config.optim.clip),
                    optax.adamw(
                        learning_rate=config.optim.learning_rate,
                        weight_decay=0.0
                    )
                )
            },
            param_labels=disentangle.utils.optax_wrap(self.param_labels())
        )
        optimizer_state = optimizer.init(disentangle.utils.optax_wrap(self.filter()))
        return optimizer, optimizer_state

    def param_labels(self):
        param_labels = jax.tree_map(lambda _: 'unregularized', self.filter())
        for attr in self.regularized_attributes:
            param_labels = disentangle.utils.relabel_attr(param_labels, attr, 'regularized')
        print(f'param_labels: {param_labels}')
        return param_labels

    def filter(self, x=None):
        if x is None:
            x = self
        x = eqx.filter(x, eqx.is_array)
        return x

