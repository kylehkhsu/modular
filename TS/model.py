import jax
import jax.numpy as jnp
import numpy as np
import einops
import utils
import equinox as eqx

from typing import Optional

LAMBDAS = {
    'target': 100,
    'transition': 1000,
    'activation_energy': 1,
    'activation_positivity': 100,
    'readout_energy': 1,
    'transition_energy': 1
}

SHOOTING_LAMBDAS = {
    'target': 100,
    'transition': 1,
    'activation_energy': 1,
    'activation_positivity': 100,
    'readout_energy': 1,
    'transition_energy': 1
}


class CollocationModel(eqx.Module):
    T: int
    D: int
    K: int
    key: jnp.ndarray
    lambdas: dict
    x: jnp.ndarray = Optional[jnp.ndarray]

    def __post_init__(self):
        keys = iter(jax.random.split(self.key, 100))
        if self.x is None:
            self.x = jax.random.uniform(next(keys), (self.T, self.D * self.K),
                                        minval=0,
                                        maxval=1)

    def loss(self, model, y, *, key=None):
        # compute best-fit transition matrix for x(t+1) = W x(t) + b
        x = model.x
        x_1 = jnp.concatenate([x, jnp.ones((self.T, 1))], axis=1)
        W = utils.least_squares(x_1[:-1], x[1:])
        # compute best-fit readout matrix for y(t) = R x(t) + d
        R = utils.least_squares(x_1, y)
        losses = {}
        y_pred = x_1 @ R
        x_next = x_1 @ W
        losses['target'] = einops.reduce((y - y_pred)**2, 't d -> ', 'mean')
        losses['transition'] = einops.reduce((x_next[:-1] - x[1:])**2,
                                             't d -> ', 'mean')
        losses['activation_energy'] = einops.reduce(x**2, 't d -> ', 'mean')
        losses['activation_positivity'] = einops.reduce(
            (0.5 * (x - jnp.abs(x)))**2, 't d -> ', 'mean')
        losses['readout_energy'] = einops.reduce(R**2, 'd f -> ', 'mean')
        losses['transition_energy'] = einops.reduce(W**2, 'd1 d2 -> ', 'mean')

        loss = sum(self.lambdas[k] * v for k, v in losses.items())
        aux = {
            'losses': losses,
            'y_pred': y_pred,
            'x_next': x_next,
            'W': W,
            'R': R
        }
        return loss, aux


class ShootingModel(eqx.Module):
    K: int
    T: int
    D: int
    key: jnp.ndarray
    lambdas: dict

    x0: Optional[jnp.ndarray] = None
    W: Optional[jnp.ndarray] = None
    bw: Optional[jnp.ndarray] = None
    R: Optional[jnp.ndarray] = None
    br: Optional[jnp.ndarray] = None

    sigma: float = 0.01

    def __post_init__(self):

        if self.x0 is None:
            self.x0 = self.sigma * jax.random.normal(self.key,
                                                     (self.K * self.D, ))
        if self.W is None:
            self.W = self.sigma * jax.random.normal(
                self.key, (self.K * self.D, self.K * self.D))
        if self.R is None:
            self.R = self.sigma * jax.random.normal(self.key,
                                                    (self.K * self.D, self.K))
        if self.bw is None:
            self.bw = self.sigma * jax.random.normal(self.key,
                                                     (self.K * self.D, ))
        if self.br is None:
            self.br = self.sigma * jax.random.normal(self.key, (self.K, ))

    def forward(self, x0):

        def f(x, dummy):
            x_next = einops.einsum(x, self.W, 'I, I W  -> W') + self.bw
            return x_next, x_next

        dummy_x = jnp.empty((self.T - 1, self.K, 0))
        _, xs = jax.lax.scan(f, x0, dummy_x)
        x = jnp.concatenate([self.x0[None, :], xs], axis=0)
        y = einops.einsum(x, self.R, 'T I, I O -> T O') + self.br

        return x, y

    def loss(self, model, y_target, key=None):

        x0, W, bw, R, br = model.x0, model.W, model.bw, model.R, model.br

        def f(x, dummy):
            x_next = einops.einsum(x, W, 'I, I W  -> W') + bw
            return x_next, x_next

        dummy_x = jnp.empty((self.T - 1, 0))
        _, xs = jax.lax.scan(f, x0, dummy_x)
        x = jnp.concatenate([x0[None, :], xs], axis=0)
        y_pred = einops.einsum(x, R, 'T I, I O -> T O') + br.T

        losses = {}
        losses['target'] = einops.reduce((y_target - y_pred)**2, 't d -> ',
                                         'mean')
        losses['transition'] = einops.reduce((x[:-1] - x[1:])**2, 't d -> ',
                                             'mean')
        losses['activation_energy'] = einops.reduce(x**2, 't d -> ', 'mean')
        losses['activation_positivity'] = einops.reduce(
            (0.5 * (x - jnp.abs(x)))**2, 't d -> ', 'mean')
        losses['readout_energy'] = einops.reduce(R**2, 'd f -> ', 'mean')
        losses['transition_energy'] = einops.reduce(W**2, 'd1 d2 -> ', 'mean')

        loss = sum(self.lambdas[k] * v for k, v in losses.items())
        aux = {'losses': losses, 'y_pred': y_pred, 'x': x, 'W': W, 'R': R}
        return loss, aux


class ShootingMultitaskModel(ShootingModel):

    K: int
    T: int
    D: int
    key: jnp.ndarray
    lambdas: dict

    x0: Optional[jnp.ndarray] = None
    W: Optional[jnp.ndarray] = None
    bw: Optional[jnp.ndarray] = None
    R: Optional[jnp.ndarray] = None
    br: Optional[jnp.ndarray] = None

    sigma: float = 0.01

    def __post_init__(self):

        if self.x0 is None:
            self.x0 = self.sigma * jax.random.normal(self.key,
                                                     (self.K * self.D, ))
        if self.W is None:
            self.W = self.sigma * jax.random.normal(
                self.key, (self.K * self.D, self.K * self.D))
        if self.R is None:
            self.R = self.sigma * jax.random.normal(self.key,
                                                    (self.K * self.D, self.K))
        if self.bw is None:
            self.bw = self.sigma * jax.random.normal(self.key,
                                                     (self.K * self.D, ))
        if self.br is None:
            self.br = self.sigma * jax.random.normal(self.key, (self.K, ))

    def forward(self, model, x0):
        W, bw, R, br = model.W, model.bw, model.R, model.br

        def f(x, dummy):
            x_next = einops.einsum(x, W, 'I, I W  -> W') + bw
            return x_next, x_next

        dummy_x = jnp.empty((self.T - 1, 0))
        _, xs = jax.lax.scan(f, x0, dummy_x)
        x = jnp.concatenate([x0[None, :], xs], axis=0)
        y_pred = einops.einsum(x, R, 'T I, I O -> T O') + br.T

        return x, y_pred

    def loss(self, model, y_target, key, u_index, T_pulse):

        x0, W, bw, R, br = model.x0, model.W, model.bw, model.R, model.br

        u = jnp.zeros(shape=(self.K * self.D, ))
        u = u.at[jnp.array(u_index)].set(1.) * self.sigma

        def f(x, dummy):
            x_next = einops.einsum(x, W, 'I, I W  -> W') + bw
            return x_next, x_next

        ## First calculate x when there is input
        x_input = jnp.empty((T_pulse, self.K * self.D))
        x_next = x0
        for t in range(T_pulse):
            x_next = einops.einsum(x_next, W, 'I, I W  -> W') + bw + u
            x_input = x_input.at[t].set(x_next)
        #x_input = jnp.concatenate([x0[None, :], x_input], axis=0)
        ## and do rest;

        dummy_x = jnp.empty((self.T - T_pulse, 0))
        _, xs = jax.lax.scan(f, x_input[-1], dummy_x)
        x = jnp.concatenate([x_input, xs], axis=0)
        y_pred = einops.einsum(x, R, 'T I, I O -> T O') + br.T

        losses = {}
        losses['target'] = einops.reduce((y_target - y_pred)**2, 't d -> ',
                                         'mean')
        losses['transition'] = einops.reduce((x[:-1] - x[1:])**2, 't d -> ',
                                             'mean')
        losses['activation_energy'] = einops.reduce(x**2, 't d -> ', 'mean')
        losses['activation_positivity'] = einops.reduce(
            (0.5 * (x - jnp.abs(x)))**2, 't d -> ', 'mean')
        losses['readout_energy'] = einops.reduce(R**2, 'd f -> ', 'mean')
        losses['transition_energy'] = einops.reduce(W**2, 'd1 d2 -> ', 'mean')

        loss = sum(self.lambdas[k] * v for k, v in losses.items())
        aux = {'losses': losses, 'y_pred': y_pred, 'x': x, 'W': W, 'R': R}
        return loss, aux


@eqx.filter_jit
def train_step(model, optimizer_state, optimizer, y, key, **kwargs):

    (_, aux), grad = eqx.filter_value_and_grad(model.loss,
                                               has_aux=True)(model, y, key,
                                                             **kwargs)
    update, optimizer_state = optimizer.update(grad, optimizer_state, model)
    model = eqx.apply_updates(model, update)
    return model, optimizer_state, aux
