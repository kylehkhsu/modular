import jax
import jax.numpy as jnp
import optax
import einops

def binary_cross_entropy(x_pred_logits, x_true_probs):
    assert x_pred_logits.ndim == x_true_probs.ndim == 3
    return jnp.mean(
        optax.sigmoid_binary_cross_entropy(
            logits=x_pred_logits,
            labels=x_true_probs,
        )
    )


def mean_squared_error(x_pred, x_true):
    assert x_pred.ndim == x_true.ndim == 3
    return jnp.mean(jnp.square(x_pred - x_true))


