import jax
import jax.numpy as jnp
import einops


def least_squares(A: jnp.ndarray, B: jnp.ndarray):
    ## solve X for AX=B
    return jnp.linalg.pinv(A) @ B
