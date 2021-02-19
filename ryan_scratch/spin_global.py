import jax
import jax.numpy as jnp
import numpy as np

from jax_dft import neural_xc

grids = np.arange(-10, 11)
density = np.ones(len(grids))

init_fun, apply_fun = neural_xc.global_conv_block_sigma(
  num_channels=8,
  grids=grids,
  minval=0.1,
  maxval=2.385345,
  downsample_factor=0
)


rng = jax.random.PRNGKey(0)


_, init_params = init_fun(rng, input_shape=(-1, len(grids), 1))


density_up = density[jnp.newaxis, :, jnp.newaxis]

density_down = density[jnp.newaxis, :, jnp.newaxis]

input_features = jnp.append(density_up, density_down, axis=2)



input_features = input_features.astype(jnp.float32)

out = apply_fun(init_params, input_features)
