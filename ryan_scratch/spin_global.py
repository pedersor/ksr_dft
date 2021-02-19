import jax
import jax.numpy as jnp
import numpy as np

from jax_dft import neural_xc
from jax_dft import np_utils

grids = np.arange(-8, 9)
density = np.ones(len(grids))

init_fun, apply_fun = neural_xc.global_conv_block_sigma(
  num_channels=8,
  grids=grids,
  minval=0.1,
  maxval=2.385345,
  downsample_factor=0
)


rng = jax.random.PRNGKey(0)


_, init_params = init_fun(rng, input_shape=(-1, len(grids), 2))


density_up = density[jnp.newaxis, :, jnp.newaxis]

density_down = 2*density[jnp.newaxis, :, jnp.newaxis]

input_features = jnp.append(density_up, density_down, axis=2)

input_features = input_features.astype(jnp.float32)

out = apply_fun(init_params, input_features)



'''
# global
network = neural_xc.build_global_local_conv_net_sigma(
    num_global_filters=8,
    num_local_filters=16,
    num_local_conv_layers=2,
    activation='swish',
    grids=grids,
    minval=0.1,
    maxval=2.385345,
    downsample_factor=0)
init_fn, neural_xc_energy_density_fn = neural_xc.global_functional_sigma(
    network, grids=grids)



init_params = init_fn(rng)
spec, flatten_init_params = np_utils.flatten(init_params)

print(len(flatten_init_params))

neural_xc_energy_density_fn(density, init_params)
'''


# global
network = neural_xc.build_global_local_conv_net(
    num_global_filters=16,
    num_local_filters=16,
    num_local_conv_layers=2,
    activation='swish',
    grids=grids,
    minval=0.1,
    maxval=2.385345,
    downsample_factor=0)
init_fn, neural_xc_energy_density_fn = neural_xc.global_functional(
    network, grids=grids)



init_params = init_fn(rng)
spec, flatten_init_params = np_utils.flatten(init_params)

print(len(flatten_init_params))

neural_xc_energy_density_fn(density, init_params)
