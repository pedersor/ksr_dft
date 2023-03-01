"""xc functional parameterized by neural network."""

import functools

import jax
from jax import lax
from jax import nn
from jax import tree_util
from jax.experimental import stax
import jax.numpy as jnp
from jax.scipy import ndimage

from ksr_dft import scf
from ksr_dft import utils
from ksr_dft import neural_xc


def gga_functional(network, grids, num_spatial_shift=1):
  """GGA take on neural functional.


  Args:
    network: an (init_fn, apply_fn) pair.
     * init_fn: The init_fn of the neural network. It takes an rng key and
         an input shape and returns an (output_shape, params) pair.
     * apply_fn: The apply_fn of the neural network. It takes params,
         inputs, and an rng key and applies the layer.
    grids: Float numpy array with shape (num_grids,).
        num_grids must be 2 ** k + 1, where k is an non-zero integer.
    num_spatial_shift: Integer, the number of spatial shift (include the
        original input).

  Returns:
    init_fn: A function takes an rng key and returns initial params.
    xc_energy_density_fn: A function takes density (1d array) and params,
        returns xc energy density with the same shape of density.

  Raises:
    ValueError: If num_spatial_shift is less than 1
        or the num_grids is not 2 ** k + 1.
  """
  if num_spatial_shift < 1:
    raise ValueError(
        f'num_spatial_shift can not be less than 1 but got {num_spatial_shift}')

  network_init_fn, network_apply_fn = network
  num_grids = grids.shape[0]

  if not neural_xc._is_power_of_two(num_grids - 1):
    raise ValueError(
        'The num_grids must be power of two plus one for global functional '
        'but got %d' % num_grids)

  def init_fn(rng):
    _, params = network_init_fn(rng=rng, input_shape=(-1, num_grids, 2))
    return params

  @jax.jit
  def xc_energy_density_fn(density, params):
    """Gets xc energy density.

    Args:
      density: Float numpy array with shape (num_grids,).
      params: Parameters of the network.

    Returns:
      Float numpy array with shape (num_grids,).
    """

    density_grad = jnp.abs(jnp.gradient(density, utils.get_dx(grids)))

    # Expand batch dimension and channel dimension. We use batch_size=1 here.
    # (1, num_grids, 1)
    density = density[jnp.newaxis, :, jnp.newaxis]
    density_grad = density_grad[jnp.newaxis, :, jnp.newaxis]

    input_features = jnp.stack([density, density_grad], axis=2)
    input_features = jnp.squeeze(input_features, axis=3)

    #TODO?: num_spatial_shift...

    # If the network use convolution layer, the backend function
    # conv_general_dilated requires float32.
    input_features = input_features.astype(jnp.float32)
    params = tree_util.tree_map(lambda x: x.astype(jnp.float32), params)
    output = network_apply_fn(params, input_features)

    # Remove the channel dimension.
    # (num_spatial_shift, num_grids)
    output = jnp.squeeze(output, axis=2)
    neural_xc._check_network_output(output, num_grids)

    #TODO?: num_spatial_shift...

    output = jnp.mean(output, axis=0).astype(jnp.float64)

    return output

  return init_fn, xc_energy_density_fn
