import os
import sys
from functools import partial
import glob
import pickle
import time

import numpy as np
import scipy
import jax
from jax import random
from jax import tree_util
from jax.config import config
import jax.numpy as jnp
from jax_dft import datasets
from jax_dft import jit_scf
from jax_dft import losses
from jax_dft import neural_xc
from jax_dft import np_utils
from jax_dft import scf
from jax_dft import utils
from jax_dft import xc

from jax_dft.neural_xc import _check_network_output, _is_power_of_two
from jax_dft import spin_scf

# Set the default dtype as float64
config.update('jax_enable_x64', True)


def global_functional(network, grids, num_spatial_shift=1):
  """Functional with global density information parameterized by neural network.

  This function takes the entire density as input and outputs the entire xc
  energy density.

  The network used in this network is a convolution neural network. This
  function will expand the batch dimension and channel dimension of the input
  density to fit the input shape of the network.

  There are two types of mapping can be applied depending on the architecture
  of the network.

  * many-to-one:
    This function is inspired by

    Schmidt, Jonathan, Carlos L. Benavides-Riveros, and Miguel AL Marques.
    "Machine Learning the Physical Nonlocal Exchange–Correlation Functional of
    Density-Functional Theory."
    The journal of physical chemistry letters 10 (2019): 6425-6431.

    The XC energy density at index x is determined by the density in
    [x - radius, x + radius].

    Note for radius=0, only the density at one point is used to predict the
    xc energy density at the same point. Thus it is equivalent to LDA.

    For radius=1, the density at one point and its nearest neighbors are used to
    predict the xc energy density at this point. It uses same information used
    by GGA, where the gradient of density is computed by the finite difference.

    For large radius, it can be considered as a non-local functional.

    Applying MLP on 1d array as a sliding window is not accelerator efficient.
    Instead, same operations can be performed by using 1d convolution with
    filter size 2 * radius + 1 as the first layer, and 1d convolution with
    filter size 1 for the rest of the layers. The channel dimension in the
    rest of the layers acts as the hidden nodes in MLP.

  * many-to-many:
    The XC energy density at index x is determined by the entire density. This
    mapping can be parameterized by a U-net structure to capture both the low
    level and high level features from the input.

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

  if not _is_power_of_two(num_grids - 1):
    raise ValueError(
        'The num_grids must be power of two plus one for global functional '
        'but got %d' % num_grids)

  def init_fn(rng):
    _, params = network_init_fn(rng=rng, input_shape=(-1, num_grids, 2))
    return params

  #TODO: jit...
  #@jax.jit
  def xc_energy_density_fn(density, spin_density, params):
    """Gets xc energy density.

    Args:
      density: Float numpy array with shape (num_grids,).
      params: Parameters of the network.

    Returns:
      Float numpy array with shape (num_grids,).
    """
    # Expand batch dimension and channel dimension. We use batch_size=1 here.
    # (1, num_grids, 1)
    density = density[jnp.newaxis, :, jnp.newaxis]
    spin_density = spin_density[jnp.newaxis, :, jnp.newaxis]

    input_features = jnp.append(density, spin_density, axis=2)

    #TODO: num_spatial_shift...

    # If the network use convolution layer, the backend function
    # conv_general_dilated requires float32.
    input_features = input_features.astype(jnp.float32)
    params = tree_util.tree_map(lambda x: x.astype(jnp.float32), params)

    output = network_apply_fn(params, input_features)

    # Remove the channel dimension.
    # (num_spatial_shift, num_grids)
    output = jnp.squeeze(output, axis=2)
    _check_network_output(output, num_grids)

    #TODO: num_spatial_shift...

    output = jnp.mean(output, axis=0).astype(jnp.float64)

    return output

  return init_fn, xc_energy_density_fn


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  import sys

  h = 0.08
  grids = np.arange(-256, 257) * h


  network = neural_xc.build_sliding_net(
    window_size=1,
    num_filters_list=[16, 16, 16],
    activation='swish')


  init_fn, neural_xc_energy_density_fn = global_functional(
    network, grids=grids)

  key=jax.random.PRNGKey(0)
  init_params = init_fn(key)
  spec, flatten_init_params = np_utils.flatten(init_params)
  print(f'number of parameters = {len(flatten_init_params)}')


  ''' H atom example 
  example_density = np.load('../data/ions/dmrg/basic_all/densities.npy')[0]
  example_density_up = example_density
  example_density_down = 0. * example_density
  example_spin_density = example_density_up - example_density_down
  '''

  ''' He atom example '''
  example_density = np.load('../data/ions/lsda/basic_all/densities.npy')[4]
  example_density_up = example_density / 2.
  example_density_down = example_density / 2.
  example_spin_density = example_density_up - example_density_down

  with open('../models/ions/ksr_lda/unpol_lda/optimal_ckpt.pkl', 'rb') as handle:
    example_params = pickle.load(handle)



  xc_energy_density = neural_xc_energy_density_fn(example_density,
                                                  example_spin_density,
                                                  init_params)



  plt.plot(grids, example_density)
  plt.plot(grids, xc_energy_density)


  xc_potential_up = spin_scf.get_xc_potential_up(
    example_density_up, example_density_down,
    tree_util.Partial(neural_xc_energy_density_fn,params=init_params), grids)

  xc_potential_down = spin_scf.get_xc_potential_down(
    example_density_up, example_density_down,
    tree_util.Partial(neural_xc_energy_density_fn, params=init_params), grids)

  plt.plot(grids, xc_potential_up)
  plt.plot(grids, xc_potential_down + 0.001)


  plt.savefig('spin_test2.pdf')
