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

# Set the default dtype as float64
config.update('jax_enable_x64', True)

if __name__ == '__main__':
  h = 0.08
  grids = np.arange(-256, 257) * h




  network = neural_xc.build_sliding_net(
    window_size=1,
    num_filters_list=[16, 16, 16],
    activation='swish')

  network_init_fn, network_apply_fn = network
  num_grids = grids.shape[0]


  def new_init_fn(rng):
    _, params = network_init_fn(rng=rng, input_shape=(-1, num_grids, 1))
    return params


  key = jax.random.PRNGKey(0)

  init_params = new_init_fn(key)
  spec, flatten_init_params = np_utils.flatten(init_params)
  print(f'number of parameters = {len(flatten_init_params)}')
