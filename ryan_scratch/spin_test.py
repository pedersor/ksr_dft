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

from jax_dft.neural_xc import global_functional_sigma
from jax_dft import spin_scf


# Set the default dtype as float64
config.update('jax_enable_x64', True)



if __name__ == '__main__':
  import matplotlib.pyplot as plt
  import sys

  h = 0.08
  grids = np.arange(-256, 257) * h


  network = neural_xc.build_sliding_net(
    window_size=1,
    num_filters_list=[16, 16, 16],
    activation='swish')


  init_fn, neural_xc_energy_density_fn = global_functional_sigma(
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

  fn = tree_util.Partial(neural_xc_energy_density_fn, params=init_params)

  xc_energy_density = fn(example_density, spin_density=example_spin_density)



  plt.plot(grids, example_density)
  plt.plot(grids, xc_energy_density)

  plt.savefig('spin_test.pdf')
