import glob
import pickle
import time
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
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys
from functools import partial
import os
from train_atoms import Train_atoms

config.update('jax_enable_x64', True)


class Test_atoms(Train_atoms):
  def __init__(self, datasets_base_dir):

    super(Test_atoms, self).__init__(datasets_base_dir)

  def set_validation_set(self, selected_ions):
    """Sets the validation set from a list of ions."""
    self.validation_set = self.complete_dataset.get_atoms(
      selected_ions=selected_ions)
    # obtain initial densities
    initial_densities = scf.get_initial_density(self.validation_set,
                                                method='noninteracting')
    self.validation_set = self.validation_set._replace(
      initial_densities=initial_densities)
    return self

  def _get_states(self, ckpt_path):
    print(f'Load {ckpt_path}')
    with open(ckpt_path, 'rb') as handle:
      params = pickle.load(handle)
    states = []
    for i in range(len(self.validation_set.locations)):
      states.append(self.kohn_sham(
        params,
        locations=self.validation_set.locations[i],
        nuclear_charges=self.validation_set.nuclear_charges[i],
        initial_densities=self.validation_set.initial_densities[i]))
    return tree_util.tree_multimap(lambda *x: jnp.stack(x), *states)

  def get_optimal_ckpt(self, path_to_ckpts):
    ckpt_list = sorted(
      glob.glob(os.path.join(path_to_ckpts, 'ckpt-?????')))

    states = []
    for ckpt_path in ckpt_list:
        states.append(self._get_states(ckpt_path))

    return

if __name__ == '__main__':
  two_electrons = Test_atoms('../data/ions/num_electrons_2')
  two_electrons.get_complete_dataset(num_grids=513)

  # set validation set
  to_validate = [3]
  two_electrons.set_validation_set(selected_ions=to_validate)

  two_electrons.get_optimal_ckpt('.')
