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

    states = self.kohn_sham(
      params,
      locations=self.validation_set.locations,
      nuclear_charges=self.validation_set.nuclear_charges,
      initial_densities=self.validation_set.initial_densities)
    return states

  def get_optimal_ckpt(self, path_to_ckpts):
    ckpt_list = sorted(
      glob.glob(os.path.join(path_to_ckpts, 'ckpt-?????')))

    for ckpt_path in ckpt_list:
      states = self._get_states(ckpt_path)

      print(states.total_energy)

    return


if __name__ == '__main__':
  two_electrons = Test_atoms('../data/ions/num_electrons_2')
  two_electrons.get_complete_dataset(num_grids=513)

  # set validation set
  to_validate = [3, 4]
  two_electrons.set_validation_set(selected_ions=to_validate)

  # set ML model
  two_electrons.init_ksr_lda_model()
  print(f'number of parameters: {two_electrons.num_parameters}')

  # get KS parameters
  two_electrons.set_ks_parameters(
    # The number of Kohn-Sham iterations in training.
    num_iterations=15,
    # @The density linear mixing factor.
    alpha=0.5,
    # Decay factor of density linear mixing factor.
    alpha_decay=0.9,
    # Enforce reflection symmetry across the origin.
    enforce_reflection_symmetry=True,
    # The number of density differences in the previous iterations to mix the
    # density. Linear mixing is num_mixing_iterations = 1.
    num_mixing_iterations=1,
    # The stopping criteria of Kohn-Sham iteration on density.
    density_mse_converge_tolerance=-1.,
    # Apply stop gradient on the output state of this step and all steps
    # before. The first KS step is indexed as 0. Default -1, no stop gradient
    # is applied.
    stop_gradient_step=-1
  )

  two_electrons.get_optimal_ckpt('')
