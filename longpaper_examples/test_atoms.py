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

  def set_test_set(self, selected_ions):
    """Sets the validation set from a list of ions."""
    self.test_set = self.complete_dataset.get_atoms(
      selected_ions=selected_ions)
    # obtain initial densities
    initial_densities = scf.get_initial_density(self.test_set,
                                                method='noninteracting')
    self.test_set = self.test_set._replace(
      initial_densities=initial_densities)
    return self

  def get_test_states(self, optimal_ckpt_path=None):
    if optimal_ckpt_path is not None:
      with open(optimal_ckpt_path, 'rb') as handle:
        self.optimal_ckpt_params = pickle.load(handle)

    states = self.kohn_sham(
      self.optimal_ckpt_params,
      locations=self.test_set.locations,
      nuclear_charges=self.test_set.nuclear_charges,
      initial_densities=self.test_set.initial_densities)
    return states

  def get_final_test_states(self, optimal_ckpt_path=None):
    states = self.get_test_states(optimal_ckpt_path=optimal_ckpt_path)
    # get only converged results
    return tree_util.tree_map(lambda x: x[:, -1], states)

  def _get_states(self, ckpt_path):
    print(f'Load {ckpt_path}')
    with open(ckpt_path, 'rb') as handle:
      params = pickle.load(handle)

    states = self.kohn_sham(
      params,
      locations=self.validation_set.locations,
      nuclear_charges=self.validation_set.nuclear_charges,
      initial_densities=self.validation_set.initial_densities)
    return params, states

  def get_optimal_ckpt(self, path_to_ckpts):
    # TODO: non-jitted/vmapped option. Dynamic num_electrons.
    ckpt_list = sorted(
      glob.glob(os.path.join(path_to_ckpts, 'ckpt-?????')))

    optimal_ckpt_params = None
    min_loss = None
    for ckpt_path in ckpt_list:
      params, states = self._get_states(ckpt_path)

      # Energy loss
      loss_value = losses.mean_square_error(
        target=self.validation_set.total_energy,
        predict=states.total_energy[:, -1]) / self.num_electrons

      # Density loss (however, KSR paper does not use this for validation)
      # loss_value += losses.mean_square_error(
      #  target=self.validation_set.density, predict=states.density[:, -1, :]
      # ) * self.grids_integration_factor / self.num_electrons

      if optimal_ckpt_params is None or loss_value < min_loss:
        optimal_ckpt_params = params
        min_loss = loss_value

    optimal_ckpt_path = os.path.join(path_to_ckpts, 'optimal_ckpt.pkl')
    print(f'optimal checkpoint loss: {min_loss}')
    print(f'Save {optimal_ckpt_path}')
    with open(optimal_ckpt_path, 'wb') as handle:
      pickle.dump(optimal_ckpt_params, handle)

    self.optimal_ckpt_params = optimal_ckpt_params
    return self


if __name__ == '__main__':
  """Obtain optimal parameters from validation."""
  two_electrons = Test_atoms('../data/ions/num_electrons_2')
  two_electrons.get_complete_dataset(num_grids=513)

  # set validation set
  to_validate = [3]
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

  # get optimal checkpoint from validation
  two_electrons.get_optimal_ckpt('')

  # set test set
  to_test = [4]
  two_electrons.set_test_set(selected_ions=to_test)

  # solve KS systems with optimal ML model
  test_states = two_electrons.get_test_states()

  # TODO: final states [-1]..
  print(test_states.total_energy.shape)
  print(two_electrons.test_set.total_energy[0])
