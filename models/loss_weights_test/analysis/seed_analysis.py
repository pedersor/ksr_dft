import os

os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_reductions'

import sys

# NOTE(pedersor): change to your local jax_dft_dev dir
abs_path_jax_dft = '/DFS-B/DATA/burke/pedersor/jax_dft_dev'
sys.path.append(abs_path_jax_dft)

import glob
import pickle
import time
from functools import partial

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

from longpaper_examples import analysis
from longpaper_examples import ksr

# Set the default dtype as float64
config.update('jax_enable_x64', True)

seed_dir = sys.argv[1]

# set dataset path
complete_dataset = datasets.Dataset(
  '/content/jax_dft_dev/data/ions/dmrg', num_grids=513)

weights = np.arange(0.25, 2.0, 0.25) / 2

ip_maes = []
ae_maes = []
density_loss_mae = []
train_energy_loss = []
train_density_loss = []

for weight in weights:

  print(f'weight = {weight}')

  # set model path
  model_base_dir = 'models/loss_weights_test/'
  weight_dir = f'w{2 * weight}'
  model_dir = os.path.join(model_base_dir, weight_dir, seed_dir)

  tester = ksr.SpinKSR(complete_dataset.grids)

  # set KS parameters
  tester.set_ks_params(
    # The number of Kohn-Sham iterations in training.
    num_iterations=30,
    # @The density linear mixing factor.
    alpha=0.5,
    # Decay factor of density linear mixing factor.
    alpha_decay=0.9,
    # Enforce reflection symmetry across the origin.
    enforce_reflection_symmetry=False,
    # The number of density differences in the previous iterations to mix the
    # density. Linear mixing is num_mixing_iterations = 1.
    num_mixing_iterations=1,
    # The stopping criteria of Kohn-Sham iteration on density.
    density_mse_converge_tolerance=-1.,
    # Apply stop gradient on the output state of this step and all steps
    # before. The first KS step is indexed as 0. Default -1, no stop gradient
    # is applied.
    stop_gradient_step=-1)

  # (semi-)local models
  network = neural_xc.build_sliding_net(
    window_size=1,
    num_filters_list=[16, 16, 16],
    activation='swish')

  # KSR-LSDA
  _, neural_xc_energy_density_fn = neural_xc.global_functional_sigma(
    network, grids=complete_dataset.grids)

  tester.set_neural_xc_functional(model_dir=model_dir,
    neural_xc_energy_density_fn=neural_xc_energy_density_fn)

  # train set:
  ions_dataset = datasets.Dataset(
    '/content/jax_dft_dev/data/ions/dmrg', num_grids=513)
  tester.set_test_set(ions_dataset.get_ions([(1, 1), (4, 2), (3, 3), (4, 4)]))
  ions_states = tester.get_test_states(
    optimal_ckpt_path=os.path.join(model_dir, 'optimal_ckpt.pkl'))

  # Energy loss
  energy_loss = weight * losses.trajectory_mse(
    target=tester.test_set.total_energy,
    predict=ions_states.total_energy[
      # The starting states have larger errors. Ignore a number of
      # starting states in loss.
    :, -1:],
    # The discount factor in the trajectory loss.
    discount=0.9, num_electrons=tester.test_set.num_electrons)

  # Density loss
  density_loss = (1 - weight) * losses.mean_square_error(
    target=tester.test_set.density,
    predict=ions_states.density[:, -1, :],
    num_electrons=tester.test_set.num_electrons
  ) * tester.grids_integration_factor

  train_energy_loss.append(energy_loss)
  train_density_loss.append(density_loss)

  # test set:
  # IP, AE, density loss

  # set ion test set
  ions_dataset = datasets.Dataset(
    '/content/jax_dft_dev/data/ions/dmrg', num_grids=513)
  tester.set_test_set(ions_dataset.get_ions())
  # load optimal checkpoint params
  ions_states = tester.get_test_states(
    optimal_ckpt_path=os.path.join(model_dir, 'optimal_ckpt.pkl'))
  ions_final_states = tester.get_final_states(ions_states)

  # set molecules test set
  molecules_dataset = datasets.Dataset(
    '/content/jax_dft_dev/data/molecules/relaxed_all', num_grids=513)
  tester.set_test_set(molecules_dataset.get_molecules())
  # load optimal checkpoint params
  molecules_states = tester.get_test_states(
    optimal_ckpt_path=os.path.join(model_dir, 'optimal_ckpt.pkl'))
  molecules_final_states = tester.get_final_states(molecules_states)

  _, ip_mae = analysis.get_ip_table(ions_dataset, ions_final_states)
  _, ae_mae = analysis.get_ae_table(molecules_final_states,
    molecules_dataset, ions_dataset, ions_final_states)

  _, _, ions_density_loss = analysis.get_error_table(ions_dataset,
    ions_final_states)
  _, _, molecules_density_loss = analysis.get_error_table(molecules_dataset,
    molecules_final_states)

  density_loss = np.concatenate((ions_density_loss, molecules_density_loss))
  density_loss_mae.append(np.mean(density_loss))
  ip_maes.append(ip_mae)
  ae_maes.append(ae_mae)

# convert to arrays..
ip_maes = np.asarray(ip_maes)
ae_maes = np.asarray(ae_maes)
density_loss_mae = np.asarray(density_loss_mae)
train_energy_loss = np.asarray(train_energy_loss)
train_density_loss = np.asarray(train_density_loss)

np.save(f'ip_maes_{seed_dir}.npy', ip_maes)
np.save(f'ae_maes_{seed_dir}.npy', ae_maes)
np.save(f'density_loss_mae_{seed_dir}.npy', density_loss_mae)
np.save(f'train_energy_loss_{seed_dir}.npy', train_energy_loss)
np.save(f'train_density_loss_{seed_dir}.npy', train_density_loss)
