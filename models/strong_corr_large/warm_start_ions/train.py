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

from ks_regularizer import ksr

# Set the default dtype as float64
config.update('jax_enable_x64', True)

# model dir to save results
model_dir = ''

# datasets
ions_path = os.path.join(abs_path_jax_dft, 'data/ions')
molecules_path = os.path.join(abs_path_jax_dft, 'data/molecules')

ions_dataset = datasets.Dataset(os.path.join(ions_path, 'dmrg'), num_grids=513)
h2_dataset = datasets.Dataset(os.path.join(molecules_path, 'h2'), num_grids=513)
h3_dataset = datasets.Dataset(os.path.join(molecules_path, 'h3'), num_grids=513)
h4_dataset = datasets.Dataset(os.path.join(molecules_path, 'h4'), num_grids=513)
li_h_dataset = datasets.Dataset(os.path.join(molecules_path, 'li_h'),
  num_grids=513)
h_be_h_dataset = datasets.Dataset(os.path.join(molecules_path, 'h_be_h'),
  num_grids=513)
h3_plus_dataset = datasets.Dataset(os.path.join(molecules_path, 'h3_plus'),
  num_grids=513)
he_h_h_he_2plus_dataset = datasets.Dataset(os.path.join(molecules_path,
  'he_h_h_he_2plus'), num_grids=513)
h2_plus_dataset = datasets.Dataset(os.path.join(molecules_path, 'h2_plus'),
  num_grids=513)
he_h_plus_dataset = datasets.Dataset(os.path.join(molecules_path,
  'he_h_plus'), num_grids=513)

grids = ions_dataset.grids  # same grids for all datasets...

# training sets
ion_training_set = ions_dataset.get_ions(
  [(1, 1), (4, 2), (3, 3), (4, 4), (2, 2)])

unpolarized_h = ions_dataset.get_ions([(1, 1)])
unpolarized_h = unpolarized_h._replace(num_unpaired_electrons=np.array([0]))

h2_training_set = h2_dataset.get_molecules([152, 384])
h4_training_set = h4_dataset.get_molecules([200, 384])
h3_plus_training_set = h3_plus_dataset.get_molecules([168, 384])
h_be_h_training_set = h_be_h_dataset.get_molecules([352])
li_h_training_set = li_h_dataset.get_molecules([312])
he_h_plus_training_set = he_h_plus_dataset.get_molecules([312, 504])
h3_training_set = h3_dataset.get_molecules([208])

training_set = datasets.concatenate_kohn_sham_states(
  ion_training_set,
  h2_training_set,
  h4_training_set,
  h3_plus_training_set,
  h_be_h_training_set,
  li_h_training_set,
  he_h_plus_training_set,
  h3_training_set)

# validation sets
ions_validation_set = ions_dataset.get_ions([(4, 3)])
h3_validation_set = h3_dataset.get_molecules([208])
he_h_h_he_2plus_validation_set = he_h_h_he_2plus_dataset.get_molecules([448])

# validation_set = ions_validation_set
validation_set = datasets.concatenate_kohn_sham_states(
  ions_validation_set, h3_validation_set, he_h_h_he_2plus_validation_set)

# init trainer with grids
trainer = ksr.SpinKSR(grids)
# set training/validation sets
trainer.set_training_set(training_set)
trainer.set_validation_set(validation_set)

# set KS parameters
trainer.set_ks_params(  # The number of Kohn-Sham iterations in training.
  num_iterations=20,  # @The density linear mixing factor.
  alpha=0.5,  # Decay factor of density linear mixing factor.
  alpha_decay=0.9,  # Enforce reflection symmetry across the origin.
  enforce_reflection_symmetry=False,
  # The number of density differences in the previous iterations to mix the
  # density. Linear mixing is num_mixing_iterations = 1.
  num_mixing_iterations=1,
  # The stopping criteria of Kohn-Sham iteration on density.
  density_mse_converge_tolerance=-1.,
  # Apply stop gradient on the output state of this step and all steps
  # before. The first KS step is indexed as 0. Default -1, no stop gradient
  # is applied.
  stop_gradient_step=-1, )

# set ML model for xc functional
network = neural_xc.build_global_local_conv_net_sigma(num_global_filters=8,
  num_local_filters=16, num_local_conv_layers=2, activation='swish',
  grids=h2_dataset.grids, minval=0.1, maxval=2.385345,
  downsample_factor=0)
init_fn, neural_xc_energy_density_fn = neural_xc.global_functional_sigma(
  network, grids=grids)

trainer.set_neural_xc_functional(model_dir=model_dir,
  neural_xc_energy_density_fn=neural_xc_energy_density_fn)

# set initial params from init_fn
key = jax.random.PRNGKey(0)
trainer.set_init_model_params(init_fn, key, verbose=1)

trainer.setup_optimization(
  initial_checkpoint_index=680,
  save_every_n=10,
  max_train_steps=500,
  num_skipped_energies=-1,)
  #initial_params_file=os.path.join(abs_path_jax_dft,
  #  'models/ions/ksr_global/t5_v1/optimal_ckpt.pkl'))

# perform training optimization
trainer.do_lbfgs_optimization(verbose=1)
