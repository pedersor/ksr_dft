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

from ks_regularizer.ksr import SpinKSR

# Set the default dtype as float64
config.update('jax_enable_x64', True)

""" Run this file using e.g. 
`$python3 train_validate.py t2 v1 s0`
This trains on the 't2' set and validates on the 'v1' set. Initial parameters
for training are generated using random seed 0. 
"""

complete_dataset = datasets.Dataset(
  os.path.join(abs_path_jax_dft, 'data/ions/dmrg'), num_grids=513)
trainer = SpinKSR(complete_dataset)

training_sets_dict = {'t2': [(1, 1), (2, 2)], 't3': [(1, 1), (2, 2), (3, 3)],
  't4': [(1, 1), (2, 2), (3, 3), (4, 4)],
  't5': [(1, 1), (2, 2), (3, 3), (4, 4), (4, 2)], }
# load training set from sys passed arg
train_dir = sys.argv[1]
to_train = training_sets_dict[train_dir]
training_set = complete_dataset.get_ions(to_train)
trainer.set_training_set(training_set)
print(f'to_train = {to_train}')

validation_sets_dict = {'v1': [(4, 3)]}
# load validation set from sys passed arg
validation_dir = sys.argv[2]
to_validate = validation_sets_dict[validation_dir]
validation_set = complete_dataset.get_ions(to_validate)
trainer.set_validation_set(validation_set)
print(f'to_validate = {to_validate}')

# load random seed num from sys passed arg and create key
random_seed_dir = sys.argv[3]
seed_num = int(random_seed_dir[1:])
key = jax.random.PRNGKey(seed_num)
print(f'seed = {seed_num}')

model_dir = ''

# set KS parameters
trainer.set_ks_params(  # The number of Kohn-Sham iterations in training.
  num_iterations=10,  # @The density linear mixing factor.
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
  stop_gradient_step=-1)

# set ML model for xc functional
network = neural_xc.build_global_local_conv_net_sigma(num_global_filters=8,
  num_local_filters=16, num_local_conv_layers=2, activation='swish',
  grids=complete_dataset.grids, minval=0.1, maxval=2.385345,
  downsample_factor=0)
init_fn, neural_xc_energy_density_fn = neural_xc.global_functional_sigma(
  network, grids=complete_dataset.grids)

trainer.set_neural_xc_functional(model_dir=model_dir,
  neural_xc_energy_density_fn=neural_xc_energy_density_fn)

# set initial params from init_fn
trainer.set_init_model_params(init_fn, key, verbose=1)

trainer.setup_optimization(initial_checkpoint_index=0, save_every_n=10,
  max_train_steps=300, num_skipped_energies=-1, )

# perform training optimization
trainer.do_lbfgs_optimization(verbose=1)

# get optimal checkpoint from validation
_ = trainer.get_optimal_ckpt(model_dir)
