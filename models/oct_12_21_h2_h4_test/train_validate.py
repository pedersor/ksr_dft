import os
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_reductions'
import pathlib
import sys
import argparse

# NOTE(pedersor): change to your local jax_dft_dev dir
abs_path_jax_dft = pathlib.Path('/DFS-B/DATA/burke/pedersor/jax_dft_dev')
sys.path.append(abs_path_jax_dft)

import numpy as np
import jax
from jax.config import config
from jax_dft import datasets
from jax_dft import neural_xc
from jax_dft import utils


from ks_regularizer import ksr

# Set the default dtype as float64
config.update('jax_enable_x64', True)

# load random seed num from sys passed arg and create key
parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--seed",
    help="random seed int",
    type=str,
    default="0",
)
args = parser.parse_args()

random_seed_dir = 's' + args.seed
seed = int(args.seed)
key = jax.random.PRNGKey(seed)
print(f'seed = {seed}')

# path to save results
model_dir = ''

# load datasets
ions_path = pathlib.Path(abs_path_jax_dft / "data/ions")
molecules_path = pathlib.Path(abs_path_jax_dft / 'data/molecules')

ions_dataset = datasets.Dataset(ions_path / 'dmrg', num_grids=513)
h2_dataset = datasets.Dataset(molecules_path / 'h2', num_grids=513)
h4_dataset = datasets.Dataset(molecules_path / 'h4', num_grids=513)

grids = ions_dataset.grids 

# training sets
unpolarized_h = ions_dataset.get_ions([(1, 1)])
unpolarized_h = unpolarized_h._replace(num_unpaired_electrons=np.array([0]))

training_set = datasets.concatenate_kohn_sham_states(
    h2_dataset.get_molecules([128, 384]),
    h4_dataset.get_molecules([208, 336]),)

# validation set
validation_set = datasets.concatenate_kohn_sham_states(
    h2_dataset.get_molecules([168, 296, 440, 552]),
    h4_dataset.get_molecules([184, 264, 328, 504]),)

# set trainer
sKRS = ksr.SpinKSR(grids)
sKRS.set_training_set(training_set)
sKRS.set_validation_set(validation_set)

# set KS parameters
sKRS.set_ks_params(  
  # The number of Kohn-Sham iterations in training.
  num_iterations=30,  
  # The density linear mixing factor.
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
  stop_gradient_step=-1,)

# set ML model for xc functional
network = neural_xc.build_global_local_conv_net_sigma(num_global_filters=8,
  num_local_filters=16, num_local_conv_layers=2, activation='swish',
  grids=grids, minval=0.1, maxval=2.385345,
  downsample_factor=0)

network = neural_xc.wrap_network_with_self_interaction_layer_sigma(
    network, grids=grids,
    interaction_fn=utils.exponential_coulomb) 
 
init_fn, neural_xc_energy_density_fn = neural_xc.global_functional_sigma(
  network, grids=grids)

sKRS.set_neural_xc_functional(model_dir=model_dir,
  neural_xc_energy_density_fn=neural_xc_energy_density_fn)

# set initial params from init_fn
sKRS.set_init_model_params(init_fn, key, verbose=1)

sKRS.setup_optimization(
  initial_checkpoint_index=0,
  save_every_n=10,
  max_train_steps=1200,
  num_skipped_energies=-1,)

sKRS.do_lbfgs_optimization(verbose=1)

# validation stage

# increase num iterations in validation
sKRS.ks_params['num_iterations'] = 40

# get optimal checkpoint from validation
_ = sKRS.get_optimal_ckpt(model_dir)
