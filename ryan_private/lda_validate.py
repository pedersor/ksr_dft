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

config.update('jax_enable_x64', True)

dataset = datasets.Dataset(path='../data/ions/num_electrons_4/', num_grids=513)

grids = dataset.grids
# test set
to_test = [(4, 4)]
test_set = dataset.get_atoms(to_test)
initial_densities = scf.get_initial_density(
  test_set, method='noninteracting')

# @title Initialize network
network = neural_xc.build_sliding_net(
  window_size=1,
  num_filters_list=[16, 16, 16],
  activation='swish')
init_fn, neural_xc_energy_density_fn = neural_xc.global_functional(
  network, grids=grids)

# @markdown The number of Kohn-Sham iterations in training.
num_iterations = 15  # @param{'type': 'integer'}
# @markdown The density linear mixing factor.
alpha = 0.5  # @param{'type': 'number'}
# @markdown Decay factor of density linear mixing factor.
alpha_decay = 0.9  # @param{'type': 'number'}
# @markdown The number of density differences in the previous iterations to mix the
# @markdown density. Linear mixing is num_mixing_iterations = 1.
num_mixing_iterations = 1  # @param{'type': 'integer'}
# @markdown The stopping criteria of Kohn-Sham iteration on density.
density_mse_converge_tolerance = -1.  # @param{'type': 'number'}
# @markdown Apply stop gradient on the output state of this step and all steps
# @markdown before. The first KS step is indexed as 0. Default -1, no stop gradient
# @markdown is applied.
stop_gradient_step = -1  # @param{'type': 'integer'}


def kohn_sham(
    params, locations, nuclear_charges, initial_density=None, use_lda=False):
  return scf.kohn_sham(
    locations=locations,
    nuclear_charges=nuclear_charges,
    num_electrons=dataset.num_electrons,
    num_iterations=num_iterations,
    grids=grids,
    xc_energy_density_fn=tree_util.Partial(
      xc.get_lda_xc_energy_density_fn() if use_lda else neural_xc_energy_density_fn,
      params=params),
    interaction_fn=utils.exponential_coulomb,
    # The initial density of KS self-consistent calculations.
    initial_density=initial_density,
    alpha=alpha,
    alpha_decay=alpha_decay,
    enforce_reflection_symmetry=True,
    num_mixing_iterations=num_mixing_iterations,
    density_mse_converge_tolerance=density_mse_converge_tolerance)


def get_states(ckpt_path):
  print(f'Load {ckpt_path}')
  with open(ckpt_path, 'rb') as handle:
    params = pickle.load(handle)
  states = []
  for i in range(len(to_test)):
    states.append(kohn_sham(
      params,
      locations=test_set.locations[i],
      nuclear_charges=test_set.nuclear_charges[i],
      initial_density=initial_densities[i]))
  return tree_util.tree_multimap(lambda *x: jnp.stack(x), *states)


ckpt_list = sorted(glob.glob('ckpt-?????'))
num_ckpts = len(ckpt_list)
states = None
for ckpt_path in ckpt_list:
  if ckpt_path == 'ckpt-00040':
    states = get_states(ckpt_path)

test_idx = 0
print(states.total_energy[:, -1])

print(states.xc_energy[:, -1])
print(states.kinetic_energy[:, -1])

plt.plot(grids,
         states.density[test_idx][-1] * states.xc_energy_density[test_idx][-1])

# real lda..
lda = kohn_sham(
  None,
  locations=test_set.locations[test_idx],
  nuclear_charges=test_set.nuclear_charges[test_idx],
  use_lda=True)

plt.plot(grids, lda.xc_energy_density[-1] * lda.density[-1], linestyle='dashed',
         label=r'real lda..')

plt.xlim(-5, 5)
plt.legend()
plt.savefig('test_0.pdf')
