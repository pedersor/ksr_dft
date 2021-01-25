import os
import sys
sys.path.append('../')
from longpaper_examples.train_validate_ions import Train_validate_ions

import jax
import jax.numpy as jnp
from jax import tree_util
from jax_dft import xc
from jax_dft import scf
from jax_dft import spin_scf
from jax_dft import utils
from jax_dft import jit_spin_scf
from jax_dft import jit_scf

from functools import partial

import matplotlib.pyplot as plt
import time

# load complete dataset
ions = Train_validate_ions('../data/ions/lsda/basic_all')
dataset = ions.get_complete_dataset(num_grids=513)

# set test set
to_test = [(1, 1), (2, 2)]
mask = dataset.get_mask_ions(to_test)
test_set = dataset.get_subdataset(mask)
test_set = test_set.get_ions()

initial_densities, initial_spin_densities = spin_scf.get_initial_density_sigma(
  test_set, "noninteracting")

test_set = test_set._replace(
  initial_densities=initial_densities,
  initial_spin_densities=initial_spin_densities
)


@jax.jit
def kohn_sham(locations, nuclear_charges,
              initial_densities, initial_spin_densities, num_electrons,
              num_unpaired_electrons):
  return _kohn_sham(locations, nuclear_charges, initial_densities,
                    initial_spin_densities, num_electrons,
                    num_unpaired_electrons)


@partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0))
def _kohn_sham(locations, nuclear_charges, initial_densities,
               initial_spin_densities, num_electrons, num_unpaired_electrons):
  return jax.lax.cond(
    num_unpaired_electrons == 0,
    true_fun=lambda _: jit_scf.kohn_sham(
      locations=locations,
      nuclear_charges=nuclear_charges,
      num_electrons=num_electrons,
      num_unpaired_electrons=num_unpaired_electrons,
      grids=test_set.grids[0],
      xc_energy_density_fn=tree_util.Partial(
        xc.get_lsda_xc_energy_density_fn(), params=None),
      interaction_fn=utils.exponential_coulomb,
      initial_density=initial_densities,
      initial_spin_density=initial_spin_densities,
      num_iterations=15,
      alpha=0.5,
      alpha_decay=0.9,
      enforce_reflection_symmetry=False,
      num_mixing_iterations=1,
      density_mse_converge_tolerance=-1,
      stop_gradient_step=-1),
    false_fun=lambda _: jit_spin_scf.kohn_sham(
      locations=locations,
      nuclear_charges=nuclear_charges,
      num_electrons=num_electrons,
      num_unpaired_electrons=num_unpaired_electrons,
      grids=test_set.grids[0],
      xc_energy_density_fn=tree_util.Partial(
        xc.get_lsda_xc_energy_density_fn(), params=None),
      interaction_fn=utils.exponential_coulomb,
      initial_density=initial_densities,
      initial_spin_density=initial_spin_densities,
      num_iterations=15,
      alpha=0.5,
      alpha_decay=0.9,
      enforce_reflection_symmetry=False,
      num_mixing_iterations=1,
      density_mse_converge_tolerance=-1,
      stop_gradient_step=-1),
    operand=None
  )


for _ in range(3):
  start_time = time.time()
  res = kohn_sham(test_set.locations, test_set.nuclear_charges,
                  test_set.initial_densities,
                  test_set.initial_spin_densities,
                  test_set.num_electrons,
                  test_set.num_unpaired_electrons)
  print(f'total time: {time.time() - start_time}')
  print(res.total_energy)
