import os
import sys

sys.path.append('../')
from jax_dft import scf

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
import sys


h = 0.08
grids = jnp.arange(-256, 257) * h
locations = jnp.asarray([[0],[0]])
nuclear_charges = jnp.asarray([[2], [3]])
num_electrons = jnp.array([2, 3])
num_unpaired_electrons = jnp.array([0, 1])

num_down_electrons = (num_electrons - num_unpaired_electrons) // 2
num_up_electrons = num_down_electrons + num_unpaired_electrons

num_up_down = jnp.transpose(jnp.array([num_up_electrons, num_down_electrons]))


external_potentials = []
for i in range(len(locations)):
  external_potentials.append(utils.get_atomic_chain_potential(
            grids=grids,
            locations=locations[i],
            nuclear_charges=nuclear_charges[i],
            interaction_fn=utils.exponential_coulomb))
external_potentials = jnp.asarray(external_potentials)


external_potentials = jnp.expand_dims(external_potentials, axis=1)
external_potentials = jnp.repeat(external_potentials, repeats=2, axis=1)



solve = jax.vmap(spin_scf.batch_solve_noninteracting_system, in_axes=(0, 0, None))

initial_up_down_densities, _ = solve(external_potentials, num_up_down, grids)


initial_densities = jnp.sum(initial_up_down_densities, axis=1)

initial_spin_densities = jnp.squeeze(-1 * jnp.diff(initial_up_down_densities, axis=1))


def kohn_sham(locations, nuclear_charges,
              initial_densities, initial_spin_densities, num_electrons,
              num_unpaired_electrons):
  return _kohn_sham(locations, nuclear_charges,initial_densities,
                    initial_spin_densities, num_electrons, num_unpaired_electrons)

@partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0))
def _kohn_sham(locations, nuclear_charges, initial_densities,
               initial_spin_densities, num_electrons, num_unpaired_electrons):


  return jax.lax.cond(num_unpaired_electrons == 0,
    lambda _: jit_scf.kohn_sham(
      locations=locations,
      nuclear_charges=nuclear_charges,
      num_electrons=num_electrons,
      num_unpaired_electrons=num_unpaired_electrons,
      grids=grids,
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
    lambda _: jit_spin_scf.kohn_sham(
      locations=locations,
      nuclear_charges=nuclear_charges,
      num_electrons=num_electrons,
      num_unpaired_electrons=num_unpaired_electrons,
      grids=grids,
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
     operand=None)


start_time = time.time()
res = kohn_sham(locations, nuclear_charges, initial_densities,
                initial_spin_densities, num_electrons, num_unpaired_electrons)
print(f'total time: {time.time() - start_time}')
print(res.total_energy)


