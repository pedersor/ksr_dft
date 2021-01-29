import numpy as np

import jax.numpy as jnp
from jax import tree_util
from jax_dft import xc
from jax_dft import scf
from jax_dft import spin_scf
from jax_dft import utils
from jax_dft import jit_spin_scf
from jax_dft import jit_scf


from jax.config import config
config.update('jax_disable_jit', False)

import matplotlib.pyplot as plt
import time
import sys

h = 0.08
grids = np.arange(-256, 257) * h
locations = np.asarray([0])
nuclear_charges = np.asarray([3])
num_electrons = 3
num_unpaired_electrons = 1


num_down_electrons = (num_electrons - num_unpaired_electrons) // 2
num_up_electrons = num_down_electrons + num_unpaired_electrons

xc_energy_density_fn = tree_util.Partial(
        xc.get_lsda_xc_energy_density_fn(), params=None)


external_potential = utils.get_atomic_chain_potential(
          grids=grids,
          locations=locations,
          nuclear_charges=nuclear_charges,
          interaction_fn=utils.exponential_coulomb)

densities, _ = spin_scf.batch_solve_noninteracting_system(
jnp.array([external_potential, external_potential]),
jnp.array([num_up_electrons, num_down_electrons]), grids)

initial_density = jnp.sum(densities, axis=0)
initial_spin_density = jnp.subtract(*densities)


start_time = time.time()

lsda_ksdft = spin_scf.kohn_sham(
  locations=locations,
  nuclear_charges=nuclear_charges,
  num_electrons=num_electrons,
  num_unpaired_electrons=num_unpaired_electrons,
  num_iterations=15,
  grids=grids,
  xc_energy_density_fn=xc_energy_density_fn,
  interaction_fn=utils.exponential_coulomb,
  # The initial density of KS self-consistent calculations.
  initial_density=initial_density,
  initial_spin_density=initial_spin_density,
  alpha=0.7,
  alpha_decay=1.,
  enforce_reflection_symmetry=False,
  num_mixing_iterations=1,
  density_mse_converge_tolerance=-1)

total_time = time.time() - start_time
print(f'total time = {total_time}')

print('total energies:')
print(lsda_ksdft.total_energy)
print('kinetic energies:')
print(lsda_ksdft.kinetic_energy)
print('xc energies:')
print(lsda_ksdft.xc_energy)
