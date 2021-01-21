import numpy as np

from jax import tree_util
from jax_dft import xc
from jax_dft import scf
from jax_dft import utils
from jax_dft import jit_scf

import matplotlib.pyplot as plt

h = 0.08
grids = np.arange(-256, 257) * h
locations = np.asarray([0])
nuclear_charges = np.asarray([3])
num_electrons = 3



xc_energy_density_fn = tree_util.Partial(
        xc.get_unpolarized_lda_xc_energy_density_fn(), params=None)

external_potential = utils.get_atomic_chain_potential(
          grids=grids,
          locations=locations,
          nuclear_charges=nuclear_charges,
          interaction_fn=utils.exponential_coulomb)

initial_density, _, _ = scf.solve_noninteracting_system(external_potential, num_electrons,
                                                  grids)

lda_ksdft = scf.kohn_sham(
  locations=locations,
  nuclear_charges=nuclear_charges,
  num_electrons=num_electrons,
  num_iterations=15,
  grids=grids,
  xc_energy_density_fn=xc_energy_density_fn,
  interaction_fn=utils.exponential_coulomb,
  # The initial density of KS self-consistent calculations.
  initial_density=initial_density,
  alpha=0.7,
  alpha_decay=1.,
  enforce_reflection_symmetry=False,
  num_mixing_iterations=1,
  density_mse_converge_tolerance=-1)

print(lda_ksdft.total_energy)

plt.plot(grids, lda_ksdft.density[-1])

plt.savefig('lda_ksdft.pdf')