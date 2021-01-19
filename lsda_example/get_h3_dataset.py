import numpy as np

from jax import tree_util
from jax_dft import xc
from jax_dft import spin_scf
from jax_dft import utils

import matplotlib.pyplot as plt
import time

h = 0.08
grids = np.arange(-256, 257) * h
locations = np.asarray([0])
nuclear_charges = np.asarray([3])
num_electrons = 3
num_unpaired_electrons = 1


xc_energy_density_fn = tree_util.Partial(
        xc.get_lsda_xc_energy_density_fn(), params=None)

start_time = time.time()

lda_ksdft = spin_scf.kohn_sham(
  locations=locations,
  nuclear_charges=nuclear_charges,
  num_electrons=num_electrons,
  num_unpaired_electrons=num_unpaired_electrons,
  num_iterations=15,
  grids=grids,
  xc_energy_density_fn=xc_energy_density_fn,
  interaction_fn=utils.exponential_coulomb,
  # The initial density of KS self-consistent calculations.
  initial_density=None,
  alpha=0.7,
  alpha_decay=1.,
  enforce_reflection_symmetry=False,
  num_mixing_iterations=1,
  density_mse_converge_tolerance=-1)

total_time = time.time() - start_time
print(f'total time = {total_time}')

print('total energies:')
print(lda_ksdft.total_energy)
print('kinetic energies:')
print(lda_ksdft.kinetic_energy)
print('xc energies:')
print(lda_ksdft.xc_energy)

plt.plot(grids, lda_ksdft.density[-1])

plt.savefig('lsda_ksdft.pdf')