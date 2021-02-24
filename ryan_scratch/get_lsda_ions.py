import sys
import os
import numpy as np
import functools

import jax.numpy as jnp
from jax import tree_util
from jax_dft import xc
from jax_dft import scf
from jax_dft import spin_scf
from jax_dft import utils
from jax_dft import jit_spin_scf
from jax_dft import jit_scf

class Generate_dataset():
  """Obtain dataset for LDA-calculated systems. Current support is for atoms.
  """

  def __init__(self, grids, selected_ions, locations=None):
    self.grids = grids
    self.selected_ions = selected_ions

    if locations is None:
      self.locations = [np.array([0])] * len(self.selected_ions)
    else:
      self.locations = locations

    # output quantities
    self.num_electrons = []
    self.num_unpaired_electrons = []
    self.nuclear_charges = []
    self.total_energies = []
    self.densities = []
    self.external_potentials = []
    self.xc_energies = []
    self.xc_energy_densities = []

  def run_lsda_selected_ions(self):
    xc_energy_density_fn = tree_util.Partial(xc.get_lsda_xc_energy_density_fn(),
      params=None)

    for ((nuclear_charge, num_electrons), center) in zip(self.selected_ions,
                                                        self.locations):

      if num_electrons % 2 == 0:
        num_unpaired_electrons = 0
      else:
        num_unpaired_electrons = 1

      num_down_electrons = (num_electrons - num_unpaired_electrons) // 2
      num_up_electrons = num_down_electrons + num_unpaired_electrons

      external_potential = utils.get_atomic_chain_potential(grids=grids,
        locations=center, nuclear_charges=np.array([nuclear_charge]),
        interaction_fn=utils.exponential_coulomb)

      densities, _ = spin_scf.batch_solve_noninteracting_system(
        jnp.array([external_potential, external_potential]),
        jnp.array([num_up_electrons, num_down_electrons]), grids)

      initial_density = jnp.sum(densities, axis=0)
      initial_spin_density = jnp.subtract(*densities)

      lsda_ksdft = jit_spin_scf.kohn_sham(external_potential=external_potential,
        num_electrons=num_electrons,
        num_unpaired_electrons=num_unpaired_electrons, num_iterations=6,
        grids=grids, xc_energy_density_fn=xc_energy_density_fn,
        interaction_fn=utils.exponential_coulomb,
        # The initial density of KS self-consistent calculations.
        initial_density=initial_density,
        initial_spin_density=initial_spin_density, alpha=0.5, alpha_decay=0.9,
        enforce_reflection_symmetry=False, num_mixing_iterations=1,
        density_mse_converge_tolerance=-1)

      lsda_ksdft = scf.get_final_state(lsda_ksdft)

      print(
        'finished: (Z, N_e) = (' + str(nuclear_charge) + ',' +
        str(num_electrons) + ')')

      self.num_electrons.append(num_electrons)
      self.num_unpaired_electrons.append(num_unpaired_electrons)
      self.nuclear_charges.append([nuclear_charge])
      self.total_energies.append(lsda_ksdft.total_energy)
      self.densities.append(lsda_ksdft.density)
      self.external_potentials.append(external_potential)
      self.xc_energies.append(lsda_ksdft.xc_energy)
      self.xc_energy_densities.append(lsda_ksdft.xc_energy_density)


  def save_dataset(self, out_dir):
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    np.save(os.path.join(out_dir, 'grids.npy'), self.grids)
    np.save(os.path.join(out_dir, 'locations.npy'), self.locations)
    np.save(os.path.join(out_dir, 'num_electrons.npy'), self.num_electrons)
    np.save(os.path.join(out_dir, 'num_unpaired_electrons.npy'),
            self.num_unpaired_electrons)
    np.save(os.path.join(out_dir, 'nuclear_charges.npy'),
            self.nuclear_charges)
    np.save(os.path.join(out_dir, 'total_energies.npy'),
            self.total_energies)
    np.save(os.path.join(out_dir, 'densities.npy'),
            self.densities)
    np.save(os.path.join(out_dir, 'external_potentials.npy'),
            self.external_potentials)
    np.save(os.path.join(out_dir, 'xc_energies.npy'),
            self.xc_energies)

    if self.xc_energy_densities:
      np.save(os.path.join(out_dir, 'xc_energy_densities.npy'),
              self.xc_energy_densities)


if __name__ == '__main__':
  """Get dataset for KSR calculations."""
  from jax_dft import datasets

  h = 0.08
  grids = np.arange(-256, 257) * h

  # previous dataset
  old = datasets.Dataset('../data/ions/lsda/', num_grids=513)
  print(old.total_energies)
  print(old.xc_energies)




  # ions are identified by: atomic number Z, number of electrons
  selected_ions = [(1, 1), (2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (4, 2),
                   (3, 3), (4, 3), (4, 4)]

  dataset = Generate_dataset(grids, selected_ions)
  dataset.run_lsda_selected_ions()

  print()
  print('Total energies:')
  print(dataset.total_energies)
  print('XC energies:')
  print(dataset.xc_energies)
  print('num_electrons')
  print(dataset.num_electrons)
  print('num_unpaired_electrons')
  print(dataset.num_unpaired_electrons)
  print('nuclear charges')
  print(dataset.nuclear_charges)

  out_dir = '../data/ions/lsda/'
  dataset.save_dataset(out_dir=out_dir)
