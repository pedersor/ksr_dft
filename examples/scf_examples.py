import numpy as np

import jax.numpy as jnp
from jax import tree_util

from jax_dft import xc
from jax_dft import utils
from jax_dft import scf
from jax_dft import jit_scf
from jax_dft import spin_scf
from jax_dft import jit_spin_scf
from jax.config import config

# Set the default dtype as float64
config.update('jax_enable_x64', True)

def lsda_scf_example():
    # grid with spacing h
    h = 0.04
    grids = np.arange(-256, 257) * h

    # H, Li examples
    symbols = ["H", "Li"]
    num_electrons = np.array([1,3])
    num_unpaired_electrons = np.array([1,1])
    nuclear_charges = np.array([[1], [3]])
    num_samples = len(num_electrons)
    locations = np.array([[0]]*num_samples)   
    external_potentials = utils.get_atomic_chain_potential_batch(
        grids=grids,
        locations=locations,
        nuclear_charges=nuclear_charges,
        interaction_fn=utils.exponential_coulomb)

    # ref. values
    ref_total_energy = np.array([-0.643, -4.181])
    ref_x_energy = np.array([-0.305, -0.999])
    ref_c_energy = np.array([-0.009, -0.045])
    ref_xc_energy = ref_x_energy + ref_c_energy
    ref_density = [None, None] 

    system_info = scf.KohnShamState(
        density=ref_density,
        total_energy=ref_total_energy,
        external_potential=external_potentials,
        nuclear_charges=nuclear_charges,
        grids=np.tile(np.expand_dims(grids, axis=0), reps=(num_samples, 1)),
        num_electrons=num_electrons,
        locations=locations,
        num_unpaired_electrons=num_unpaired_electrons,
        xc_energy=ref_xc_energy,
        converged=np.array([True, True]))

    # generate initial densities
    initial_densities, initial_spin_densities = (
        spin_scf.get_initial_density_sigma(system_info, method='noninteracting'))
    system_info = system_info._replace(
    initial_densities=initial_densities,
    initial_spin_densities=initial_spin_densities)

    # use LDA functional
    xc_energy_density_fn = tree_util.Partial(
    xc.get_lsda_xc_energy_density_fn(), params=None)

    for i in range(len(system_info.num_electrons)):

        # non-jitted KS scf
        lsda_ksdft_scf = spin_scf.kohn_sham(
            num_electrons=system_info.num_electrons[i],
            num_iterations=30,
            grids=grids,
            xc_energy_density_fn=xc_energy_density_fn,
            interaction_fn=utils.exponential_coulomb,
            initial_density=system_info.initial_densities[i],
            alpha=0.5,
            alpha_decay=0.9,
            enforce_reflection_symmetry=False,
            num_mixing_iterations=1,
            density_mse_converge_tolerance=-1,

            num_unpaired_electrons=system_info.num_unpaired_electrons[i],
            initial_spin_density=system_info.initial_spin_densities[i],
            locations=system_info.locations[i],
            nuclear_charges=system_info.nuclear_charges[i],
            )

        # jitted KS scf
        lsda_ksdft_jit_scf = jit_spin_scf.kohn_sham(
            num_electrons=system_info.num_electrons[i],
            num_unpaired_electrons=system_info.num_unpaired_electrons[i],
            num_iterations=30,
            grids=grids,
            xc_energy_density_fn=xc_energy_density_fn,
            interaction_fn=utils.exponential_coulomb,
            initial_density=system_info.initial_densities[i],
            alpha=0.5,
            alpha_decay=0.9,
            enforce_reflection_symmetry=False,
            num_mixing_iterations=1,
            density_mse_converge_tolerance=-1,

            external_potential=system_info.external_potential[i],
            initial_spin_density=system_info.initial_spin_densities[i],
            )

        print(f'Example system: {symbols[i]}')
        states = {"spin_scf result: ": lsda_ksdft_scf, "jit_spin_scf result: ": lsda_ksdft_jit_scf}
        for label, state in states.items():
            total_energy = state.total_energy[-1].block_until_ready()
            kinetic_energy = state.kinetic_energy[-1]
            xc_energy = state.xc_energy[-1]

            print(label)
            print(f'E = {total_energy}')
            print(f'E_xc = {xc_energy}')

            # check against reference values
            np.testing.assert_allclose(system_info.total_energy[i], total_energy, rtol=0, atol=10**(-3))
            np.testing.assert_allclose(system_info.xc_energy[i], xc_energy, rtol=0, atol=10**(-3))
            print('(match ref. values)')

def lda_scf_example():
  # grid with spacing h
  h = 0.04
  grids = np.arange(-256, 257) * h

  # He, Be examples
  symbols = ["He", "Be"]
  num_electrons = np.array([2,4])
  num_unpaired_electrons = np.array([0,0])
  nuclear_charges = np.array([[2], [4]])
  num_samples = len(num_electrons)
  locations = np.array([[0]]*num_samples)   
  external_potentials = utils.get_atomic_chain_potential_batch(
      grids=grids,
      locations=locations,
      nuclear_charges=nuclear_charges,
      interaction_fn=utils.exponential_coulomb)

  # ref. values
  ref_total_energy = np.array([-2.196, -6.784])
  ref_x_energy = np.array([-0.633, -1.371])
  ref_c_energy = np.array([-0.050, -0.080])
  ref_xc_energy = ref_x_energy + ref_c_energy
  ref_density = [None, None] 

  system_info = scf.KohnShamState(
      density=ref_density,
      total_energy=ref_total_energy,
      external_potential=external_potentials,
      nuclear_charges=nuclear_charges,
      grids=np.tile(np.expand_dims(grids, axis=0), reps=(num_samples, 1)),
      num_electrons=num_electrons,
      locations=locations,
      num_unpaired_electrons=num_unpaired_electrons,
      xc_energy=ref_xc_energy,
      converged=np.array([True, True]))

  # generate initial densities
  initial_densities, initial_spin_densities = (
      spin_scf.get_initial_density_sigma(system_info, method='noninteracting'))
  system_info = system_info._replace(
    initial_densities=initial_densities,
    initial_spin_densities=initial_spin_densities)

  # use LDA functional
  xc_energy_density_fn = tree_util.Partial(
    xc.get_unpolarized_lda_xc_energy_density_fn(), params=None)

  for i in range(len(system_info.num_electrons)):
    
    lda_ksdft_scf = scf.kohn_sham(
        num_electrons=system_info.num_electrons[i],
        num_iterations=30,
        grids=grids,
        xc_energy_density_fn=xc_energy_density_fn,
        interaction_fn=utils.exponential_coulomb,
        initial_density=system_info.initial_densities[i],
        alpha=0.5,
        alpha_decay=0.9,
        enforce_reflection_symmetry=False,
        num_mixing_iterations=1,
        density_mse_converge_tolerance=-1,

        # scf
        locations=system_info.locations[i],
        nuclear_charges=system_info.nuclear_charges[i],
        )


    lda_ksdft_jit_scf = jit_scf.kohn_sham(
        num_electrons=system_info.num_electrons[i],
        num_iterations=30,
        grids=grids,
        xc_energy_density_fn=xc_energy_density_fn,
        interaction_fn=utils.exponential_coulomb,
        initial_density=system_info.initial_densities[i],
        alpha=0.5,
        alpha_decay=0.9,
        enforce_reflection_symmetry=False,
        num_mixing_iterations=1,
        density_mse_converge_tolerance=-1,
        
        # jit scf
        external_potential=system_info.external_potential[i],
        num_unpaired_electrons=system_info.num_unpaired_electrons[i],
        initial_spin_density=system_info.initial_spin_densities[i],
        )

    print(f'Example system: {symbols[i]}')
    states = {"scf result: ": lda_ksdft_scf, "jit_scf result: ": lda_ksdft_jit_scf}
    for label, state in states.items():
        total_energy = state.total_energy[-1].block_until_ready()
        kinetic_energy = state.kinetic_energy[-1]
        xc_energy = state.xc_energy[-1]

        print(label)
        print(f'E = {total_energy}')
        print(f'E_xc = {xc_energy}')

        # check against reference values
        np.testing.assert_allclose(system_info.total_energy[i], total_energy, rtol=0, atol=10**(-3))
        np.testing.assert_allclose(system_info.xc_energy[i], xc_energy, rtol=0, atol=10**(-3))
        print('(match ref. values)')



if __name__ == '__main__':
  lsda_scf_example()
  lda_scf_example()

