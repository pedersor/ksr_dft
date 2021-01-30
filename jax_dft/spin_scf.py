import functools
import typing
from typing import Optional, Union

import jax
from jax import tree_util
import jax.numpy as jnp
from jax_dft import utils
from jax_dft import scf

# testing: to delete
import matplotlib.pyplot as plt
import sys
import time

ArrayLike = Union[float, bool, jnp.ndarray]


@functools.partial(jax.jit)
def _wavefunctions_to_density(num_electrons, wavefunctions, grids):
  """Converts wavefunctions to density."""
  # create one hot-type vector to retrieve relevant lowest eigenvectors
  counts = jnp.arange(len(grids))
  one_hot = jnp.where(counts < num_electrons,
                      1.0, 0.0)
  one_hot = jnp.expand_dims(one_hot, axis=1)
  # Normalize the wavefunctions.
  wavefunctions = wavefunctions / jnp.sqrt(jnp.sum(
    wavefunctions ** 2, axis=1, keepdims=True) * utils.get_dx(grids))
  wavefunctions = wavefunctions * one_hot
  # Each eigenstate has spin up and spin down.
  return jnp.sum(wavefunctions ** 2, axis=0)


def wavefunctions_to_density(num_electrons, wavefunctions, grids):
  """Converts wavefunctions to density.

  Note each eigenstate contains two states: spin up and spin down.

  Args:
    num_electrons: Integer, the number of electrons in the system. The first
        num_electrons states are occupid.
    wavefunctions: Float numpy array with shape (num_eigen_states, num_grids).
    grids: Float numpy array with shape (num_grids,).

  Returns:
    Float numpy array with shape (num_grids,).
  """
  return _wavefunctions_to_density(num_electrons, wavefunctions, grids)


def get_total_eigen_energies(num_electrons, eigen_energies):
  """Gets the total eigen energies of the first num_electrons states.

  Args:
    num_electrons: Integer, the number of electrons in the system. The first
        num_electrons states are occupid.
    eigen_energies: Float numpy array with shape (num_eigen_states,).

  Returns:
    Float.
  """
  # create one hot-type vector to retrieve relevant lowest eigen_energies
  counts = jnp.arange(len(eigen_energies))
  one_hot = jnp.where(counts < num_electrons, 1.0, 0.0)
  return jnp.sum(one_hot * eigen_energies)


@functools.partial(jax.jit)
def _solve_noninteracting_system(external_potential, num_electrons, grids):
  """Solves noninteracting system."""
  eigen_energies, wavefunctions_transpose = jnp.linalg.eigh(
    # Hamiltonian matrix.
    scf.get_kinetic_matrix(grids) + jnp.diag(external_potential))
  density = wavefunctions_to_density(
    num_electrons, jnp.transpose(wavefunctions_transpose), grids)
  total_eigen_energies = get_total_eigen_energies(
    num_electrons=num_electrons, eigen_energies=eigen_energies)
  return density, total_eigen_energies


def solve_noninteracting_system(external_potential, num_electrons, grids):
  """Solves noninteracting system.

  Args:
    external_potential: Float numpy array with shape (num_grids,).
    num_electrons: Integer, the number of electrons in the system. The first
        num_electrons states are occupid.
    grids: Float numpy array with shape (num_grids,).

  Returns:
    density: Float numpy array with shape (num_grids,).
        The ground state density.
    total_eigen_energies: Float, the total energy of the eigen states.
  """
  return _solve_noninteracting_system(external_potential, num_electrons, grids)


def batch_solve_noninteracting_system(external_potential, num_electrons, grids):
  return jax.vmap(solve_noninteracting_system, in_axes=(0, 0, None),
                  out_axes=(0))(external_potential, num_electrons, grids)


def get_xc_energy_sigma(densities, xc_energy_density_fn, grids):
  r"""Gets xc energy.

  E_xc = \int density * xc_energy_density_fn(density) dx.

  Args:
    density: Float numpy array with shape (num_grids,).
    xc_energy_density_fn: function takes density and returns float numpy array
        with shape (num_grids,).
    grids: Float numpy array with shape (num_grids,).

  Returns:
    Float.
  """
  density_up, density_down = densities
  density = density_up + density_down
  spin_density = density_up - density_down
  # spin_density = jnp.zeros(len(density))

  return jnp.dot(
    0.5*(xc_energy_density_fn(density, spin_density=spin_density)
        + xc_energy_density_fn(density, spin_density=-1.*spin_density))
         , density) * utils.get_dx(grids)


@functools.partial(jax.jit, static_argnums=1)
def get_xc_potential_sigma(densities, xc_energy_density_fn, grids):
  """Gets xc potential.

  The xc potential is derived from xc_energy_density through automatic
  differentiation.

  Args:
    grad_argnum:
    density_up: Float numpy array with shape (num_grids,).
    density_down: Float numpy array with shape (num_grids,).
    xc_energy_density_fn: function takes density and returns float numpy array
        with shape (num_grids,).
    grids: Float numpy array with shape (num_grids,).

  Returns:
    Float numpy array with shape (num_grids,).
  """

  return jax.grad(get_xc_energy_sigma)(
    densities, xc_energy_density_fn, grids)


def _flip_and_average_fn(fn, locations, grids):
  """Flips and averages a function at the center of the locations."""

  def output_fn(array):
    output_array = utils.flip_and_average(
      locations=locations, grids=grids, array=array)
    return utils.flip_and_average(
      locations=locations, grids=grids, array=fn(output_array))

  return output_fn


def kohn_sham_iteration(
    state,
    xc_energy_density_fn,
    interaction_fn,
    enforce_reflection_symmetry):
  """One iteration of Kohn-Sham calculation.

  Note xc_energy_density_fn must be wrapped by jax.tree_util.Partial so this
  function can take a callable. When the arguments of this callable changes,
  e.g. the parameters of the neural network, kohn_sham_iteration() will not be
  recompiled.

  Args:
    state: KohnShamState.
    xc_energy_density_fn: function takes density (num_grids,) and returns
        the energy density (num_grids,).
    interaction_fn: function takes displacements and returns
        float numpy array with the same shape of displacements.
    enforce_reflection_symmetry: Boolean, whether to enforce reflection
        symmetry. If True, the system are symmetric respecting to the center.

  Returns:
    KohnShamState, the next state of Kohn-Sham iteration.
  """
  # TODO: enforce_reflection_symmetry for spin?

  hartree_potential = scf.get_hartree_potential(
    density=state.density,
    grids=state.grids,
    interaction_fn=interaction_fn)

  num_down_electrons = (state.num_electrons - state.num_unpaired_electrons) // 2
  num_up_electrons = num_down_electrons + state.num_unpaired_electrons
  density_up = (state.density + state.spin_density) / 2
  density_down = (state.density - state.spin_density) / 2
  densities = (density_up, density_down)

  xc_potential_up, xc_potential_down = get_xc_potential_sigma(densities,
                                                              xc_energy_density_fn,
                                                              state.grids)
  xc_potential_up = jnp.nan_to_num(xc_potential_up) / utils.get_dx(state.grids)
  xc_potential_down = jnp.nan_to_num(
    xc_potential_down) / utils.get_dx(state.grids)

  ks_potentials_sigma = jnp.array(
    [hartree_potential + xc_potential_up + state.external_potential,
     hartree_potential + xc_potential_down + state.external_potential])
  num_electrons_sigma = jnp.array([num_up_electrons, num_down_electrons])

  densities, total_eigen_energies_sigma = (
    batch_solve_noninteracting_system(ks_potentials_sigma,
                                      num_electrons_sigma, state.grids))
  densities = tuple(densities)

  density = 0
  spin_density = 0
  kinetic_energy = 0
  for i, (density_sigma, total_eigen_energy_sigma,
          ks_potential_sigma) in enumerate(zip(densities,
                                               total_eigen_energies_sigma,
                                               ks_potentials_sigma)):
    density += density_sigma
    spin_density += (-1) ** (i) * density_sigma
    kinetic_energy += (total_eigen_energy_sigma -
                       scf.get_external_potential_energy(
                         external_potential=ks_potential_sigma,
                         density=density_sigma,
                         grids=state.grids))

  xc_energy_density = xc_energy_density_fn(density, spin_density)

  # xc energy
  xc_energy = get_xc_energy_sigma(
    densities,
    xc_energy_density_fn=xc_energy_density_fn,
    grids=state.grids)

  total_energy = (
    # kinetic energy
      kinetic_energy
      # Hartree energy
      + scf.get_hartree_energy(
    density=density,
    grids=state.grids,
    interaction_fn=interaction_fn)
      # xc energy
      + xc_energy
      # external energy
      + scf.get_external_potential_energy(
    external_potential=state.external_potential,
    density=density,
    grids=state.grids)
  )

  # TODO: enforce_reflection_symmetry for spin

  return state._replace(
    density=density,
    spin_density=spin_density,
    total_energy=total_energy,
    hartree_potential=hartree_potential,
    xc_energy=xc_energy,
    kinetic_energy=kinetic_energy,
    xc_energy_density=xc_energy_density)


def kohn_sham(
    locations,
    nuclear_charges,
    num_electrons,
    num_unpaired_electrons,
    num_iterations,
    grids,
    xc_energy_density_fn,
    interaction_fn,
    initial_density=None,
    initial_spin_density=None,
    alpha=0.5,
    alpha_decay=0.9,
    enforce_reflection_symmetry=False,
    num_mixing_iterations=2,
    density_mse_converge_tolerance=-1.):
  """Runs Kohn-Sham to solve ground state of external potential.

  Args:
    locations: Float numpy array with shape (num_nuclei,), the locations of
        atoms.
    nuclear_charges: Float numpy array with shape (num_nuclei,), the nuclear
        charges.
    num_electrons: Integer, the number of electrons in the system. The first
        num_electrons states are occupied.
    num_unpaired_electrons: Integer, the number of unpaired electrons in the
        system. All unpaired electrons are defaulted to spin `up` by convention.
    num_iterations: Integer, the number of Kohn-Sham iterations.
    grids: Float numpy array with shape (num_grids,).
    xc_energy_density_fn: function takes density (num_grids,) and returns
        the energy density (num_grids,).
    interaction_fn: function takes displacements and returns
        float numpy array with the same shape of displacements.
    initial_density: Float numpy array with shape (num_grids,), initial guess
        of the density for Kohn-Sham calculation. Default None, the initial
        density is non-interacting solution from the external_potential.
    alpha: Float between 0 and 1, density linear mixing factor, the fraction
        of the output of the k-th Kohn-Sham iteration.
        If 0, the input density to the k-th Kohn-Sham iteration is fed into
        the (k+1)-th iteration. The output of the k-th Kohn-Sham iteration is
        completely ignored.
        If 1, the output density from the k-th Kohn-Sham iteration is fed into
        the (k+1)-th iteration, equivalent to no density mixing.
    alpha_decay: Float between 0 and 1, the decay factor of alpha. The mixing
        factor after k-th iteration is alpha * alpha_decay ** k.
    enforce_reflection_symmetry: Boolean, whether to enforce reflection
        symmetry. If True, the density are symmetric respecting to the center.
    num_mixing_iterations: Integer, the number of density differences in the
        previous iterations to mix the density.
    density_mse_converge_tolerance: Float, the stopping criteria. When the MSE
        density difference between two iterations is smaller than this value,
        the Kohn Sham iterations finish. The outputs of the rest of the steps
        are padded by the output of the converged step. Set this value to
        negative to disable early stopping.

  Returns:
    KohnShamState, the states of all the Kohn-Sham iteration steps.
  """
  external_potential = utils.get_atomic_chain_potential(
    grids=grids,
    locations=locations,
    nuclear_charges=nuclear_charges,
    interaction_fn=interaction_fn)

  num_down_electrons = (num_electrons - num_unpaired_electrons) // 2
  num_up_electrons = num_down_electrons + num_unpaired_electrons

  if initial_density is None and initial_spin_density is None:
    # Use the non-interacting solution from the external_potential as initial
    # guess.
    densities, _ = batch_solve_noninteracting_system(
      jnp.array([external_potential, external_potential]),
      jnp.array([num_up_electrons, num_down_electrons]), grids)

    initial_density = jnp.sum(densities, axis=0)
    initial_spin_density = jnp.squeeze(-1*jnp.diff(densities, axis=0))

  # Create initial state.
  state = scf.KohnShamState(
    density=initial_density,
    spin_density=initial_spin_density,
    total_energy=jnp.inf,
    locations=locations,
    nuclear_charges=nuclear_charges,
    external_potential=external_potential,
    grids=grids,
    num_electrons=num_electrons,
    num_unpaired_electrons=num_unpaired_electrons)

  states = []
  differences = None
  converged = False
  for _ in range(num_iterations):
    if converged:
      states.append(state)
      continue

    old_state = state
    state = kohn_sham_iteration(
      state=old_state,
      xc_energy_density_fn=xc_energy_density_fn,
      interaction_fn=interaction_fn,
      enforce_reflection_symmetry=enforce_reflection_symmetry)
    density_difference = state.density - old_state.density
    if differences is None:
      differences = jnp.array([density_difference])
    else:
      differences = jnp.vstack([differences, density_difference])
    if jnp.mean(
        jnp.square(density_difference)) < density_mse_converge_tolerance:
      converged = True
    state = state._replace(converged=converged)
    # Density mixing.
    state = state._replace(
      density=old_state.density
              + alpha * jnp.mean(differences[-num_mixing_iterations:], axis=0))
    states.append(state)
    alpha *= alpha_decay

  return tree_util.tree_multimap(lambda *x: jnp.stack(x), *states)


def get_initial_density_sigma(state, method):
  """Gets initial density for Kohn-Sham calculation.

  Args:
    state: KohnShamState contains a batch of states.
    method: String, the density initialization method.

  Returns:
    Float numpy array with shape (batch_size, num_grids).

  Raises:
    ValueError: If the initialization method is not exact or noninteracting.
  """

  if method == 'exact':
    # TODO: implement
    raise NotImplementedError()
  elif method == 'noninteracting':
    num_down_electrons = (state.num_electrons -
                          state.num_unpaired_electrons) // 2
    num_up_electrons = num_down_electrons + state.num_unpaired_electrons

    # array contain pairs [num_up, num_down]
    num_up_down = jnp.transpose(
      jnp.array([num_up_electrons, num_down_electrons]))

    # external_potential pairs [external_potential, external_potential]
    external_potentials = jnp.expand_dims(state.external_potential, axis=1)
    external_potentials = jnp.repeat(external_potentials, repeats=2, axis=1)

    # get initial density and spin_density
    solve = jax.vmap(batch_solve_noninteracting_system,
                     in_axes=(0, 0, None))
    initial_up_down_densities, _ = solve(external_potentials, num_up_down,
                                         state.grids[0])
    initial_densities = jnp.sum(initial_up_down_densities, axis=1)
    initial_spin_densities = jnp.squeeze(
      -1 * jnp.diff(initial_up_down_densities, axis=1))
    if len(state.num_electrons) == 1:
      # expand back dimension (special case if num_states = 1)
      initial_spin_densities = initial_spin_densities[jnp.newaxis, :]

    return initial_densities, initial_spin_densities

  else:
    raise ValueError(f'Unknown initialization method {method}')
