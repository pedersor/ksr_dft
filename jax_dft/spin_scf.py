import functools
import typing
from typing import Optional, Union

import jax
from jax import tree_util
import jax.numpy as jnp
from jax_dft import utils
from jax_dft import scf

ArrayLike = Union[float, bool, jnp.ndarray]


def get_xc_energy(density_up, density_down, xc_energy_density_fn, grids):
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
  density = density_up + density_down
  #spin_density = jnp.where(density == 0., 0.0,
  #  jnp.absolute(density_up - density_down)/ density)

  #spin_density = density_up - density_down
  spin_density = jnp.zeros(len(density))


  return jnp.dot(xc_energy_density_fn(density, spin_density), density) * utils.get_dx(grids)


def get_xc_potential_up(density_up, density_down, xc_energy_density_fn, grids):
  """Gets xc potential.

  The xc potential is derived from xc_energy_density through automatic
  differentiation.

  Args:
    density: Float numpy array with shape (num_grids,).
    xc_energy_density_fn: function takes density and returns float numpy array
        with shape (num_grids,).
    grids: Float numpy array with shape (num_grids,).

  Returns:
    Float numpy array with shape (num_grids,).
  """
  return jax.grad(get_xc_energy, argnums=0)(
    density_up, density_down, xc_energy_density_fn, grids) / utils.get_dx(grids)


def get_xc_potential_down(density_up, density_down, xc_energy_density_fn, grids):
  """Gets xc potential.

  The xc potential is derived from xc_energy_density through automatic
  differentiation.

  Args:
    density: Float numpy array with shape (num_grids,).
    xc_energy_density_fn: function takes density and returns float numpy array
        with shape (num_grids,).
    grids: Float numpy array with shape (num_grids,).

  Returns:
    Float numpy array with shape (num_grids,).
  """
  return jax.grad(get_xc_energy, argnums=1)(
    density_up, density_down, xc_energy_density_fn, grids) / utils.get_dx(grids)


class KohnShamState(typing.NamedTuple):
  """A namedtuple containing the state of an Kohn-Sham iteration.

  Attributes:
    density: A float numpy array with shape (num_grids,).
    total_energy: Float, the total energy of Kohn-Sham calculation.
    locations: A float numpy array with shape (num_nuclei,).
    nuclear_charges: A float numpy array with shape (num_nuclei,).
    external_potential: A float numpy array with shape (num_grids,).
    grids: A float numpy array with shape (num_grids,).
    num_electrons: Integer, the number of electrons in the system. The first
        num_electrons states are occupied.
    hartree_potential: A float numpy array with shape (num_grids,).
    xc_potential: A float numpy array with shape (num_grids,).
    xc_energy_density: A float numpy array with shape (num_grids,).
    gap: Float, the Kohn-Sham gap.
    converged: Boolean, whether the state is converged.
  """

  density: jnp.ndarray
  spin_density: jnp.ndarray
  total_energy: ArrayLike
  locations: jnp.ndarray
  nuclear_charges: jnp.ndarray
  external_potential: jnp.ndarray
  grids: jnp.ndarray
  num_electrons: ArrayLike
  num_unpaired_electrons: ArrayLike
  initial_densities: Optional[jnp.ndarray] = None
  xc_energy: Optional[ArrayLike] = None
  kinetic_energy: Optional[ArrayLike] = None
  hartree_potential: Optional[jnp.ndarray] = None
  xc_potential: Optional[jnp.ndarray] = None
  xc_energy_density: Optional[jnp.ndarray] = None
  gap: Optional[ArrayLike] = None
  converged: Optional[ArrayLike] = False


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
    num_electrons,
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
    num_electrons: Integer, the number of electrons in the system. The first
        num_electrons states are occupid.
    xc_energy_density_fn: function takes density (num_grids,) and returns
        the energy density (num_grids,).
    interaction_fn: function takes displacements and returns
        float numpy array with the same shape of displacements.
    enforce_reflection_symmetry: Boolean, whether to enforce reflection
        symmetry. If True, the system are symmetric respecting to the center.

  Returns:
    KohnShamState, the next state of Kohn-Sham iteration.
  """
  # TODO: enforce_reflection_symmetry for spin

  hartree_potential = scf.get_hartree_potential(
    density=state.density,
    grids=state.grids,
    interaction_fn=interaction_fn)
  xc_potential = get_xc_potential(
    density=state.density,
    xc_energy_density_fn=xc_energy_density_fn,
    grids=state.grids)
  ks_potential = hartree_potential + xc_potential + state.external_potential
  xc_energy_density = xc_energy_density_fn(state.density)

  # Solve Kohn-Sham equation.
  density, total_eigen_energies, gap = solve_noninteracting_system(
    external_potential=ks_potential,
    num_electrons=num_electrons,
    grids=state.grids)

  # KS kinetic energy = total_eigen_energies - external_potential_energy
  kinetic_energy = total_eigen_energies - scf.get_external_potential_energy(
    external_potential=ks_potential,
    density=density,
    grids=state.grids)

  # xc energy
  xc_energy = get_xc_energy(
    density=density,
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
    total_energy=total_energy,
    hartree_potential=hartree_potential,
    xc_potential=xc_potential,
    xc_energy=xc_energy,
    kinetic_energy=kinetic_energy,
    xc_energy_density=xc_energy_density,
    gap=gap)


def kohn_sham(
    locations,
    nuclear_charges,
    num_electrons,
    num_iterations,
    grids,
    xc_energy_density_fn,
    interaction_fn,
    initial_density=None,
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
        num_electrons states are occupid.
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
  if initial_density is None:
    # Use the non-interacting solution from the external_potential as initial
    # guess.
    initial_density, _, _ = solve_noninteracting_system(
      external_potential=external_potential,
      num_electrons=num_electrons,
      grids=grids)
  # Create initial state.
  state = KohnShamState(
    density=initial_density,
    total_energy=jnp.inf,
    locations=locations,
    nuclear_charges=nuclear_charges,
    external_potential=external_potential,
    grids=grids,
    num_electrons=num_electrons)
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
      num_electrons=num_electrons,
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
