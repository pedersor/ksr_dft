# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for self-consistent field calculation simplified for jit."""

import functools

import jax
import jax.numpy as jnp

from jax_dft import scf
from jax_dft import utils
from jax_dft import spin_scf

# testing: to delete
import matplotlib.pyplot as plt
import sys
import time

def _flip_and_average_on_center(array):
  """Flips and averages array on the center."""
  return (array + jnp.flip(array)) / 2


def _flip_and_average_on_center_fn(fn):
  """Flips and averages a function on the center."""
  def averaged_fn(array):
    return _flip_and_average_on_center(fn(_flip_and_average_on_center(array)))
  return averaged_fn


def _connection_weights(num_iterations, num_mixing_iterations):
  """Gets the connection weights."""
  mask = jnp.triu(
      jnp.tril(jnp.ones((num_iterations, num_iterations))),
      k=-num_mixing_iterations + 1)
  return mask / jnp.sum(mask, axis=1, keepdims=True)


@functools.partial(jax.jit, static_argnums=(7, 8))
def _kohn_sham_iteration(
    density,
    spin_density,
    external_potential,
    grids,
    num_electrons,
    num_unpaired_electrons,
    xc_energy_density_fn,
    interaction_fn,
    enforce_reflection_symmetry):
  """One iteration of Kohn-Sham calculation."""
  # NOTE(leeley): Since num_electrons in KohnShamState need to specify as
  # static argument in jit function, this function can not directly take
  # KohnShamState as input arguments. The related attributes in KohnShamState
  # are used as input arguments for this helper function.

  #TODO: enforce_reflection_symmetry?

  hartree_potential = scf.get_hartree_potential(
      density=density,
      grids=grids,
      interaction_fn=interaction_fn)

  num_down_electrons = (num_electrons - num_unpaired_electrons) // 2
  num_up_electrons = num_down_electrons + num_unpaired_electrons
  density_up = (density + spin_density) / 2
  density_down = (density - spin_density) / 2
  densities = (density_up, density_down)

  xc_potential_up, xc_potential_down = spin_scf.get_xc_potential_sigma(
    densities, xc_energy_density_fn, grids)
  xc_potential_up = jnp.nan_to_num(xc_potential_up) / utils.get_dx(grids)
  xc_potential_down = jnp.nan_to_num(
    xc_potential_down) / utils.get_dx(grids)

  ks_potentials_sigma = jnp.array(
    [hartree_potential + xc_potential_up + external_potential,
     hartree_potential + xc_potential_down + external_potential])
  num_electrons_sigma = jnp.array([num_up_electrons, num_down_electrons])

  densities, total_eigen_energies_sigma = (
    spin_scf.batch_solve_noninteracting_system(ks_potentials_sigma,
      num_electrons_sigma, grids))


  # new density and spin density
  density = jnp.sum(densities, axis=0)
  spin_density = jnp.squeeze(-1*jnp.diff(densities, axis=0))


  # KS kinetic energy = total_eigen_energies - external_potential_energy
  kinetic_energy_up = (total_eigen_energies_sigma[0] -
                     scf.get_external_potential_energy(
                       external_potential=ks_potentials_sigma[0],
                       density=densities[0],
                       grids=grids))
  kinetic_energy_down = (total_eigen_energies_sigma[1] -
                       scf.get_external_potential_energy(
                         external_potential=ks_potentials_sigma[1],
                         density=densities[1],
                         grids=grids))
  kinetic_energy = kinetic_energy_up + kinetic_energy_down


  xc_energy_density = 0.5*(
      xc_energy_density_fn(density, spin_density=spin_density)
    + xc_energy_density_fn(density, spin_density=-1.*spin_density))

  # xc energy
  xc_energy = spin_scf.get_xc_energy_sigma(
    tuple(densities),
    xc_energy_density_fn=xc_energy_density_fn,
    grids=grids)

  total_energy = (
      # kinetic energy
      kinetic_energy
      # Hartree energy
      + scf.get_hartree_energy(
          density=density,
          grids=grids,
          interaction_fn=interaction_fn)
      # xc energy
      + xc_energy
      # external energy
      + scf.get_external_potential_energy(
          external_potential=external_potential,
          density=density,
          grids=grids)
      )

  if enforce_reflection_symmetry:
    density = _flip_and_average_on_center(density)

  return (
      density,
      spin_density,
      total_energy,
      kinetic_energy,
      xc_energy,
      hartree_potential,
      xc_energy_density)


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
  (
      density,
      spin_density,
      total_energy,
      kinetic_energy,
      xc_energy,
      hartree_potential,
      xc_energy_density) = _kohn_sham_iteration(
          state.density,
          state.spin_density,
          state.external_potential,
          state.grids,
          state.num_electrons,
          state.num_unpaired_electrons,
          xc_energy_density_fn,
          interaction_fn,
          enforce_reflection_symmetry)
  return state._replace(
      density=density,
      spin_density=spin_density,
      total_energy=total_energy,
      hartree_potential=hartree_potential,
      xc_energy=xc_energy,
      kinetic_energy=kinetic_energy,
      xc_energy_density=xc_energy_density)


@functools.partial(jax.jit, static_argnums=(3, 6, 9, 10, 11, 12, 13, 14))
def _kohn_sham(
    external_potential,
    num_electrons,
    num_unpaired_electrons,
    num_iterations,
    grids,
    xc_energy_density_fn,
    interaction_fn,
    initial_density,
    initial_spin_density,
    alpha,
    alpha_decay,
    enforce_reflection_symmetry,
    num_mixing_iterations,
    density_mse_converge_tolerance,
    stop_gradient_step):
  """Jit-able Kohn Sham calculation."""
  num_grids = grids.shape[0]
  weights = _connection_weights(num_iterations, num_mixing_iterations)

  def _converged_kohn_sham_iteration(old_state_differences):
    old_state, differences = old_state_differences
    return old_state._replace(converged=True), differences

  def _uncoveraged_kohn_sham_iteration(idx_old_state_alpha_differences):
    idx, old_state, alpha, differences = idx_old_state_alpha_differences
    state = kohn_sham_iteration(
        state=old_state,
        xc_energy_density_fn=xc_energy_density_fn,
        interaction_fn=interaction_fn,
        enforce_reflection_symmetry=enforce_reflection_symmetry)
    differences = jax.ops.index_update(
        differences, idx, state.density - old_state.density)
    # Density mixing.
    state = state._replace(
        density=old_state.density + alpha * jnp.dot(weights[idx], differences))
    return state, differences

  def _single_kohn_sham_iteration(carry, inputs):
    del inputs
    idx, old_state, alpha, converged, differences = carry
    state, differences = jax.lax.cond(
        converged,
        true_operand=(old_state, differences),
        true_fun=_converged_kohn_sham_iteration,
        false_operand=(idx, old_state, alpha, differences),
        false_fun=_uncoveraged_kohn_sham_iteration)
    converged = jnp.mean(jnp.square(
        state.density - old_state.density)) < density_mse_converge_tolerance
    state = jax.lax.cond(
        idx <= stop_gradient_step,
        true_fun=jax.lax.stop_gradient,
        false_fun=lambda x: x,
        operand=state)
    return (idx + 1, state, alpha * alpha_decay, converged, differences), state

  # Create initial state.
  state = scf.KohnShamState(
      density=initial_density,
      spin_density=initial_spin_density,
      total_energy=jnp.inf,
      external_potential=external_potential,
      grids=grids,
      num_electrons=num_electrons,
      num_unpaired_electrons=num_unpaired_electrons,
      # Add dummy fields so the input and output of lax.scan have the same type
      # structure.
      xc_energy=0.,
      kinetic_energy=0.,
      hartree_potential=jnp.zeros_like(grids),
      xc_energy_density=jnp.zeros_like(grids),
      converged=False)
  # Initialize the density differences with all zeros since the carry in
  # lax.scan must keep the same shape.
  differences = jnp.zeros((num_iterations, num_grids))

  _, states = jax.lax.scan(
      _single_kohn_sham_iteration,
      init=(0, state, alpha, state.converged, differences),
      xs=jnp.arange(num_iterations))
  return states


def kohn_sham(
    external_potential,
    num_electrons,
    num_unpaired_electrons,
    num_iterations,
    grids,
    xc_energy_density_fn,
    interaction_fn,
    initial_density,
    initial_spin_density,
    alpha=0.5,
    alpha_decay=0.9,
    enforce_reflection_symmetry=False,
    num_mixing_iterations=2,
    density_mse_converge_tolerance=-1.,
    stop_gradient_step=-1):
  """Jit-able Kohn Sham calculation.

  In order to make it jit-able. The following options are removed from
  kohn_sham():

    * There is no default initial density.
    * There is no convergence criteria and early stopping.
    * Reflection symmetry flip density at the center of the grids, not
        locations.

  Besides, the for loop is replaced by jax.lax.scan so it is jit friendly.
  Otherwise, jit on GPU runs into issues for big amount of for loop steps.

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
        of the density for Kohn-Sham calculation.
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
        symmetry.
    num_mixing_iterations: Integer, the number of density differences in the
        previous iterations to mix the density.
    density_mse_converge_tolerance: Float, the stopping criteria. When the MSE
        density difference between two iterations is smaller than this value,
        the Kohn Sham iterations finish. The outputs of the rest of the steps
        are padded by the output of the converged step. Set this value to
        negative to disable early stopping.
    stop_gradient_step: Integer, apply stop gradient on the output state of
        this step and all steps before. The first KS step is indexed as 0.

  Returns:
    KohnShamState, the states of all the Kohn-Sham iteration steps.
  """
  return _kohn_sham(
    external_potential,
    num_electrons,
    num_unpaired_electrons,
    num_iterations,
    grids,
    xc_energy_density_fn,
    interaction_fn,
    initial_density,
    initial_spin_density,
    alpha,
    alpha_decay,
    enforce_reflection_symmetry,
    num_mixing_iterations,
    density_mse_converge_tolerance,
    stop_gradient_step)
