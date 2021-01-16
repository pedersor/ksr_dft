import functools
import typing
from typing import Optional, Union

import jax
from jax import tree_util
import jax.numpy as jnp
from jax_dft import utils

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