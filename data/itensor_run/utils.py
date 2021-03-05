"""Utility functions."""

import numpy as np

EXPONENTIAL_COULOMB_AMPLITUDE = 1.071295
EXPONENTIAL_COULOMB_KAPPA = 1 / 2.385345


def get_dx(grids):
  """Gets the grid spacing from grids array.

  Args:
    grids: Float numpy array with shape (num_grids,).

  Returns:
    Float, grid spacing.

  Raises:
    ValueError: If grids.ndim is not 1.
  """
  if grids.ndim != 1:
    raise ValueError('grids.ndim is expected to be 1 but got %d' % grids.ndim)
  return (np.amax(grids) - np.amin(grids)) / (grids.size - 1)


def exponential_coulomb(displacements, amplitude=EXPONENTIAL_COULOMB_AMPLITUDE,
    kappa=EXPONENTIAL_COULOMB_KAPPA):
  """Exponential Coulomb interaction.

  v(x) = amplitude * exp(-abs(x) * kappa)

  1d interaction described in
  One-dimensional mimicking of electronic structure: The case for exponentials.
  Physical Review B 91.23 (2015): 235141.
  https://arxiv.org/pdf/1504.05620.pdf

  This potential is used in
  Pure density functional for strong correlation and the thermodynamic limit
  from machine learning.
  Physical Review B 94.24 (2016): 245129.
  https://arxiv.org/pdf/1609.03705.pdf

  Args:
    displacements: Float numpy array.
    amplitude: Float, parameter of exponential Coulomb interaction.
    kappa: Float, parameter of exponential Coulomb interaction.

  Returns:
    Float numpy array with the same shape of displacements.
  """
  return amplitude * np.exp(-np.abs(displacements) * kappa)


def get_atomic_chain_potential(grids, locations, nuclear_charges,
    interaction_fn):
  """Gets atomic chain potential.

  Args:
    grids: Float numpy array with shape (num_grids,).
    locations: Float numpy array with shape (num_nuclei,),
        the locations of the nuclei.
    nuclear_charges: Float numpy array with shape (num_nuclei,),
        the charges of nuclei.
    interaction_fn: function takes displacements and returns
        float numpy array with the same shape of displacements.

  Returns:
    Float numpy array with shape (num_grids,).

  Raises:
    ValueError: If grids.ndim, locations.ndim or nuclear_charges.ndim is not 1.
  """
  if grids.ndim != 1:
    raise ValueError('grids.ndim is expected to be 1 but got %d' % grids.ndim)
  if locations.ndim != 1:
    raise ValueError(
      'locations.ndim is expected to be 1 but got %d' % locations.ndim)
  if nuclear_charges.ndim != 1:
    raise ValueError(
      'nuclear_charges.ndim is expected to be 1 but got %d' % nuclear_charges.ndim)
  displacements = np.expand_dims(grids, axis=0) - np.expand_dims(locations,
    axis=1)
  return np.dot(-nuclear_charges, interaction_fn(displacements))


def get_unif_separated_nuclei_positions(grids, num_locations, separation):
  """Gets nuclei positions (locations.npy) for a given uniform separation. """

  num_grids = grids.shape[0]
  grids_center_idx = num_grids // 2
  sep_steps = int(round(float(separation / get_dx(grids))))

  # positions of nuclei
  nuclear_locations = []
  if num_locations % 2 == 0:
    if sep_steps % 2 == 0:
      init_left_nuclei = grids_center_idx - int(sep_steps / 2)
      init_right_nuclei = grids_center_idx + int(sep_steps / 2)
      nuclear_locations.append(grids[init_left_nuclei])
      nuclear_locations.append(grids[init_right_nuclei])

      for i in range(int(num_locations / 2) - 1):
        nuclear_locations.append(grids[init_right_nuclei + (i + 1) * sep_steps])
        nuclear_locations.append(grids[init_left_nuclei - (i + 1) * sep_steps])
    else:
      init_left_nuclei = grids_center_idx - int((sep_steps - 1) / 2)
      init_right_nuclei = grids_center_idx + int((sep_steps + 1) / 2)
      nuclear_locations.append(grids[init_left_nuclei])
      nuclear_locations.append(grids[init_right_nuclei])

      for i in range(int(num_locations / 2) - 1):
        nuclear_locations.append(grids[init_right_nuclei + (i + 1) * sep_steps])
        nuclear_locations.append(grids[init_left_nuclei - (i + 1) * sep_steps])
  else:
    # odd num_locations. First place nuclei in center.
    nuclear_locations.append(grids[grids_center_idx])
    for i in range(2, num_locations + 1):
      location = grids_center_idx + (-1) ** i * (i // 2) * sep_steps
      nuclear_locations.append(grids[location])

  nuclear_locations = np.asarray(nuclear_locations)
  nuclear_locations = np.sort(nuclear_locations)
  return nuclear_locations
