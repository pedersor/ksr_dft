# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

# Lint as: python3
"""Loads dataset."""

import os

from absl import logging
import numpy as np
import glob

from ksr_dft import scf
from ksr_dft import utils

# pytype: disable=attribute-error

_TEST_DISTANCE_X100 = {
    'h2_plus':
        set([
            64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 248, 256,
            264, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448,
            464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656,
            672, 688, 704, 720, 736, 752, 768, 784, 800, 816, 832, 848
        ]),
    'h2':
        set([
            40, 56, 72, 88, 104, 120, 136, 152, 184, 200, 216, 232, 248, 264,
            280, 312, 328, 344, 360, 376, 392, 408, 424, 456, 472, 488, 504,
            520, 536, 568, 584, 600
        ]),
    'h4':
        set([
            104, 120, 136, 152, 168, 200, 216, 232, 248, 280, 296, 312, 344,
            360, 376, 392, 408, 424, 440, 456, 472, 488, 520, 536, 552, 568,
            584, 600
        ]),
    'h2_h2':
        set([
            16, 48, 80, 112, 144, 176, 208, 240, 272, 304, 336, 368, 400, 432,
            464, 496, 528, 560, 592, 624, 656, 688, 720, 752, 784, 816, 848,
            880, 912, 944, 976
        ]),
}


class Dataset(object):
  """Loads dataset from path.

  """

  def __init__(self,
               path=None,
               data=None,
               num_grids=None,
               interaction_fn=utils.exponential_coulomb,
               name=None):
    """Initializer.

    Args:
      path: String, the path to the data.
      data: Dict of numpy arrays containing
          * num_electrons, float scalar.
          * grids, float numpy array with shape (num_grids,).
          * locations, float numpy array with shape (num_samples, num_nuclei).
          * nuclear_charges, float numpy array with shape
                (num_samples, num_nuclei).
          * distances_x100, float numpy array with shape (num_samples,).
          * distances, float numpy array with shape (num_samples,).
          * total_energies, float numpy array with shape (num_samples,).
          * densities, float numpy array with shape (num_samples, num_grids).
          * external_potentials, float numpy array with shape
                (num_samples, num_grids).
      num_grids: Integer, specify the number of grids for the density and
          external potential. If None, the original grids size are used.
          Otherwise, the original grids are trimmed into num_grids grid points
          in the center.
      interaction_fn: function takes displacements and returns
          float numpy array with the same shape of displacements.
      name: String, the name of the dataset.

    Raises:
      ValueError: if path and data are both None.
    """
    if path is None and data is None:
      raise ValueError('path and data cannot both be None.')
    self.name = name
    # Load from path if data is not given in the input argument.
    if data is None and path is not None:
      data = self._load_from_path(path)
    for name, array in data.items():
      setattr(self, name, array)
    self.data = data
    self._set_num_grids(num_grids)
    self._check_num_electrons_consistency()
    self._check_external_potential_consistency(interaction_fn)
    self.total_num_samples = self.total_energies.shape[0]

  def _load_from_path(self, path):
    """Loads npy files from path.
    Note(pedersor): allow_pickle=True for the case of jagged arrays."""

    file_open = open
    data = {}
    files = glob.glob(os.path.join(path, '*.npy'))
    for file in files:
      base = os.path.basename(file)
      name = os.path.splitext(base)[0]
      with file_open(file, 'rb') as f:
        data[name] = np.load(f, allow_pickle=True)
    return data

  def load_misc(self, attribute, array=None, path=None, file=None):
    """Load miscellaneous quantities from file or array.
    E.g. exchange-correlation energies."""
    if array is not None:
      setattr(self, attribute, array)
      self.data[attribute] = array
    else:
      if file is None:
        file = attribute + '.npy'
      file_open = open
      with file_open(os.path.join(path, file), 'rb') as f:
        setattr(self, attribute, np.load(f))
        self.data[attribute] = np.load(f)
    return self

  def _set_num_grids(self, num_grids):
    """Sets number of grids and trim arrays with grids dimension."""
    # grids is 1d array.
    original_num_grids = len(self.grids)
    if num_grids is None:
      self.num_grids = original_num_grids
      logging.info('This dataset has %d grids.', self.num_grids)
    else:
      if num_grids > original_num_grids:
        raise ValueError('num_grids (%d) cannot be '
                         'greater than the original number of grids (%d).' %
                         (num_grids, original_num_grids))
      self.num_grids = num_grids
      diff = original_num_grids - num_grids
      if diff % 2:
        left_grids_removed = (diff - 1) // 2
        right_grids_removed = (diff + 1) // 2
      else:
        left_grids_removed = diff // 2
        right_grids_removed = diff // 2
      self.grids = self.grids[left_grids_removed:original_num_grids -
                              right_grids_removed]
      self.densities = self.densities[:, left_grids_removed:original_num_grids -
                                      right_grids_removed]
      self.external_potentials = self.external_potentials[:, left_grids_removed:
                                                          original_num_grids -
                                                          right_grids_removed]
      logging.info(
          'The original number of grids (%d) are trimmed into %d grids.',
          original_num_grids, self.num_grids)

  def _check_external_potential_consistency(self, interaction_fn):
    """Checks whether external_potential.npy is consistent with given
    nuclear_charges.npy and locations.npy file. """

    for external_potential, nuclear_charges, locations in zip(
        self.external_potentials, self.nuclear_charges, self.locations):
      if np.allclose(
          external_potential,
          utils.get_atomic_chain_potential(self.grids, locations,
                                           nuclear_charges, interaction_fn)):
        pass
      else:
        raise ValueError('external_potentials.npy is not consistent with given '
                         'nuclear_charges.npy and locations.npy')
    return True

  def _check_num_electrons_consistency(self):
    """Checks whether num_electrons.npy and densities.npy are consistent."""

    for num_electrons, density in zip(self.num_electrons, self.densities):
      if np.allclose(num_electrons, np.trapz(density, self.grids)):
        pass
      else:
        raise ValueError('num_electrons.npy is not consistent with given '
                         'densities.npy')
    return True

  def get_mask(self, selected_distance_x100=None):
    """Gets mask from distance_x100."""
    if selected_distance_x100 is None:
      mask = np.ones(self.total_num_samples, dtype=bool)
    else:
      selected_distance_x100 = set(selected_distance_x100)
      mask = np.array([
          distance in selected_distance_x100 for distance in self.distances_x100
      ])
      if len(selected_distance_x100) != np.sum(mask):
        raise ValueError(
            'selected_distance_x100 contains distance that is not in the '
            'dataset.')
    return mask

  def get_test_mask(self):
    """Gets mask for test set."""
    return self.get_mask(_TEST_DISTANCE_X100[self.name])

  def get_molecules(self, selected_distance_x100=None):
    """Selects molecules from list of integers."""
    mask = self.get_mask(selected_distance_x100)
    num_samples = np.sum(mask)

    if hasattr(self, 'num_unpaired_electrons'):
      num_unpaired_electrons = self.num_unpaired_electrons[mask]
    else:
      num_unpaired_electrons = np.repeat(None, repeats=num_samples)

    if hasattr(self, 'spin_densities'):
      spin_densities = self.spin_densities[mask]
    else:
      spin_densities = np.repeat(None, repeats=num_samples)

    return scf.KohnShamState(
        density=self.densities[mask],
        spin_density=spin_densities,
        total_energy=self.total_energies[mask],
        locations=self.locations[mask],
        nuclear_charges=self.nuclear_charges[mask],
        external_potential=self.external_potentials[mask],
        grids=np.tile(np.expand_dims(self.grids, axis=0),
                      reps=(num_samples, 1)),
        num_electrons=self.num_electrons[mask],
        num_unpaired_electrons=num_unpaired_electrons,
        converged=np.repeat(True, repeats=num_samples),
    )

  def get_subdataset(self,
                     mask=None,
                     exceptions={'grids'},
                     downsample_step=None):
    """Gets subdataset."""
    if mask is None:
      mask = np.ones(self.total_num_samples, dtype=bool)

    # downsample data, if specified.
    if downsample_step is not None:
      sample_mask = np.zeros(self.total_num_samples, dtype=bool)
      sample_mask[::downsample_step] = True
      mask = np.logical_and(mask, sample_mask)

    sub_data = {}
    for name, array in self.data.items():
      if name in exceptions:
        sub_data[name] = array
      else:
        sub_data[name] = array[mask]
    return Dataset(data=sub_data)

  def get_mask_ions(self, selected_ions=None):
    """Gets mask from selected_ions, a list of tuples corresponding to
    (nuclear charge, total num of electrons)."""
    if selected_ions is None:
      mask = np.ones(self.total_num_samples, dtype=bool)
    else:
      selected_ions = set(selected_ions)
      mask = np.array([
          (nuclear_charge[0], num_electron) in selected_ions
          for (nuclear_charge,
               num_electron) in zip(self.nuclear_charges, self.num_electrons)
      ])
      if len(selected_ions) != np.sum(mask):
        raise ValueError(
            'selected_ions contains (nuclear_charge, num_electron) that is not in'
            ' the dataset.')
    return mask

  def get_ions(self, selected_ions=None):
    """Gets atoms from selected_ions, a list of tuples corresponding to
    (nuclear charge, total num of electrons)."""
    mask = self.get_mask_ions(selected_ions)
    num_samples = np.sum(mask)

    if hasattr(self, 'num_unpaired_electrons'):
      num_unpaired_electrons = self.num_unpaired_electrons[mask]
    else:
      num_unpaired_electrons = np.repeat(None, repeats=num_samples)

    if hasattr(self, 'spin_densities'):
      spin_densities = self.spin_densities[mask]
    else:
      spin_densities = np.repeat(None, repeats=num_samples)

    return scf.KohnShamState(
        density=self.densities[mask],
        spin_density=spin_densities,
        total_energy=self.total_energies[mask],
        locations=self.locations[mask],
        nuclear_charges=self.nuclear_charges[mask],
        external_potential=self.external_potentials[mask],
        grids=np.tile(np.expand_dims(self.grids, axis=0),
                      reps=(num_samples, 1)),
        num_electrons=self.num_electrons[mask],
        num_unpaired_electrons=num_unpaired_electrons,
        converged=np.repeat(True, repeats=num_samples),
    )


def concatenate_kohn_sham_states(*ks_states):
  """Concatenate at least two ks state objects into a single ks state object."""

  density = np.concatenate([ks_state.density for ks_state in ks_states])
  total_energy = np.concatenate(
      [ks_state.total_energy for ks_state in ks_states])

  # support jagged arrays
  locations = [list(ks_state.locations) for ks_state in ks_states]
  locations = np.asarray(sum(locations, []))
  nuclear_charges = [list(ks_state.nuclear_charges) for ks_state in ks_states]
  nuclear_charges = np.asarray(sum(nuclear_charges, []))

  external_potential = np.concatenate(
      [ks_state.external_potential for ks_state in ks_states])
  grids = np.concatenate([ks_state.grids for ks_state in ks_states])
  num_electrons = np.concatenate(
      [ks_state.num_electrons for ks_state in ks_states])
  num_unpaired_electrons = np.concatenate(
      [ks_state.num_unpaired_electrons for ks_state in ks_states])
  converged = np.concatenate([ks_state.converged for ks_state in ks_states])

  return scf.KohnShamState(
      density=density,
      total_energy=total_energy,
      locations=locations,
      nuclear_charges=nuclear_charges,
      external_potential=external_potential,
      grids=grids,
      num_electrons=num_electrons,
      num_unpaired_electrons=num_unpaired_electrons,
      converged=converged,
  )
