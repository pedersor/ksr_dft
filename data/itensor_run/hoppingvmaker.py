import numpy as np
from jax_dft import utils
from jax_dft import constants

import jax.numpy as jnp


def get_ham1c(grids, external_potential):
  """Gets Hamiltonian file. """

  N = len(grids)
  finite_diff_coeffs = np.array([30, -16, 1]) / 24 / (utils.get_dx(grids) ** 2)

  i_lst = []
  j_lst = []
  ham_lst = []
  for i in range(1, N + 1):
    for j in range(1, N + 1):
      if i == j:
        i_lst.append(i)
        j_lst.append(j)
        ham_lst.append(finite_diff_coeffs[0] + external_potential[i - 1])
      elif i - j == 1 or j - i == 1:
        i_lst.append(i)
        j_lst.append(j)
        ham_lst.append(finite_diff_coeffs[1])
      elif i - j == 2 or j - i == 2:
        i_lst.append(i)
        j_lst.append(j)
        ham_lst.append(finite_diff_coeffs[2])

  to_out = np.array([i_lst, j_lst, ham_lst])
  to_out = to_out.T
  with open('Ham1c', 'w') as datafile_id:
    # here you open the ascii file
    np.savetxt(datafile_id, to_out, fmt=['%i', '%i', '%.20f'],
      header=str(N) + '\n' + str(0), comments='')
    return


def exponential_coulomb(displacements,
                        amplitude=constants.EXPONENTIAL_COULOMB_AMPLITUDE,
                        kappa=constants.EXPONENTIAL_COULOMB_KAPPA):
  return amplitude * np.exp(-np.abs(displacements) * kappa)


def get_vuncomp(grids, interaction_fn=exponential_coulomb, lam=1):
  N = len(grids)

  i_lst = []
  j_lst = []
  vuncomp_lst = []
  for i in range(1, N + 1):
    for j in range(1, N + 1):
      vuncomp = -lam * interaction_fn(grids[i - 1] - grids[j - 1])
      i_lst.append(i)
      j_lst.append(j)
      vuncomp_lst.append(vuncomp)

  to_out = np.array([i_lst, j_lst, vuncomp_lst])
  to_out = to_out.T
  with open('Vuncomp', 'w') as datafile_id:
    # here you open the ascii file
    np.savetxt(datafile_id, to_out, fmt=['%i', '%i', '%.20f'], header=str(N),
      comments='')
    return


if __name__ == '__main__':
  import sys
  import numpy as np
  import matplotlib.pyplot as plt

  # LiH molcule example

  h = 0.08  # grid spacing
  grids = np.arange(-256, 257) * h

  locations = utils.get_unif_separated_nuclei_positions(grids, num_locations=2,
    separation=2)
  nuclear_charges = np.array([3, 1])

  external_potential = utils.get_atomic_chain_potential(grids=grids,
    locations=locations, nuclear_charges=nuclear_charges,
    interaction_fn=utils.exponential_coulomb)

  get_ham1c(grids, external_potential)
  get_vuncomp(grids)
