import numpy as np
from ksr_dft import utils


def round_to_print(to_print, round_to_dec=4):
  """Round a value to be printed in table."""
  if isinstance(to_print, str):
    rounded_to_print = to_print
  else:
    rounded_to_print = format(float(to_print), f'.{round_to_dec}f')

  return rounded_to_print


def scientific_round_to_print(to_print, round_to_dec=1):
  """Round to print but with scientific notation."""
  return format(float(to_print), f'.{round_to_dec}e')


def table_print(to_print, round_to_dec=4, last_in_row=False):
  """Print table line by line in latex form. """

  to_print = round_to_print(to_print, round_to_dec)

  if last_in_row:
    end = ' '
    print(to_print, end=end)
    print(r'\\')
    print('\hline')
  else:
    end = ' & '
    print(to_print, end=end)


def get_error_table(test_dataset, final_states):
  """Generate table of energy errors and density losses."""

  energy_error_lst = []
  energy_abs_error_lst = []
  density_loss_lst = []
  for i in range(len(test_dataset.total_energies)):

    # total energies
    ksr_total_energy = final_states.total_energy[i]
    exact_total_energy = test_dataset.total_energies[i]
    total_energy_error = ksr_total_energy - exact_total_energy

    energy_error_lst.append(total_energy_error)
    energy_abs_error_lst.append(np.abs(total_energy_error))

    # densities
    ksr_density = final_states.density[i]
    exact_density = test_dataset.densities[i]
    density_loss = np.mean((ksr_density - exact_density)**2 /
                           test_dataset.num_electrons[i]) * utils.get_dx(
                               test_dataset.grids) * len(test_dataset.grids)
    density_loss_lst.append(density_loss)

    # print table:

    # chemical symbols
    table_print(test_dataset.latex_symbols[i])

    # rounding to 4 decimal places
    ksr_total_energy = round_to_print(ksr_total_energy)
    total_energy_error = round_to_print(total_energy_error)
    table_print(ksr_total_energy + ' (' + total_energy_error + ')')
    table_print(scientific_round_to_print(density_loss), last_in_row=True)

  return (np.asarray(energy_error_lst), np.asarray(energy_abs_error_lst),
          np.asarray(density_loss_lst))


def get_ip_table(test_dataset,
                 final_states,
                 ions_idx_to_compare=np.array(
                     [-1, -1, -1, -1, 1, 2, 3, 5, 6, 8])):
  """Generate IP table with errors."""

  abs_error_lst = []
  error_lst = []
  for i in range(len(test_dataset.total_energies)):

    if ions_idx_to_compare[i] == -1:
      # single electron cases
      ksr_ip = np.abs(final_states.total_energy[i])
      exact_ip = np.abs(test_dataset.total_energies[i])
    else:
      idx = ions_idx_to_compare[i]
      ksr_ip = np.abs(final_states.total_energy[i] -
                      final_states.total_energy[idx])
      exact_ip = np.abs(test_dataset.total_energies[i] -
                        test_dataset.total_energies[idx])

    error = ksr_ip - exact_ip
    error_lst.append(error)
    abs_error_lst.append(np.abs(error))

    # table prints
    ksr_ip = round_to_print(ksr_ip)
    error = round_to_print(error)
    table_print(test_dataset.latex_symbols[i])
    table_print(f'{ksr_ip} ({error})', last_in_row=True)

  return np.mean(np.asarray(error_lst)), np.mean(np.asarray(abs_error_lst))


def get_total_separated_ions_energy_lst(molecules_dataset,
                                        ions_dataset,
                                        ions_final_states='exact'):
  """Gets the total energy of separated ions in molecule to be used in AE
  calculations. """

  total_separated_ions_energy_lst = []
  for ion_multiset in molecules_dataset.dissocation_info:

    total_separated_ions_energy = 0
    possible_ions_lst = [(nuc_charge[0], num_el) for nuc_charge, num_el in zip(
        ions_dataset.nuclear_charges, ions_dataset.num_electrons)]

    for key, multiplicity in ion_multiset.items():
      idx = possible_ions_lst.index(key)
      if ions_final_states == 'exact':
        total_separated_ions_energy += multiplicity * ions_dataset.total_energies[
            idx]
      else:
        total_separated_ions_energy += multiplicity * ions_final_states.total_energy[
            idx]

    total_separated_ions_energy_lst.append(total_separated_ions_energy)

  return total_separated_ions_energy_lst


def get_ae_table(molecules_final_states, molecules_dataset, ions_dataset,
                 ions_final_states):
  """Generate AE table with errors for molecules. """

  exact_total_separated_ions_energy_lst = get_total_separated_ions_energy_lst(
      molecules_dataset, ions_dataset, ions_final_states='exact')
  ksr_total_separated_ions_energy_lst = get_total_separated_ions_energy_lst(
      molecules_dataset, ions_dataset, ions_final_states)

  abs_error_lst = []
  error_lst = []
  for i in range(len(molecules_dataset.total_energies)):

    nuclear_energy = utils.get_nuclear_interaction_energy(
        molecules_dataset.locations[i], molecules_dataset.nuclear_charges[i],
        utils.exponential_coulomb)

    exact_ae = np.abs(nuclear_energy + molecules_dataset.total_energies[i] -
                      exact_total_separated_ions_energy_lst[i])
    ksr_ae = np.abs(nuclear_energy + molecules_final_states.total_energy[i] -
                    ksr_total_separated_ions_energy_lst[i])

    error = ksr_ae - exact_ae
    error_lst.append(error)
    abs_error_lst.append(np.abs(error))

    # table prints
    ksr_ae = round_to_print(ksr_ae)
    error = round_to_print(error)
    table_print(molecules_dataset.latex_symbols[i])
    table_print(f'{ksr_ae} ({error})', last_in_row=True)

  return np.mean(np.asarray(error_lst)), np.mean(np.asarray(abs_error_lst))
