from test_atoms import Test_atoms
import numpy as np


def round_to_print(to_print, round_to_dec=4):
  """Round a value to be printed in table."""
  if isinstance(to_print, float):
    rounded_to_print = format(to_print, '.' + str(round_to_dec) + 'f')
  elif isinstance(to_print, str):
    rounded_to_print = to_print
  else:
    rounded_to_print = format(float(to_print), '.' + str(round_to_dec) + 'f')

  return rounded_to_print


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


if __name__ == '__main__':
  path = '../data/ions/basic_all'
  two_electrons = Test_atoms(datasets_base_dir=path)
  dataset = two_electrons.get_complete_dataset(num_grids=513)

  # set ML model
  two_electrons.init_ksr_lda_model()

  # get KS parameters
  two_electrons.set_ks_parameters(
    # The number of Kohn-Sham iterations in training.
    num_iterations=15,
    # @The density linear mixing factor.
    alpha=0.5,
    # Decay factor of density linear mixing factor.
    alpha_decay=0.9,
    # Enforce reflection symmetry across the origin.
    enforce_reflection_symmetry=True,
    # The number of density differences in the previous iterations to mix the
    # density. Linear mixing is num_mixing_iterations = 1.
    num_mixing_iterations=1,
    # The stopping criteria of Kohn-Sham iteration on density.
    density_mse_converge_tolerance=-1.,
    # Apply stop gradient on the output state of this step and all steps
    # before. The first KS step is indexed as 0. Default -1, no stop gradient
    # is applied.
    stop_gradient_step=-1
  )

  # generate latex table
  latex_symbols = np.array(
    ['H', 'He$^+$', 'Li$^{++}$', 'Be$^{3+}$', 'He', 'Li$^+$',
     'Be$^{++}$', 'Li', 'Be$^+$', 'Be'])
  dataset.load_misc(attribute='latex_symbols', array=latex_symbols)

  # set test set
  to_test = None  # tests all in dataset
  mask = dataset.get_mask_atoms(to_test)
  test_dataset = dataset.get_subdataset(mask)

  # load optimal checkpoint params
  two_electrons.set_test_set(test_dataset)
  final_states = two_electrons.get_final_test_states(
    optimal_ckpt_path='optimal_ckpt.pkl')

  curr_num_electrons = -1
  total_energy_MAE = []
  xc_energy_MAE = []
  for i in range(len(test_dataset.latex_symbols)):
    if test_dataset.num_electrons[i] != curr_num_electrons:
      table_print(str(test_dataset.num_electrons[i]))
      curr_num_electrons = test_dataset.num_electrons[i]
    else:
      table_print(' ')

    # ion chemical symbols
    table_print(test_dataset.latex_symbols[i])

    # total energies
    ksr_lda_total_energy = final_states.total_energy[i]
    unif_gas_lda_total_energy = test_dataset.total_energies[i]
    total_energy_error = unif_gas_lda_total_energy - ksr_lda_total_energy
    total_energy_MAE.append(np.abs(total_energy_error))
    # rounding to 4 decimal places
    ksr_lda_total_energy = round_to_print(ksr_lda_total_energy)
    total_energy_error = round_to_print(total_energy_error)
    table_print(ksr_lda_total_energy + ' (' + total_energy_error + ')')

    # xc energies
    ksr_lda_xc_energy = final_states.xc_energy[i]
    unif_gas_lda_xc_energy = test_dataset.xc_energies[i]
    xc_energy_error = unif_gas_lda_xc_energy - ksr_lda_xc_energy
    xc_energy_MAE.append(np.abs(xc_energy_error))
    # rounding to 4 decimal places
    ksr_lda_xc_energy = round_to_print(ksr_lda_xc_energy)
    xc_energy_error = round_to_print(xc_energy_error)
    table_print(ksr_lda_xc_energy + ' (' + xc_energy_error + ')',
                last_in_row=True)

  # separate MAE results from upper table
  print('\hline')

  table_print('MAE')
  table_print(' ')  # spacer
  table_print(np.mean(total_energy_MAE))
  table_print(np.mean(xc_energy_MAE), last_in_row=True)
