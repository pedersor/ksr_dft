import os
from train_validate_ions import Train_validate_ions
import numpy as np
import matplotlib.pyplot as plt


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


def get_ions_table_MAE(test_dataset, final_states):
  """Generate table of atoms and errors"""
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
    ksr_total_energy = final_states.total_energy[i]
    exact_total_energy = test_dataset.total_energies[i]
    total_energy_error = exact_total_energy - ksr_total_energy
    total_energy_MAE.append(np.abs(total_energy_error))
    # rounding to 4 decimal places
    ksr_total_energy = round_to_print(ksr_total_energy)
    total_energy_error = round_to_print(total_energy_error)
    table_print(ksr_total_energy + ' (' + total_energy_error + ')')

    # xc energies
    ksr_xc_energy = final_states.xc_energy[i]
    exact_xc_energy = test_dataset.xc_energies[i]
    xc_energy_error = exact_xc_energy - ksr_xc_energy
    xc_energy_MAE.append(np.abs(xc_energy_error))
    # rounding to 4 decimal places
    ksr_xc_energy = round_to_print(ksr_xc_energy)
    xc_energy_error = round_to_print(xc_energy_error)
    table_print(ksr_xc_energy + ' (' + xc_energy_error + ')',
                last_in_row=True)

  # separate MAE results from upper table
  print('\hline')

  table_print('MAE')
  table_print(' ')  # spacer
  table_print(np.mean(total_energy_MAE))
  table_print(np.mean(xc_energy_MAE), last_in_row=True)


def get_plots(test_dataset, final_states, model_dir):
  """Generate density and xc energy density plots on a single plt figure."""
  num_columns = len(test_dataset.latex_symbols) // 2
  _, axs = plt.subplots(
    nrows=2,
    ncols=num_columns,
    figsize=(3 * num_columns, 6), sharex=True, sharey=True)
  axs[-1][num_columns // 2].set_xlabel('x')
  for i, ax in enumerate(axs.ravel()):
    ax.set_title(test_dataset.latex_symbols[i])
    ax.plot(final_states.grids[i], final_states.density[i],
            label=r'$n^{\mathrm{KSR-LDA}}$')
    ax.plot(test_dataset.grids, test_dataset.densities[i], 'k--',
            label=r'$n^{\mathrm{LDA}}$')
    ax.plot(final_states.grids[i], final_states.xc_energy_density[i],
            label=r'$\epsilon^{\mathrm{KSR-LDA}}_{\mathrm{XC}}$')
    ax.plot(test_dataset.grids, test_dataset.xc_energy_densities[i], 'g--',
            label=r'$\epsilon^{\mathrm{LDA}}_{\mathrm{XC}}$')
    ax.set_xlim(-10, 10)
  axs[-1][-1].legend(bbox_to_anchor=(1.2, 0.8))
  plt.savefig(os.path.join(model_dir, 'xc_energy_densities_plots.pdf'),
              bbox_inches='tight')


if __name__ == '__main__':
  path = '../data/ions/unpol_lda/basic_all'
  ions = Train_validate_ions(datasets_base_dir=path)
  dataset = ions.get_complete_dataset(num_grids=513)

  # set ML model for xc functional
  model_dir = '../models/ions/unpol_lda'
  ions.init_ksr_lda_model(model_dir=model_dir)

  # get KS parameters
  ions.set_ks_params(
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

  latex_symbols = np.array(
    ['H', 'He$^+$', 'Li$^{++}$', 'Be$^{3+}$', 'He', 'Li$^+$',
     'Be$^{++}$', 'Li', 'Be$^+$', 'Be'])
  dataset.load_misc(attribute='latex_symbols', array=latex_symbols)

  # set test set
  to_test = None  # tests all in dataset
  mask = dataset.get_mask_ions(to_test)
  test_dataset = dataset.get_subdataset(mask)

  # load optimal checkpoint params
  ions.set_test_set(test_dataset)
  final_states = ions.get_final_test_states(
    optimal_ckpt_path=os.path.join(model_dir, 'optimal_ckpt.pkl'))

  # generate latex table with MAE
  get_ions_table_MAE(test_dataset, final_states)

  # generate density and xc energy density plots
  get_plots(test_dataset, final_states, model_dir)
