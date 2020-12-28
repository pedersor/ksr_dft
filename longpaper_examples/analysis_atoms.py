from test_atoms import Test_atoms
from jax_dft import scf

""" TODO:

Train on He, validate on Li+
test on all else: Be++, Li, Be+, Be.

- total energy, xc energy table 
- xc density plots..
- density plots..


"""

if __name__ == '__main__':
  two_electrons = Test_atoms('../data/ions/num_electrons_2')
  two_electrons.get_complete_dataset(num_grids=513)

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

  # set test set
  to_test = [3, 4]
  two_electrons.set_test_set(selected_ions=to_test)

  # load optimal checkpoint params
  final_states = two_electrons.get_final_test_states(
    optimal_ckpt_path='optimal_ckpt.pkl')

  # test set converged total energies
  print(final_states.total_energy)
