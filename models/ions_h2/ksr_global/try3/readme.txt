# model dir to save results
model_dir = 'models/ions_h2/ksr_global/try4'

# datasets
ions_dataset = datasets.Dataset('data/ions/dmrg', num_grids=513)
h2_dataset = datasets.Dataset('data/molecules/h2', num_grids=513)
h3_dataset = datasets.Dataset('data/molecules/h3', num_grids=513)
h4_dataset = datasets.Dataset('data/molecules/h4', num_grids=513)
grids = ions_dataset.grids # same grids for all datasets...

# training sets
ions_to_train = [(1, 1), (4, 2), (3, 3), (4, 4)]
ion_training_set = ions_dataset.get_ions(ions_to_train)
h2_mols_to_train = [128, 384]
h2_training_set = h2_dataset.get_molecules(h2_mols_to_train)
h4_mols_to_train = [208, 336]
h4_training_set = h4_dataset.get_molecules(h4_mols_to_train)
training_set = datasets.concatenate_kohn_sham_states(ion_training_set, h2_training_set, h4_training_set)

# validation sets
ions_to_validate = [(4, 3), (2,2)]
ions_validation_set = ions_dataset.get_ions(ions_to_validate)
h3_mols_to_validate = [208]
h3_validation_set = h3_dataset.get_molecules(h3_mols_to_validate)
validation_set = datasets.concatenate_kohn_sham_states(ions_validation_set, h3_validation_set)


# init trainer with grids
trainer = ksr.SpinKSR(grids)
# set training/validation sets
trainer.set_training_set(training_set)
trainer.set_validation_set(validation_set)

# set KS parameters
trainer.set_ks_params(  # The number of Kohn-Sham iterations in training.
  num_iterations=20,  # @The density linear mixing factor.
  alpha=0.5,  # Decay factor of density linear mixing factor.
  alpha_decay=0.9,  # Enforce reflection symmetry across the origin.
  enforce_reflection_symmetry=False,
  # The number of density differences in the previous iterations to mix the
  # density. Linear mixing is num_mixing_iterations = 1.
  num_mixing_iterations=1,
  # The stopping criteria of Kohn-Sham iteration on density.
  density_mse_converge_tolerance=-1.,
  # Apply stop gradient on the output state of this step and all steps
  # before. The first KS step is indexed as 0. Default -1, no stop gradient
  # is applied.
  stop_gradient_step=-1, )

# set ML model for xc functional
network = neural_xc.build_global_local_conv_net_sigma(num_global_filters=8,
  num_local_filters=16, num_local_conv_layers=2, activation='swish',
  grids=h2_dataset.grids, minval=0.1, maxval=2.385345,
  downsample_factor=0)
init_fn, neural_xc_energy_density_fn = neural_xc.global_functional_sigma(
  network, grids=grids)

trainer.set_neural_xc_functional(model_dir=model_dir,
  neural_xc_energy_density_fn=neural_xc_energy_density_fn)

trainer.setup_optimization(
    initial_checkpoint_index=0,
    save_every_n=10,
    max_train_steps=300,
    num_skipped_energies=-1,
    initial_params_file='models/ions/ksr_global_spin/t5_v1/overall_optimal_ckpt.pkl')

# perform training optimization
trainer.do_lbfgs_optimization(verbose=1)

# get optimal checkpoint from validation
_ = trainer.get_optimal_ckpt(model_dir)
