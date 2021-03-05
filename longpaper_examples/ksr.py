import os
import sys
from functools import partial
import glob
import pickle
import time

import numpy as np
import scipy
import jax
from jax import random
from jax import tree_util
from jax.config import config
import jax.numpy as jnp
from jax_dft import datasets
from jax_dft import jit_scf
from jax_dft import losses
from jax_dft import neural_xc
from jax_dft import np_utils
from jax_dft import spin_scf
from jax_dft import utils
from jax_dft import xc
from jax_dft import jit_spin_scf

# Set the default dtype as float64
config.update('jax_enable_x64', True)


class SpinKSR(object):
  def __init__(self, grids):
    self.grids = grids
    self.grids_integration_factor = utils.get_dx(grids) * len(grids)

    # get jitted fns
    self.loss_value_and_grad_fn = jax.jit(jax.value_and_grad(self.loss_fn))

  def set_training_set(self, training_set):
    # obtain initial densities and spin densities
    initial_densities, initial_spin_densities = (
      spin_scf.get_initial_density_sigma(training_set, method='noninteracting'))
    self.training_set = training_set._replace(
      initial_densities=initial_densities,
      initial_spin_densities=initial_spin_densities)

    unpolarized_mask = np.array(
      [num_unpaired_electrons == 0 for num_unpaired_electrons in
        self.training_set.num_unpaired_electrons])
    polarized_mask = np.invert(unpolarized_mask)

    unpolarized_training_set = tree_util.tree_map(lambda x: x[unpolarized_mask],
      self.training_set)
    polarized_training_set = tree_util.tree_map(lambda x: x[polarized_mask],
      self.training_set)

    self.training_set = (unpolarized_training_set, polarized_training_set)

    return self

  def set_neural_xc_functional(self, model_dir, neural_xc_energy_density_fn):
    """ Sets the neural XC functional and model directory."""

    # update model directory
    self.model_dir = model_dir
    self.neural_xc_energy_density_fn = neural_xc_energy_density_fn

    return self

  def set_init_model_params(self, init_fn, key=jax.random.PRNGKey(0),
      verbose=1):
    """Set initial model parameters from init_fn."""

    init_params = init_fn(key)
    spec, flatten_init_params = np_utils.flatten(init_params)

    # sets spec (for unflatting params) and flattened params
    self.spec = spec
    self.flatten_init_params = flatten_init_params

    if verbose == 1:
      num_parameters = len(flatten_init_params)
      print(f'number of parameters = {num_parameters}')
    else:
      pass

    return self

  def set_ks_params(self, **kwargs):
    self.ks_params = kwargs
    return self

  @partial(jax.jit, static_argnums=0)
  def kohn_sham(self, params, external_potentials, initial_densities,
      initial_spin_densities, num_electrons, num_unpaired_electrons):
    return self._kohn_sham(params, external_potentials, initial_densities,
      initial_spin_densities, num_electrons, num_unpaired_electrons)

  @partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0))
  def _kohn_sham(self, params, external_potentials, initial_densities,
      initial_spin_densities, num_electrons, num_unpaired_electrons):
    return jax.lax.cond(num_unpaired_electrons == 0,
      true_fun=lambda _: jit_scf.kohn_sham(
        external_potential=external_potentials, num_electrons=num_electrons,
        num_unpaired_electrons=num_unpaired_electrons, grids=self.grids,
        xc_energy_density_fn=tree_util.Partial(self.neural_xc_energy_density_fn,
          params=params), interaction_fn=utils.exponential_coulomb,
        initial_density=initial_densities,
        initial_spin_density=initial_spin_densities,
        num_iterations=self.ks_params['num_iterations'],
        alpha=self.ks_params['alpha'],
        alpha_decay=self.ks_params['alpha_decay'],
        enforce_reflection_symmetry=self.ks_params[
          'enforce_reflection_symmetry'],
        num_mixing_iterations=self.ks_params['num_mixing_iterations'],
        density_mse_converge_tolerance=self.ks_params[
          'density_mse_converge_tolerance'],
        stop_gradient_step=self.ks_params['stop_gradient_step']),
      false_fun=lambda _: jit_spin_scf.kohn_sham(
        external_potential=external_potentials, num_electrons=num_electrons,
        num_unpaired_electrons=num_unpaired_electrons, grids=self.grids,
        xc_energy_density_fn=tree_util.Partial(self.neural_xc_energy_density_fn,
          params=params), interaction_fn=utils.exponential_coulomb,
        initial_density=initial_densities,
        initial_spin_density=initial_spin_densities,
        num_iterations=self.ks_params['num_iterations'],
        alpha=self.ks_params['alpha'],
        alpha_decay=self.ks_params['alpha_decay'],
        enforce_reflection_symmetry=self.ks_params[
          'enforce_reflection_symmetry'],
        num_mixing_iterations=self.ks_params['num_mixing_iterations'],
        density_mse_converge_tolerance=self.ks_params[
          'density_mse_converge_tolerance'],
        stop_gradient_step=self.ks_params['stop_gradient_step']), operand=None)

  @partial(jax.jit, static_argnums=0)
  def unrestricted_kohn_sham(self, params, external_potentials,
      initial_densities, initial_spin_densities, num_electrons,
      num_unpaired_electrons):
    return self._unrestricted_kohn_sham(params, external_potentials,
      initial_densities, initial_spin_densities, num_electrons,
      num_unpaired_electrons)

  @partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0))
  def _unrestricted_kohn_sham(self, params, external_potentials,
      initial_densities, initial_spin_densities, num_electrons,
      num_unpaired_electrons):
    return jit_spin_scf.kohn_sham(external_potential=external_potentials,
      num_electrons=num_electrons,
      num_unpaired_electrons=num_unpaired_electrons, grids=self.grids,
      xc_energy_density_fn=tree_util.Partial(self.neural_xc_energy_density_fn,
        params=params), interaction_fn=utils.exponential_coulomb,
      initial_density=initial_densities,
      initial_spin_density=initial_spin_densities,
      num_iterations=self.ks_params['num_iterations'],
      alpha=self.ks_params['alpha'], alpha_decay=self.ks_params['alpha_decay'],
      enforce_reflection_symmetry=self.ks_params['enforce_reflection_symmetry'],
      num_mixing_iterations=self.ks_params['num_mixing_iterations'],
      density_mse_converge_tolerance=self.ks_params[
        'density_mse_converge_tolerance'],
      stop_gradient_step=self.ks_params['stop_gradient_step'])

  @partial(jax.jit, static_argnums=0)
  def restricted_kohn_sham(self, params, external_potentials, initial_densities,
      initial_spin_densities, num_electrons, num_unpaired_electrons):
    return self._restricted_kohn_sham(params, external_potentials,
      initial_densities, initial_spin_densities, num_electrons,
      num_unpaired_electrons)

  @partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0))
  def _restricted_kohn_sham(self, params, external_potentials,
      initial_densities, initial_spin_densities, num_electrons,
      num_unpaired_electrons):
    return jit_scf.kohn_sham(external_potential=external_potentials,
      num_electrons=num_electrons,
      num_unpaired_electrons=num_unpaired_electrons, grids=self.grids,
      xc_energy_density_fn=tree_util.Partial(self.neural_xc_energy_density_fn,
        params=params), interaction_fn=utils.exponential_coulomb,
      initial_density=initial_densities,
      initial_spin_density=initial_spin_densities,
      num_iterations=self.ks_params['num_iterations'],
      alpha=self.ks_params['alpha'], alpha_decay=self.ks_params['alpha_decay'],
      enforce_reflection_symmetry=self.ks_params['enforce_reflection_symmetry'],
      num_mixing_iterations=self.ks_params['num_mixing_iterations'],
      density_mse_converge_tolerance=self.ks_params[
        'density_mse_converge_tolerance'],
      stop_gradient_step=self.ks_params['stop_gradient_step'])

  def loss_fn(self, flatten_params):
    """Get losses."""
    params = np_utils.unflatten(self.spec, flatten_params)

    unpolarized_training_set, polarized_training_set = self.training_set

    unpolarized_states = self.restricted_kohn_sham(params,
      unpolarized_training_set.external_potential,
      unpolarized_training_set.initial_densities,
      unpolarized_training_set.initial_spin_densities,
      unpolarized_training_set.num_electrons,
      unpolarized_training_set.num_unpaired_electrons)
    num_unpolarized_states = len(unpolarized_states.num_electrons)

    polarized_states = self.unrestricted_kohn_sham(params,
      polarized_training_set.external_potential,
      polarized_training_set.initial_densities,
      polarized_training_set.initial_spin_densities,
      polarized_training_set.num_electrons,
      polarized_training_set.num_unpaired_electrons)
    num_polarized_states = len(polarized_states.num_electrons)

    # unpolarized energy loss
    loss_value = losses.trajectory_mse(
      target=unpolarized_training_set.total_energy,
      predict=unpolarized_states.total_energy[
        # The starting states have larger errors. Ignore a number of
        # starting states in loss.
      :, self.optimization_params['num_skipped_energies']:],
      # The discount factor in the trajectory loss.
      discount=0.9,
      num_electrons=unpolarized_training_set.num_electrons) * num_unpolarized_states

    # polarized energy loss
    loss_value += losses.trajectory_mse(
      target=polarized_training_set.total_energy,
      predict=polarized_states.total_energy[
        # The starting states have larger errors. Ignore a number of
        # starting states in loss.
      :, self.optimization_params['num_skipped_energies']:],
      # The discount factor in the trajectory loss.
      discount=0.9,
      num_electrons=polarized_training_set.num_electrons) * num_polarized_states

    # unpolarized density loss
    loss_value += (
        losses.mean_square_error(target=unpolarized_training_set.density,
          predict=unpolarized_states.density[:, -1, :],
          num_electrons=unpolarized_training_set.num_electrons) * self.grids_integration_factor * num_unpolarized_states)

    # polarized density loss
    loss_value += (
        losses.mean_square_error(target=polarized_training_set.density,
          predict=polarized_states.density[:, -1, :],
          num_electrons=polarized_training_set.num_electrons) * self.grids_integration_factor * num_polarized_states)

    # take average
    return loss_value / (num_polarized_states + num_unpolarized_states)

  def setup_optimization(self, **kwargs):
    self.loss_record = []
    self.optimization_params = kwargs

    if 'initial_checkpoint_index' in self.optimization_params:
      initial_checkpoint_index = self.optimization_params[
        'initial_checkpoint_index']
      if initial_checkpoint_index != 0:
        checkpoint_path = f'ckpt-{initial_checkpoint_index:05d}'
        checkpoint_path = os.path.join(self.model_dir, checkpoint_path)
        with open(checkpoint_path, 'rb') as handle:
          init_params = pickle.load(handle)

        # sets spec (for unflatting params) and flattened params
        # for specified checkpoint
        self.spec, self.flatten_init_params = np_utils.flatten(init_params)
    else:
      self.optimization_params['initial_checkpoint_index'] = 0

    if 'initial_params_file' in self.optimization_params:
      init_params_path = self.optimization_params['initial_params_file']
      with open(init_params_path, 'rb') as handle:
        init_params = pickle.load(handle)

      # sets spec (for unflatting params) and flattened params
      # for specified checkpoint
      self.spec, self.flatten_init_params = np_utils.flatten(init_params)

    if 'loss_weight' not in self.optimization_params:
      self.optimization_params['loss_weight'] = 0.5

    return self

  def np_loss_and_grad_fn(self, flatten_params, verbose):
    """Gets loss value and gradient of parameters as float and numpy array."""
    start_time = time.time()
    # Automatic differentiation.
    train_set_loss, train_set_gradient = self.loss_value_and_grad_fn(
      flatten_params)
    step_time = time.time() - start_time
    step = self.optimization_params['initial_checkpoint_index'] + len(
      self.loss_record)
    if verbose == 1:
      print(f'step {step}, loss {train_set_loss} in {step_time} sec')
    else:
      pass

    # Save checkpoints.
    if len(self.loss_record) % self.optimization_params['save_every_n'] == 0:
      checkpoint_path = f'ckpt-{step:05d}'
      checkpoint_path = os.path.join(self.model_dir, checkpoint_path)
      if verbose == 1:
        print(f'Save checkpoint {checkpoint_path}', flush=True)
      else:
        pass
      with open(checkpoint_path, 'wb') as handle:
        pickle.dump(np_utils.unflatten(self.spec, flatten_params), handle)

    self.loss_record.append(train_set_loss)
    return train_set_loss, np.array(train_set_gradient)

  def do_lbfgs_optimization(self, verbose=1):

    _, loss, _ = scipy.optimize.fmin_l_bfgs_b(self.np_loss_and_grad_fn,
      x0=self.flatten_init_params, args=(verbose,),
      # Maximum number of function evaluations.
      maxfun=self.optimization_params['max_train_steps'], factr=1, m=20,
      pgtol=1e-14)
    print(f'Final loss = {loss}')
    return self

  # Validation fns -------------------------------------------------------------

  def set_validation_set(self, validation_set):
    """Sets the validation set from a list of ions."""
    # obtain initial densities and spin densities
    initial_densities, initial_spin_densities = (
      spin_scf.get_initial_density_sigma(validation_set,
        method='noninteracting'))

    self.validation_set = validation_set._replace(
      initial_densities=initial_densities,
      initial_spin_densities=initial_spin_densities)
    return self

  def set_test_set(self, test_set):
    """Sets the test set from a list of ions."""
    # obtain initial densities and spin densities
    initial_densities, initial_spin_densities = (
      spin_scf.get_initial_density_sigma(test_set, method='noninteracting'))

    self.test_set = test_set._replace(initial_densities=initial_densities,
      initial_spin_densities=initial_spin_densities)
    return self

  def get_test_states(self, optimal_ckpt_path=None):
    if optimal_ckpt_path is not None:
      with open(optimal_ckpt_path, 'rb') as handle:
        self.optimal_ckpt_params = pickle.load(handle)

    states = self.kohn_sham(self.optimal_ckpt_params,
      external_potentials=self.test_set.external_potential,
      initial_densities=self.test_set.initial_densities,
      initial_spin_densities=self.test_set.initial_spin_densities,
      num_electrons=self.test_set.num_electrons,
      num_unpaired_electrons=self.test_set.num_unpaired_electrons)
    return states

  def get_final_states(self, states):
    """ Get only converged results. """

    return tree_util.tree_map(lambda x: x[:, -1], states)

  def _get_states(self, ckpt_path):
    print(f'Load {ckpt_path}')
    with open(ckpt_path, 'rb') as handle:
      params = pickle.load(handle)

    states = self.kohn_sham(params, self.validation_set.external_potential,
      self.validation_set.initial_densities,
      self.validation_set.initial_spin_densities,
      self.validation_set.num_electrons,
      self.validation_set.num_unpaired_electrons)

    return params, states

  def get_optimal_ckpt(self, path_to_ckpts):
    # TODO: non-jitted/vmapped option.
    ckpt_list = sorted(glob.glob(os.path.join(path_to_ckpts, 'ckpt-?????')))

    optimal_ckpt_params = None
    optimal_ckpt_path = None
    min_loss = None
    for ckpt_path in ckpt_list:
      params, states = self._get_states(ckpt_path)

      # Energy loss
      loss_value = losses.mean_square_error(
        target=self.validation_set.total_energy,
        predict=states.total_energy[:, -1],
        num_electrons=self.validation_set.num_electrons)

      # Density loss (however, KSR paper does not use this for validation)
      loss_value += losses.mean_square_error(target=self.validation_set.density,
        predict=states.density[:, -1, :],
        num_electrons=self.validation_set.num_electrons) * self.grids_integration_factor

      print(f'loss value = {loss_value}')
      if optimal_ckpt_params is None or loss_value < min_loss:
        optimal_ckpt_params = params
        optimal_ckpt_path = ckpt_path
        min_loss = loss_value

    print(f'optimal checkpoint: {optimal_ckpt_path}')
    optimal_ckpt_path = os.path.join(path_to_ckpts, 'optimal_ckpt.pkl')
    print(f'optimal checkpoint loss: {min_loss}')
    print(f'Save {optimal_ckpt_path}')
    with open(optimal_ckpt_path, 'wb') as handle:
      pickle.dump(optimal_ckpt_params, handle)

    self.optimal_ckpt_params = optimal_ckpt_params
    return optimal_ckpt_path


class PureKSR(SpinKSR):
  def __init__(self, grids):
    super().__init__(grids)
    self.loss_value_and_grad_fn = jax.jit(jax.value_and_grad(self.loss_fn))

  def set_training_set(self, training_set):
    # obtain initial densities and spin densities
    initial_densities, initial_spin_densities = (
      spin_scf.get_initial_density_sigma(training_set, method='noninteracting'))
    self.training_set = training_set._replace(
      initial_densities=initial_densities,
      initial_spin_densities=initial_spin_densities)

    return self

  @partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0))
  def _kohn_sham(self, params, external_potentials, initial_densities,
      initial_spin_densities, num_electrons, num_unpaired_electrons):
    return jit_scf.kohn_sham(external_potential=external_potentials,
      num_electrons=num_electrons,
      num_unpaired_electrons=num_unpaired_electrons, grids=self.grids,
      xc_energy_density_fn=tree_util.Partial(self.neural_xc_energy_density_fn,
        params=params), interaction_fn=utils.exponential_coulomb,
      initial_density=initial_densities,
      initial_spin_density=initial_spin_densities,
      num_iterations=self.ks_params['num_iterations'],
      alpha=self.ks_params['alpha'], alpha_decay=self.ks_params['alpha_decay'],
      enforce_reflection_symmetry=self.ks_params['enforce_reflection_symmetry'],
      num_mixing_iterations=self.ks_params['num_mixing_iterations'],
      density_mse_converge_tolerance=self.ks_params[
        'density_mse_converge_tolerance'],
      stop_gradient_step=self.ks_params['stop_gradient_step'])

  def loss_fn(self, flatten_params):
    """Get losses."""
    params = np_utils.unflatten(self.spec, flatten_params)
    states = self.restricted_kohn_sham(params,
      self.training_set.external_potential, self.training_set.initial_densities,
      self.training_set.initial_spin_densities, self.training_set.num_electrons,
      self.training_set.num_unpaired_electrons)
    # Energy loss
    loss_value = losses.trajectory_mse(target=self.training_set.total_energy,
      predict=states.total_energy[
        # The starting states have larger errors. Ignore a number of
        # starting states in loss.
      :, self.optimization_params['num_skipped_energies']:],
      # The discount factor in the trajectory loss.
      discount=0.9, num_electrons=self.training_set.num_electrons)
    # Density loss
    loss_value += losses.mean_square_error(target=self.training_set.density,
      predict=states.density[:, -1, :],
      num_electrons=self.training_set.num_electrons) * self.grids_integration_factor
    return loss_value


if __name__ == '__main__':
  """Test and validate ions example."""

  ## Load data and setup model

  # load complete dataset
  ions_dataset = datasets.Dataset('../data/ions/lsda', num_grids=513)
  grids = ions_dataset.grids
  trainer = SpinKSR(grids)

  # set ML model for xc functional
  model_dir = '../models/ions/ksr_lsda/lsda'
  # (semi-)local models
  network = neural_xc.build_sliding_net(window_size=1,
    num_filters_list=[16, 16, 16], activation='swish')
  init_fn, neural_xc_energy_density_fn = neural_xc.global_functional_sigma(
    network, grids=grids)

  trainer.set_neural_xc_functional(model_dir=model_dir,
    neural_xc_energy_density_fn=neural_xc_energy_density_fn)

  # write model specs to README file
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  # set initial params from init_fn
  key = jax.random.PRNGKey(0)
  trainer.set_init_model_params(init_fn, key)
  # optional: setting all parameters to be negative can help prevent exploding
  # loss in initial steps esp. for KSR-LDA.
  trainer.flatten_init_params = -np.abs(trainer.flatten_init_params)

  # set KS parameters
  trainer.set_ks_params(  # The number of Kohn-Sham iterations in training.
    num_iterations=6,  # @The density linear mixing factor.
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

  ## Train Ions

  # set training set
  to_train = [(1, 1), (2, 2)]
  training_set = ions_dataset.get_ions(to_train)
  trainer.set_training_set(training_set)

  # setup parameters associated with the optimization
  # TODO: add aditional params..
  trainer.setup_optimization(initial_checkpoint_index=0, save_every_n=10,
    max_train_steps=100,
    # number of iterations skipped in energy loss evaluation
    num_skipped_energies=1,  # can start from initial params file
    # initial_params_file='/path/to/params.pkl',
  )

  # perform training optimization
  trainer.do_lbfgs_optimization(verbose=1)

  ## Validate Ions

  # set validation set
  to_validate = [(3, 3)]
  validation_set = ions_dataset.get_ions(to_validate)
  trainer.set_validation_set(validation_set)
  # get optimal checkpoint from validation
  trainer.get_optimal_ckpt(model_dir)
