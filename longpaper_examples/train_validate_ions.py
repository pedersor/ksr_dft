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
from jax_dft import scf
from jax_dft import utils
from jax_dft import xc

# Set the default dtype as float64
config.update('jax_enable_x64', True)


class Train_validate_ions(object):
  def __init__(self, datasets_base_dir):
    self.datasets_base_dir = datasets_base_dir

    # get jitted fns
    self.loss_value_and_grad_fn = jax.jit(jax.value_and_grad(self.loss_fn))

  def get_complete_dataset(self, num_grids=None):
    dataset = datasets.Dataset(path=self.datasets_base_dir, num_grids=num_grids)
    self.grids = dataset.grids
    self.grids_integration_factor = utils.get_dx(self.grids) * len(self.grids)
    self.num_electrons = dataset.num_electrons

    # Check distances are symmetric
    if not np.all(utils.location_center_at_grids_center_point(
        dataset.locations, self.grids)):
      raise ValueError(
        'Training set contains examples '
        'not centered at the center of the grids.')

    self.complete_dataset = dataset
    return dataset

  def set_training_set(self, training_set):
    training_set = training_set.get_ions()
    # obtain initial densities
    initial_densities = scf.get_initial_density(training_set,
                                                method='noninteracting')
    self.training_set = training_set._replace(
      initial_densities=initial_densities)

    return self

  def init_ksr_lda_model(self, model_dir):
    """KSR-LDA model. Window size = 1 constrains model to only local
    information."""

    network = neural_xc.build_sliding_net(
      window_size=1,
      num_filters_list=[16, 16, 16],
      activation='swish')

    init_fn, neural_xc_energy_density_fn = neural_xc.global_functional(
      network, grids=self.grids)
    self.neural_xc_energy_density_fn = neural_xc_energy_density_fn

    # update model directory
    self.model_dir = model_dir

    return init_fn

  def init_ksr_global_model(self, model_dir):
    """ KSR-global model."""

    network = neural_xc.build_global_local_conv_net(
      num_global_filters=16,
      num_local_filters=16,
      num_local_conv_layers=2,
      activation='swish',
      grids=self.grids,
      minval=0.1,
      maxval=2.385345,
      downsample_factor=0)
    network = neural_xc.wrap_network_with_self_interaction_layer(
      network, grids=self.grids, interaction_fn=utils.exponential_coulomb)

    init_fn, neural_xc_energy_density_fn = neural_xc.global_functional(
      network, grids=self.grids)
    self.neural_xc_energy_density_fn = neural_xc_energy_density_fn

    # update model directory
    self.model_dir = model_dir

    return init_fn

  def set_init_ksr_model_params(self, init_fn, key=jax.random.PRNGKey(0),
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

  @partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0))
  def _kohn_sham(self, params, locations, nuclear_charges,
                 initial_densities, num_electrons):
    return jit_scf.kohn_sham(
      locations=locations,
      nuclear_charges=nuclear_charges,
      num_electrons=num_electrons,
      grids=self.grids,
      xc_energy_density_fn=tree_util.Partial(
        self.neural_xc_energy_density_fn,
        params=params),
      interaction_fn=utils.exponential_coulomb,
      # The initial density of KS self-consistent calculations.
      initial_density=initial_densities,
      num_iterations=self.ks_params['num_iterations'],
      alpha=self.ks_params['alpha'],
      alpha_decay=self.ks_params['alpha_decay'],
      enforce_reflection_symmetry=self.ks_params['enforce_reflection_symmetry'],
      num_mixing_iterations=self.ks_params['num_mixing_iterations'],
      density_mse_converge_tolerance=self.ks_params[
        'density_mse_converge_tolerance'],
      stop_gradient_step=self.ks_params['stop_gradient_step'])

  def kohn_sham(self, params, locations, nuclear_charges,
                initial_densities, num_electrons):
    return self._kohn_sham(params, locations,
                           nuclear_charges,
                           initial_densities, num_electrons)

  def loss_fn(self, flatten_params):
    """Get losses."""
    params = np_utils.unflatten(self.spec, flatten_params)
    states = self.kohn_sham(params, self.training_set.locations,
                            self.training_set.nuclear_charges,
                            self.training_set.initial_densities,
                            self.training_set.num_electrons)
    # Energy loss
    loss_value = losses.trajectory_mse(
      target=self.training_set.total_energy,
      predict=states.total_energy[
              # The starting states have larger errors. Ignore the number of
              # starting states (here 10) in loss.
              :, 10:],
      # The discount factor in the trajectory loss.
      discount=0.9,
      num_electrons=self.training_set.num_electrons)
    # Density loss
    loss_value += losses.mean_square_error(
      target=self.training_set.density,
      predict=states.density[:, -1, :],
      num_electrons=self.training_set.num_electrons) * self.grids_integration_factor
    return loss_value

  def setup_optimization(self, **kwargs):
    self.loss_record = []
    self.optimization_params = kwargs

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
        print(f'Save checkpoint {checkpoint_path}')
      else:
        pass
      with open(checkpoint_path, 'wb') as handle:
        pickle.dump(np_utils.unflatten(self.spec, flatten_params), handle)

    self.loss_record.append(train_set_loss)
    return train_set_loss, np.array(train_set_gradient)

  def do_lbfgs_optimization(self, verbose=1):

    _, loss, _ = scipy.optimize.fmin_l_bfgs_b(
      self.np_loss_and_grad_fn,
      x0=self.flatten_init_params,
      args=(verbose,),
      # Maximum number of function evaluations.
      maxfun=self.optimization_params['max_train_steps'],
      factr=1,
      m=20,
      pgtol=1e-14)
    print(f'Final loss = {loss}')
    return self

  # Validation fns -------------------------------------------------------------

  def set_validation_set(self, validation_set):
    """Sets the validation set from a list of ions."""
    validation_set = validation_set.get_ions()
    # obtain initial densities
    initial_densities = scf.get_initial_density(validation_set,
                                                method='noninteracting')
    self.validation_set = validation_set._replace(
      initial_densities=initial_densities)
    return self

  def set_test_set(self, test_set):
    """Sets the validation set from a list of ions."""
    test_set = test_set.get_ions()
    # obtain initial densities
    initial_densities = scf.get_initial_density(test_set,
                                                method='noninteracting')
    self.test_set = test_set._replace(
      initial_densities=initial_densities)
    return self

  def get_test_states(self, optimal_ckpt_path=None):
    if optimal_ckpt_path is not None:
      with open(optimal_ckpt_path, 'rb') as handle:
        self.optimal_ckpt_params = pickle.load(handle)

    states = self.kohn_sham(
      self.optimal_ckpt_params,
      locations=self.test_set.locations,
      nuclear_charges=self.test_set.nuclear_charges,
      initial_densities=self.test_set.initial_densities,
      num_electrons=self.test_set.num_electrons)
    return states

  def get_final_test_states(self, optimal_ckpt_path=None):
    states = self.get_test_states(optimal_ckpt_path=optimal_ckpt_path)
    # get only converged results
    return tree_util.tree_map(lambda x: x[:, -1], states)

  def _get_states(self, ckpt_path):
    print(f'Load {ckpt_path}')
    with open(ckpt_path, 'rb') as handle:
      params = pickle.load(handle)

    states = self.kohn_sham(
      params,
      locations=self.validation_set.locations,
      nuclear_charges=self.validation_set.nuclear_charges,
      initial_densities=self.validation_set.initial_densities,
      num_electrons=self.validation_set.num_electrons)
    return params, states

  def get_optimal_ckpt(self, path_to_ckpts):
    # TODO: non-jitted/vmapped option.
    ckpt_list = sorted(
      glob.glob(os.path.join(path_to_ckpts, 'ckpt-?????')))

    optimal_ckpt_params = None
    min_loss = None
    for ckpt_path in ckpt_list:
      params, states = self._get_states(ckpt_path)

      # Energy loss
      loss_value = losses.mean_square_error(
        target=self.validation_set.total_energy,
        predict=states.total_energy[:, -1],
        num_electrons=self.validation_set.num_electrons)

      # Density loss (however, KSR paper does not use this for validation)
      # loss_value += losses.mean_square_error(
      #  target=self.validation_set.density, predict=states.density[:, -1, :]
      # ) * self.grids_integration_factor / self.num_electrons

      if optimal_ckpt_params is None or loss_value < min_loss:
        optimal_ckpt_params = params
        min_loss = loss_value

    optimal_ckpt_path = os.path.join(path_to_ckpts, 'optimal_ckpt.pkl')
    print(f'optimal checkpoint loss: {min_loss}')
    print(f'Save {optimal_ckpt_path}')
    with open(optimal_ckpt_path, 'wb') as handle:
      pickle.dump(optimal_ckpt_params, handle)

    self.optimal_ckpt_params = optimal_ckpt_params
    return self


if __name__ == '__main__':
  """Test and validate ions example."""

  ## Load data and setup model

  # load complete dataset
  ions = Train_validate_ions('../data/ions/unpol_lda/basic_all')
  dataset = ions.get_complete_dataset(num_grids=513)

  # set ML model for xc functional
  model_dir = '../models/ions/unpol_lda'
  init_fn = ions.init_ksr_lda_model(model_dir=model_dir)
  # write model specs to README file
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  readme_file = os.path.join(model_dir, 'README.txt')
  with open(readme_file, "w") as fh:
    fh.writelines("name: KSR-LDA\n")
    fh.writelines('''network = neural_xc.build_sliding_net(
      window_size=1,
      num_filters_list=[16, 16, 16],
      activation='swish')\n''')

  # set initial params from init_fn
  key = jax.random.PRNGKey(0)
  ions.set_init_ksr_model_params(init_fn, key)
  # optional: setting all parameters to be negative can help prevent exploding
  # loss in initial steps esp. for KSR-LDA.
  ions.flatten_init_params = -np.abs(ions.flatten_init_params)

  # set KS parameters
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

  ## Train Ions

  # set training set
  to_train = [(2, 2), (3, 3)]
  mask = dataset.get_mask_ions(to_train)
  training_set = dataset.get_subdataset(mask)
  ions.set_training_set(training_set)

  # setup parameters associated with the optimization
  # TODO: add aditional params..
  ions.setup_optimization(
    initial_checkpoint_index=0,
    save_every_n=10,
    max_train_steps=100
  )

  # perform training optimization
  ions.do_lbfgs_optimization(verbose=1)

  ## Validate Ions

  # set validation set
  to_validate = [(1, 1)]
  mask = dataset.get_mask_ions(to_validate)
  validation_set = dataset.get_subdataset(mask)
  ions.set_validation_set(validation_set)

  # get optimal checkpoint from validation
  ions.get_optimal_ckpt(model_dir)

  # append training set and validation set info to README
  readme_file = os.path.join(model_dir, 'README.txt')
  with open(readme_file, "a") as fh:
    fh.writelines('\n')
    fh.writelines(f"Trained on: {to_train} \n")
    fh.writelines(f"Validated on: {to_validate} \n")
