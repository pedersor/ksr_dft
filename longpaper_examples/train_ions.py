import glob
import pickle
import time
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
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys
from functools import partial

config.update('jax_enable_x64', True)


class Train_ions:

  def __init__(self, datasets_base_dir):
    self.datasets_base_dir = datasets_base_dir
    self.loss_record = []

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

  def init_ksr_lda_model(self, key=jax.random.PRNGKey(0)):
    # KSR-LDA model. Window size = 1 constrains model to only local information.
    network = neural_xc.build_sliding_net(
      window_size=1,
      num_filters_list=[16, 16, 16],
      activation='swish')

    init_fn, neural_xc_energy_density_fn = neural_xc.global_functional(
      network, grids=self.grids)

    self.neural_xc_energy_density_fn = neural_xc_energy_density_fn

    init_params = init_fn(key)
    spec, flatten_init_params = np_utils.flatten(init_params)

    self.spec = spec
    # taking all the init params to be negative seems to improve
    # convergence of the loss.
    self.flatten_init_params = flatten_init_params
    self.num_parameters = len(flatten_init_params)

    return self

  def init_ksr_global_model(self, key=jax.random.PRNGKey(0)):
    # KSR-LDA model. Window size = 1 constrains model to only local information.
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

    init_params = init_fn(key)
    spec, flatten_init_params = np_utils.flatten(init_params)

    self.spec = spec
    # taking all the init params to be negative seems to improve
    # convergence of the loss.
    self.flatten_init_params = flatten_init_params
    self.num_parameters = len(flatten_init_params)

    return self

  def set_ks_parameters(self, **kwargs):
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
    self.optimization_params = kwargs
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


if __name__ == '__main__':
  ions = Train_ions('../data/ions/unpol_lda/basic_all')
  dataset = ions.get_complete_dataset(num_grids=513)

  # set training set
  to_train = [(2, 2), (3, 3)]
  mask = dataset.get_mask_ions(to_train)
  training_set = dataset.get_subdataset(mask)
  ions.set_training_set(training_set)

  # get ML model for xc functional
  key = jax.random.PRNGKey(3)
  ions.init_ksr_lda_model(key=key)
  print(f'number of parameters: {ions.num_parameters}')

  # get KS parameters
  ions.set_ks_parameters(
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

  # setup parameters associated with the optimization
  # TODO: add aditional params..
  ions.setup_optimization(
    initial_checkpoint_index=0,
    save_every_n=10,
    max_train_steps=100
  )

  # perform training optimization
  ions.do_lbfgs_optimization(verbose=1)
