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

config.update('jax_enable_x64', True)

dataset = datasets.Dataset(path='../data/atoms/', num_grids=513)
grids = dataset.grids
train_set = dataset.get_atoms([(2,2)])

#@title Check distances are symmetric
if not np.all(utils.location_center_at_grids_center_point(
    train_set.locations, grids)):
  raise ValueError(
      'Training set contains examples '
      'not centered at the center of the grids.')

#@title Initial density
initial_density = scf.get_initial_density(train_set, method='noninteracting')

#@title Initialize network
network = neural_xc.build_sliding_net(
    window_size=1,
    num_filters_list=[16,16,16],
    activation='swish')
init_fn, neural_xc_energy_density_fn = neural_xc.global_functional(
    network, grids=grids)

init_params = init_fn(random.PRNGKey(0))
initial_checkpoint_index = 0
spec, flatten_init_params = np_utils.flatten(init_params)
print(f'number of parameters: {len(flatten_init_params)}')

# @markdown The number of Kohn-Sham iterations in training.
num_iterations = 15  # @param{'type': 'integer'}
# @markdown The density linear mixing factor.
alpha = 0.5  # @param{'type': 'number'}
# @markdown Decay factor of density linear mixing factor.
alpha_decay = 0.9  # @param{'type': 'number'}
# @markdown The number of density differences in the previous iterations to mix the
# @markdown density. Linear mixing is num_mixing_iterations = 1.
num_mixing_iterations = 1  # @param{'type': 'integer'}
# @markdown The stopping criteria of Kohn-Sham iteration on density.
density_mse_converge_tolerance = -1.  # @param{'type': 'number'}
# @markdown Apply stop gradient on the output state of this step and all steps
# @markdown before. The first KS step is indexed as 0. Default -1, no stop gradient
# @markdown is applied.
stop_gradient_step = -1  # @param{'type': 'integer'}


def _kohn_sham(flatten_params, locations, nuclear_charges, initial_density):
    return jit_scf.kohn_sham(
        locations=locations,
        nuclear_charges=nuclear_charges,
        num_electrons=dataset.num_electrons,
        num_iterations=num_iterations,
        grids=grids,
        xc_energy_density_fn=tree_util.Partial(
            neural_xc_energy_density_fn,
            params=np_utils.unflatten(spec, flatten_params)),
        interaction_fn=utils.exponential_coulomb,
        # The initial density of KS self-consistent calculations.
        initial_density=initial_density,
        alpha=alpha,
        alpha_decay=alpha_decay,
        enforce_reflection_symmetry=True,
        num_mixing_iterations=num_mixing_iterations,
        density_mse_converge_tolerance=density_mse_converge_tolerance,
        stop_gradient_step=stop_gradient_step)


_batch_jit_kohn_sham = jax.vmap(_kohn_sham, in_axes=(None, 0, 0, 0))

grids_integration_factor = utils.get_dx(grids) * len(grids)


def loss_fn(
        flatten_params, locations, nuclear_charges,
        initial_density, target_energy, target_density):
    """Get losses."""
    states = _batch_jit_kohn_sham(
        flatten_params, locations, nuclear_charges, initial_density)
    # Energy loss
    loss_value = losses.trajectory_mse(
        target=target_energy,
        predict=states.total_energy[
                # The starting states have larger errors. Ignore the number of
                # starting states (here 10) in loss.
                :, 10:],
        # The discount factor in the trajectory loss.
        discount=0.9) / dataset.num_electrons
    # Density loss
    loss_value += losses.mean_square_error(
        target=target_density, predict=states.density[:, -1, :]
    ) * grids_integration_factor / dataset.num_electrons
    return loss_value


value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

# @markdown The frequency of saving checkpoints.
save_every_n = 20  # @param{'type': 'integer'}

loss_record = []


def np_value_and_grad_fn(flatten_params):
    """Gets loss value and gradient of parameters as float and numpy array."""
    start_time = time.time()
    # Automatic differentiation.
    train_set_loss, train_set_gradient = value_and_grad_fn(
        flatten_params,
        locations=train_set.locations,
        nuclear_charges=train_set.nuclear_charges,
        initial_density=initial_density,
        target_energy=train_set.total_energy,
        target_density=train_set.density)
    step_time = time.time() - start_time
    step = initial_checkpoint_index + len(loss_record)
    print(f'step {step}, loss {train_set_loss} in {step_time} sec')

    # Save checkpoints.
    if len(loss_record) % save_every_n == 0:
        checkpoint_path = f'ckpt-{step:05d}'
        print(f'Save checkpoint {checkpoint_path}')
        with open(checkpoint_path, 'wb') as handle:
            pickle.dump(np_utils.unflatten(spec, flatten_params), handle)

    loss_record.append(train_set_loss)
    return train_set_loss, np.array(train_set_gradient)


# @title Use L-BFGS optimizer to update neural network functional
# @markdown This cell trains the model. Each step takes about 1.6s.
max_train_steps = 200  # @param{'type': 'integer'}

_, _, info = scipy.optimize.fmin_l_bfgs_b(
    np_value_and_grad_fn,
    x0=-np.abs(np.array(flatten_init_params)),
    # Maximum number of function evaluations.
    maxfun=max_train_steps,
    factr=1,
    m=20,
    pgtol=1e-14)
print(info)
