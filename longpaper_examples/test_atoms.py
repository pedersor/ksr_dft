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


class Test_atoms:
  def __init__(self, path_to_ckpts):
    self.path_to_ckpts = path_to_ckpts


  def get_optimal_ckpt(self):
    return 





