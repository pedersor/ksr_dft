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

