from jax_dft import scf
import jax.numpy as jnp
import jax
import timeit
import sys

num_electrons = jnp.array([2, 2, 2, 3, 3, 3, 3, 3, 3, 3])


num_electrons = jnp.expand_dims(num_electrons, axis=1)


wavefunctions = jnp.asarray([i*jnp.ones(10) for i in range(1,11)])

print(wavefunctions)
print(wavefunctions/ num_electrons)
