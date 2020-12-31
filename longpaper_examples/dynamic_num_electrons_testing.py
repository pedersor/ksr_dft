from jax_dft import scf
import jax.numpy as jnp
import jax
import timeit
import sys

num_electrons = 3


grids = jnp.arange(10)


one_hot_helper = jnp.arange(len(grids))

one_hot = jnp.where(one_hot_helper < num_electrons,
                    1.0, 0.0)

one_hot = jnp.expand_dims(one_hot, axis=1)


wavefunctions = jnp.asarray([i*jnp.ones(10) for i in range(1,11)])

print(wavefunctions*one_hot)

sys.exit()





num_electrons_one_hot = []
for i in range(1,21):
  if i <= num_electrons:
    num_electrons_one_hot.append(jnp.ones(10))
  else:
    num_electrons_one_hot.append(jnp.zeros(10))

num_electrons_one_hot = jnp.asarray(num_electrons_one_hot)

time, _ = scf.wavefunctions_to_density(num_electrons_one_hot, wavefunctions, grids)
