import numpy as np


dx = 0.08
grids = np.arange(-256, 257) * dx


r_s = 1
n = 1 / (2 * r_s)
s_vals = np.linspace(0, 5)

for s in s_vals:
  print(s)

  density = np.zeros(len(grids))

  n_plus = n + (2*np.pi*(n**2))*s*dx
  n_minus = n - (2*np.pi*(n**2))*s*dx

  density[0] = density[-1] = n_minus
  density[1] = density[-2] = n
  density[2] = density[-3] = n_plus

  print(density)


