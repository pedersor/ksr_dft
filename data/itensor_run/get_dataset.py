import os
import sys
sys.path.append('../../')

from collections import deque
import numpy as np

from jax_dft import utils


def parse_output(grids, output_path):
  """General parse output.txt from real-space iTensor calculation.

  Args:
      grids: Numpy array with shape (num_grids,).
      output_path: output.txt file from iTensor calculation.

  Returns:
      v_ee: <V_ee>, total electon-electron potential energy.
      t_plus_v_ext: <T + V_ext>, total one-body kinetic energy and external
          potential energy.
      density: density(x), normalized density on a grid with shape (num_grids,).
  """

  def get_val_after_equals(line):
    line_split = line.split("=")
    return float(line_split[1].rstrip())

  # grid spacing
  h = utils.get_dx(grids)

  # density
  density = deque([])

  # spin density (n_up - n_down)
  m = deque([])

  v_ee = t_plus_v_ext = None

  for line in reversed(list(open(output_path))):
    # remove trailing spaces, tabs, etc.
    line = line.rstrip()

    if 'Sz_' and 'n_' in line:
      line_split = line.split("   ")
      Sz_line = line_split[0]
      n_line = line_split[1]

      m.appendleft(get_val_after_equals(Sz_line))
      density.appendleft(get_val_after_equals(n_line))
    elif '<V>' in line:
      v_ee = get_val_after_equals(line)
    elif '<H2>' in line:
      t_plus_v_ext = get_val_after_equals(line)
      break

  density = np.asarray(density)
  # normalize density
  density = density / h

  return v_ee, t_plus_v_ext, density


if __name__ == '__main__':
  # LiH example

  h = 0.08  # grid spacing
  grids = np.arange(-256, 257) * h
  # range of separations in Bohr: (min, max)
  separations = np.arange(0, 6, h)
  nuclear_charges = np.array([3, 1])

  total_energies = []
  densities = []
  for sep in separations:
    sep_steps = int(round(float(sep / h)))

    curr_dir = f'R{sep_steps}'
    output = os.path.join(curr_dir, 'output.txt')
    v_ee, t_plus_v_ext, density = parse_output(grids, output)

    total_energy = v_ee + t_plus_v_ext
    total_energies.append(total_energy)
    densities.append(density)

  total_energies = np.asarray(total_energies)
  densities = np.asarray(densities)
  np.save('dataset/total_energies.npy', total_energies)
  np.save('dataset/densities.npy', densities)
