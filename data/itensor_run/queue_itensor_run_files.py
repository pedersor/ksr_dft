import os
import sys
sys.path.append('../../')


from shutil import copyfile
import numpy as np
from jax_dft import utils

import hoppingvmaker


def mkdir_p(dir):
  """Make a directory (dir) if it doesn't exist."""
  if not os.path.exists(dir):
    os.mkdir(dir)


def edit_input_file(separation):
  with open('input', 'r') as file:
    input = file.readlines()

  for i, line in enumerate(input):
    if 'separation' in line:
      input[i] = f'separation = {separation}\n'

  with open('input', 'w') as file:
    file.writelines(input)


h = 0.08  # grid spacing
grids = np.arange(-256, 257) * h
# range of separations in Bohr: (min, max)
separations = np.arange(0, 6, h)

nuclear_charges = np.array([3, 1])

cwd = os.getcwd()
for sep in separations:
  sep_steps = int(round(float(sep / h)))

  curr_dir = f'R{sep_steps}'
  print(curr_dir)
  mkdir_p(curr_dir)

  # edit the input file
  edit_input_file(sep_steps)

  locations = utils.get_unif_separated_nuclei_positions(grids,
    num_locations=len(nuclear_charges), separation=sep)

  external_potential = utils.get_atomic_chain_potential(grids=grids,
    locations=locations, nuclear_charges=nuclear_charges,
    interaction_fn=utils.exponential_coulomb)

  hoppingvmaker.get_ham1c(grids, external_potential)
  hoppingvmaker.get_vuncomp(grids)

  # compress vuncomp to MPO
  os.system('''julia compressMPO.jl Vuncomp''')
  os.remove('Vuncomp') # remove the large uncompressed file

  copyfile('input', os.path.join(curr_dir, 'input'))
  copyfile('electronBO.cc', os.path.join(curr_dir, 'electronBO.cc'))
  copyfile('Makefile', os.path.join(curr_dir, 'Makefile'))

  os.chdir(curr_dir)
  job_name = curr_dir
  with open('jobscript', "w") as fh:
    # slurm commands
    fh.writelines("#!/bin/bash\n")
    fh.writelines(f'''#SBATCH --job-name="{job_name}"\n''')
    fh.writelines('#SBATCH --account=burke\n')
    fh.writelines('#SBATCH --partition=nes2.8,brd2.4\n')
    fh.writelines("#SBATCH --ntasks=1\n")
    fh.writelines("#SBATCH --nodes=1\n")
    fh.writelines("#SBATCH --cpus-per-task=2\n")
    fh.writelines("#SBATCH --time=12:00:00\n")
    fh.writelines("\n")

    # load modules and run
    fh.writelines('ml purge\n')
    fh.writelines('ml gnu/9.1.0\n')
    fh.writelines('ml openblas/0.3.6\n')
    fh.writelines("\n")

    # compile and run
    fh.writelines('make\n')
    fh.writelines(f'srun electronBO input > output.txt \n')

    # remove large not needed data
    fh.writelines('rm electronBO\n')
    fh.writelines('rm electronBO.o\n')
    fh.writelines('rm sites\n')

  # queue runfile and cd out of dir
  os.system('''sbatch jobscript ''')
  os.chdir(cwd)
