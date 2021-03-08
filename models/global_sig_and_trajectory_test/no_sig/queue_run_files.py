import os
from shutil import copyfile
import numpy as np


def mkdir_p(dir):
  """Make a directory (dir) if it doesn't exist."""
  if not os.path.exists(dir):
    os.mkdir(dir)


num_skipped_list = np.array([0, 2, 4, 6, 8, 9], dtype=int)
seeds = np.arange(0, 30, 1)

cwd = os.getcwd()
run_file = os.path.join(cwd, 'train_validate.py')

for num_skipped in num_skipped_list:

  num_skipped_dir = os.path.join(cwd, f'n{num_skipped}')
  mkdir_p(num_skipped_dir)
  copyfile('get_optimal_seed.py',
    os.path.join(num_skipped_dir, 'get_optimal_seed.py'))

  for seed in seeds:

    seed_dir = os.path.join(cwd, num_skipped_dir, f's{seed}')
    mkdir_p(seed_dir)

    os.chdir(seed_dir)
    job_name = f'n{num_skipped}_s{seed}'

    with open('jobscript', "w") as fh:
      # slurm commands
      fh.writelines("#!/bin/bash\n")
      fh.writelines(f'''#SBATCH --job-name="{job_name}"\n''')
      fh.writelines('#SBATCH --account=burke\n')
      fh.writelines('#SBATCH --partition=nes2.8,brd2.4\n')
      fh.writelines("#SBATCH --ntasks=1\n")
      fh.writelines("#SBATCH --nodes=1\n")
      fh.writelines("#SBATCH --cpus-per-task=8\n")
      fh.writelines("#SBATCH --time=12:00:00\n")
      fh.writelines("#SBATCH --mem=8G\n")
      fh.writelines("\n")

      # load modules and run
      fh.writelines('ml purge\n')
      fh.writelines('ml gnu/9.1.0\n')
      # python 3 miniconda env with user packages
      fh.writelines('ml miniconda/3/own\n')
      fh.writelines(f'srun python {run_file} '
                    f'n{num_skipped} s{seed} > output.txt \n')

    # queue runfile and cd out of dir
    os.system('''sbatch jobscript ''')
    os.chdir(cwd)
