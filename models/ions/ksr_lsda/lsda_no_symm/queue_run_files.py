import os
from shutil import copyfile
import numpy as np


def mkdir_p(dir):
  """Make a directory (dir) if it doesn't exist."""
  if not os.path.exists(dir):
    os.mkdir(dir)


training_dirs = np.arange(2, 6, 1)
validation_dirs = np.array([1])
seed_dirs = np.arange(0, 10, 1)


cwd = os.getcwd()
run_file = os.path.join(cwd, 'train_validate.py')

for train_dir in training_dirs:
  for val_dir in validation_dirs:

    train_val_dir = os.path.join(cwd, f't{train_dir}_v{val_dir}')
    mkdir_p(train_val_dir)

    for seed in seed_dirs:

      curr_dir = os.path.join(cwd, train_val_dir, f's{seed}')
      mkdir_p(curr_dir)

      os.chdir(curr_dir)
      job_name = f't{train_dir}_v{val_dir}_s{seed}'

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
        fh.writelines(
          f'srun python {run_file} '
          f't{train_dir} v{val_dir} s{seed} > output.txt \n')

      # queue runfile and cd out of dir
      os.system('''sbatch jobscript ''')
      os.chdir(cwd)
