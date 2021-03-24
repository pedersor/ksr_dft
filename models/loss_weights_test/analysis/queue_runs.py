import os
from shutil import copyfile
import numpy as np

seeds = np.arange(0, 30, 1)

for seed in seeds:
  job_name = f'weights_s{seed}'

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
    # python 3 miniconda env with user packages
    fh.writelines('ml miniconda/3/own\n')
    fh.writelines(f'srun python seed_analysis.py '
                  f's{seed}\n')

  # queue runfile and cd out of dir
  os.system('''sbatch jobscript ''')
