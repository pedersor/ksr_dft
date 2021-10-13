import pathlib
from shutil import copyfile
import textwrap
import subprocess

import numpy as np


base_cwd = pathlib.Path.cwd()
run_file = base_cwd / 'train_validate.py'

seed_dirs = np.arange(0, 30, 1)
for seed in seed_dirs:

  # create seed directories
  cwd = base_cwd / f's{seed}'
  cwd.mkdir(parents=True, exist_ok=True)
  
  # write the output header to contain run_file 
  # i.e. output is sufficient for reproduce 
  i = 0
  output_file = cwd / f'output_{i}.out'
  while output_file.exists():
    i += 1
    output_file = cwd / f'output_{i}.out'

  with open(output_file, "w") as fh:
    fh.writelines(f"=== start contents of run_file = {run_file} ===")
    fh.writelines(open(run_file).readlines())
    fh.writelines(f"=== end contents of run_file = {run_file} ===")

  job_name = f'sig_s{seed}'

  with open(cwd / 'jobscript', "w") as fh:
    # slurm commands
    lines = (f"""\
        #!/bin/bash
        #SBATCH --job-name="{job_name}"
        #SBATCH --account=burke
        #SBATCH --partition=nes2.8,brd2.4,sib2.9
        #SBATCH --ntasks=1
        #SBATCH --nodes=1
        #SBATCH --cpus-per-task=8
        #SBATCH --time=24:00:00
        #SBATCH --mem=16G

        ml purge
        ml miniconda/3/own
        srun python {run_file} --seed {seed} >> {output_file}

    """)

    lines = textwrap.dedent(lines)
    fh.writelines(lines)

  # slurm batch submit
  proc = subprocess.Popen("sbatch jobscript", shell=True, cwd=cwd, stdout=None)
