import os
import glob
from shutil import copyfile

# get optimal ckpt paths from seeds in current dir
seed_list = glob.glob('s*')

curr_min = 0.1
overall_optimal_ckpt_seed = None
overall_optimal_loss = None
overall_optimal_ckpt = None
for seed in seed_list:

  output = os.path.join(seed, 'output.txt')
  optimal_ckpt = os.path.join(seed, 'optimal_ckpt.pkl')

  # open output starting from bottom of file
  for line in reversed(list(open(output))):
    # remove trailing spaces, tabs, etc.
    line = line.rstrip()

    if 'optimal checkpoint loss' in line:
      optimal_ckpt_loss = float(line.split(": ")[1])
      if optimal_ckpt_loss < curr_min:
        overall_optimal_loss = optimal_ckpt_loss
        overall_optimal_ckpt_seed = optimal_ckpt
        curr_min = optimal_ckpt_loss
      else:
        break

    if 'optimal checkpoint:' in line:
      overall_optimal_ckpt = line.split(": ")[1]

if overall_optimal_ckpt_seed:
  print('overall_optimal_ckpt: ')
  print(overall_optimal_ckpt_seed)
  print('validation loss: ')
  print(overall_optimal_loss)
  print('specific checkpoint: ')
  print(overall_optimal_ckpt)

  copyfile(overall_optimal_ckpt_seed, 'optimal_ckpt.pkl')
else:
  print('problem finding overall optimal ckpt.')
