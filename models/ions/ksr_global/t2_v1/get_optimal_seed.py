import os
import glob
from shutil import copyfile

# get optimal ckpt paths from seeds in current dir
seed_list = glob.glob('s*')

curr_min = 0.1
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
      print(optimal_ckpt_loss)
      if optimal_ckpt_loss < curr_min:  
        overall_optimal_ckpt = optimal_ckpt
        curr_min = optimal_ckpt_loss

if overall_optimal_ckpt:
  print('overall_optimal_ckpt: ')
  print(overall_optimal_ckpt)
  copyfile(overall_optimal_ckpt, 'optimal_ckpt.pkl')
else:
  print('problem finding overall optimal ckpt.')
