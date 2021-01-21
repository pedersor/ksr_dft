import os
import numpy as np
import sys

sys.path.append('../')
from longpaper_examples.train_validate_ions import Train_validate_ions
from jax_dft import scf


# load complete dataset
ions = Train_validate_ions('../data/ions/unpol_lda/basic_all')
dataset = ions.get_complete_dataset(num_grids=513)
# set training set
to_train = [(2, 2), (3, 3)]
mask = dataset.get_mask_ions(to_train)
training_set = dataset.get_subdataset(mask)
training_set = training_set.get_ions()

print(training_set.num_electrons)

densities = scf.get_initial_density(training_set, 'noninteracting')
print(np.sum(densities[0])*0.08)
print(np.sum(densities[1])*0.08)

print(densities[0])
print(densities[1])