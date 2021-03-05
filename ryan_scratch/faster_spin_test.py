from longpaper_examples import ksr
from jax_dft import datasets

# load complete dataset
ions_dataset = datasets.Dataset('../data/ions/lsda', num_grids=513)
grids = ions_dataset.grids
trainer = ksr.SpinKSR(grids)

# set training set
to_train = [(1, 1), (2, 2)]
training_set = ions_dataset.get_ions(to_train)
trainer.set_training_set(training_set)

