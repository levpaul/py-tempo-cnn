repo for trying to make a tempo estimator for guitars


# Steps to create a new Datset

For both test and train:

Augment data with `augment-audio.py`

Extract spectros with `gen_spectrograms.py`

Squash spectrograms to npy saves with `squash_spectros.py`

Adjust dataloader and train!
