import numpy as np


def get_data_set(name="train"):
    x = None
    y = None

    if name is "train":
        npzfile = np.load("./data/train_set.npz")
        x = npzfile['x']
        y = npzfile['y']
    elif name is "test":
        npzfile = np.load("./data_set/test_set.npz")
        x = npzfile['x']
        y = npzfile['y']

    return x, y
