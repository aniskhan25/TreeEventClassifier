import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit

from treeevent.utils.hexcode import hexcodes


def get_hexcodes(coords, sz_cell = (65000.0 / 2)):
    (q, r, s), _id = hexcodes(coords, sz_cell)
    coords = coords.copy()
    coords.loc[:, "q"] = q
    coords.loc[:, "r"] = r
    coords.loc[:, "s"] = s
    coords.loc[:, "id"] = _id
    return coords


def get_masks(coords, train_size=0.7):

    coords_with_hexcodes = get_hexcodes(coords)


    n = len(coords_with_hexcodes)
    indices = np.arange(n)

    gs = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=42)

    train_indices, test_indices = next(gs.split(indices, groups=coords_with_hexcodes.id))
    train_indices, val_indices  = next(gs.split(train_indices, groups=coords_with_hexcodes.loc[train_indices].id))

    train_mask = np.full((n), False)
    val_mask = np.full((n), False)
    test_mask = np.full((n), False)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask