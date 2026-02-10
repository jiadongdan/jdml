import numpy as np
import h5py
from ..utils import check_random_generator

def load_symmetry_h5(data_path, use_symmetry=True, data_proportion=1.0, seed=None):
    """Load train/val/test splits from an HDF5 file with optional symmetry channels and subsampling.

    Parameters
    ----------
    data_path : str or Path
        Path to the HDF5 file containing train/val/test splits.
    use_symmetry : bool
        If False, keep only the first channel (index 0).
    data_proportion : float
        Fraction of training data to use (0, 1]. Val/test are not subsampled.
    seed : int or RandomState, optional
        Random seed for reproducible subsampling.

    Returns
    -------
    train_data, train_lbs, val_data, val_lbs, test_data, test_lbs : np.ndarray
        Labels are zero-indexed based on the training set minimum.
    """
    splits = ('train', 'val', 'test')

    with h5py.File(data_path, 'r') as f:
        data = {s: f[f'{s}/data'][:] for s in splits}
        labels = {s: f[f'{s}/labels'][:] for s in splits}

    if not use_symmetry:
        for s in splits:
            data[s] = data[s][:, 0:1, :, :]

    if data_proportion < 1.0:
        rng = check_random_generator(seed)
        n_samples = data['train'].shape[0]
        subset_size = max(1, int(np.round(n_samples * data_proportion)))
        indices = rng.choice(n_samples, size=subset_size, replace=False)
        data['train'] = data['train'][indices]
        labels['train'] = labels['train'][indices]

    # Zero-index labels based on training set minimum
    label_offset = labels['train'].min()
    for s in splits:
        labels[s] -= label_offset

    return (data['train'], labels['train'],
            data['val'], labels['val'],
            data['test'], labels['test'])
