import math
import h5py
import numpy as np
# torch related
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def load_h5_data(data_path):
    with h5py.File(data_path, 'r') as f:
        X_train = f['train']['X'][:].astype(np.float32)
        y_train = f['train']['y'][:].astype(np.int64)
        X_test  = f['test']['X'][:].astype(np.float32)
        y_test  = f['test']['y'][:].astype(np.int64)
    print(f"Loaded X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Loaded X_test  {X_test.shape},  y_test  {y_test.shape}")
    return X_train, y_train, X_test, y_test

def load_symmetry_h5(data_path, use_symmetry=True, data_proportion=1.0, seed=None):
    with h5py.File(data_path, 'r') as f:
        train_data = f['train/data'][:]
        train_lbs = f['train/labels'][:]
        val_data = f['val/data'][:]
        val_lbs = f['val/labels'][:]
        test_data = f['test/data'][:]
        test_lbs = f['test/labels'][:]
    if not use_symmetry:
        train_data = train_data[:,0:1,:,:]
        val_data = val_data[:,0:1,:,:]
        test_data = test_data[:,0:1,:,:]
    if data_proportion < 1.0:
        rng = check_random_generator(seed)
        n_samples = train_data.shape[0]
        subset_size = int(np.round(n_samples*data_proportion))
        indices = rng.choice(n_samples, size=subset_size, replace=False)
        train_data = train_data[indices]
        train_lbs = train_lbs[indices]
    train_lbs_min = train_lbs.min()
    train_lbs -= train_lbs_min
    val_lbs -= train_lbs_min
    test_lbs -= train_lbs_min
    return train_data, train_lbs, val_data, val_lbs, test_data, test_lbs

class SingleChannelDataset(Dataset):
    def __init__(self, X, y, rgb=False, device=None):
        """
        Args:
            X (torch.Tensor): Input tensor of shape (N, C, H, W).
            y (torch.Tensor): Labels of shape (N,).
            rgb (bool): If True, repeat first channel 3 times to simulate RGB.
            device (torch.device or str, optional): Move tensors to this device.
        """
        assert X.ndim == 4, f"Expected X of shape (N, C, H, W), got {X.shape}"
        self.X = X[:, 0:1, :, :]  # keep only the first channel
        if rgb:
            self.X = self.X.repeat(1, 3, 1, 1)  # repeat to 3 channels

        self.y = y
        if device is not None:
            self.X = self.X.to(device)
            self.y = self.y.to(device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
