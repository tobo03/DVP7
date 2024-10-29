import random
import torch
import numpy as np
from typing import Tuple

def CreateBatch(X_tensor: torch.Tensor, y_train: np.ndarray, batch_size: int) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Creates a batch from the given dataset.
    
    Parameters
    ----------
        X_tensor (torch.Tensor) : The input tensor with all data samples. Shape should be (sample_size, features).
        y_train (np.ndarray) : The labels corresponding to the samples in X_tensor. Shape should be (sample_size,).
        batch_size (int) : The size of the batch to create.
 
    Returns
    --------
        X_batch (torch.Tensor) : A tensor of shape (batch_size, features).
        y_batch (np.ndarray) : An array of shape (batch_size, 1).
    """
    batch_i = random.sample(range(len(X_tensor)), batch_size)
    X_batch = torch.stack( [X_tensor[j] for j in batch_i] )
    y_batch = np.array([y_train[j] for j in batch_i])

    return X_batch, y_batch