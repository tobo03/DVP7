import random
import torch
import numpy as np
from typing import Optional, Tuple, Union

def createBatch(X_tensor: torch.Tensor, y_train: np.ndarray, batch_size: int, oneHot: Optional[int] = None) -> Tuple[torch.Tensor, Union[np.ndarray, torch.Tensor], list]:
    """
    Creates a batch from the given dataset.
    
    Parameters
    ----------
        X_tensor (torch.Tensor) : The input tensor with all data samples. Shape should be (sample_size, features).
        y_train (np.ndarray) : The labels corresponding to the samples in X_tensor. Shape should be (sample_size,).
        batch_size (int) : The size of the batch to create.
        oneHot (int | none) : if int, converts y_batch to a One Hot Encoded tensor of shape (batch_size, n_classes = oneHot).
 
    Returns
    --------
        X_batch (torch.Tensor) : A tensor of shape (batch_size, features).
        y_batch (np.ndarray | torch.Tensor) : if np.ndarray, an array of shape (batch_size,), if torch.Tensor, a tensor of shape (batch_size, n_classes = oneHot).
        index (list) : A list of indices for the chosen samples.
    """
    index = random.sample(range(len(X_tensor)), batch_size)
    X_batch = torch.stack( [X_tensor[j] for j in index] )
    y_batch = np.array([y_train[j] for j in index])
    
    if isinstance(oneHot, int):
        y_batch = torch.tensor(y_batch, dtype=torch.long)
        y_batch = torch.nn.functional.one_hot(y_batch, num_classes = oneHot)

    return X_batch, y_batch, index