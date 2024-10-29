import random
import torch



def CreateBatch(X_tensor, y_train, batch_size):

    batch_i = random.sample(range(len(X_tensor)), batch_size)
    X_batch = torch.stack( [X_tensor[j] for j in batch_i] )
    y_batch = [y_train[j] for j in batch_i]

    return X_batch, y_batch