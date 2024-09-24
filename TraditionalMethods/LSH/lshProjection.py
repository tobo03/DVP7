# %%
#Comparing HOG, Random Projection, Min Hash running time with CIFAR10
from sklearn.datasets import fetch_openml
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
import time
#from datasketch import MinHash, MinHashLSH

# %%
#Define Random Projection LSH
def random_projection_lsh(data:list[list[float]], n_hash:int=128, seed = 1) -> list[int]:
    """"    
    
    data : a single image as a matrix of floats

    n_hash : size of the hash
    
    returns: list of binary values
    """
    np.random.seed(seed)
    n_dimensions = data.shape[1]
    random_vectors = np.random.randn(n_hash, n_dimensions)
    projections = np.dot(data, random_vectors.T)
    hash_codes = (projections > 0).astype(int)
    return hash_codes 


