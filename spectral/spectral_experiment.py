import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.sparse.linalg import eigsh
from pretrainedModel import pretrainedModel
from tensorflow import keras
from PIL import Image
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import time
import warnings


warnings.filterwarnings("ignore")

def mean_average_precision(test_hashes, training_hashes, test_labels, training_labels):
    aps = []
    for i, test_hash in enumerate(test_hashes):
        label = test_labels[i]
        distances = np.abs(training_hashes - test_hashes[i]).sum(axis=1)
        tp = np.where(training_labels==label, 1, 0)
        hash_df = pd.DataFrame({"distances":distances, "tp":tp}).reset_index()
        hash_df = hash_df.sort_values(["distances", "index"]).reset_index(drop=True)
        hash_df = hash_df.drop(["index", "distances"], axis=1).reset_index()
        hash_df = hash_df[hash_df["tp"]==1]
        hash_df["tp"] = hash_df["tp"].cumsum()
        hash_df["index"] = hash_df["index"] +1 
        precision = np.array(hash_df["tp"]) / np.array(hash_df["index"])
        ap = precision.mean()
        aps.append(ap)
    
    return np.array(aps).mean()

def p_at_k(test_hashes, training_hashes, test_labels, training_labels, ks):
    k_dic = {k:[] for k in ks}
    for i, test_hash in enumerate(test_hashes):
        label = test_labels[i]
        distances = np.abs(training_hashes - test_hashes[i]).sum(axis=1)
        tp = np.where(training_labels==label, 1, 0)
        hash_df = pd.DataFrame({"distances":distances, "tp":tp}).reset_index()
        hash_df = hash_df.sort_values(["distances", "index"]).reset_index(drop=True)
        for k in ks:
            df_temp = hash_df[:k]
            patk = df_temp["tp"].sum()/k
            k_dic[k].append(patk)
    return tuple([np.array(k_dic[k]).mean() for k in ks])

X_train = np.load(r'C:\Users\Test\Desktop\p7\Spectral\features\train_features_vgg16_cifar10.npy')
X_test = np.load(r'C:\Users\Test\Desktop\p7\Spectral\features\test_features_vgg16_cifar10.npy')
y_train = np.load(r'C:\Users\Test\Desktop\p7\Spectral\features\train_labels_vgg16_cifar10.npy')
y_test = np.load(r'C:\Users\Test\Desktop\p7\Spectral\features\test_labels_vgg16_cifar10.npy')

X_val = X_train[-5000:,:]
X_train = X_train[:-5000,:]
y_val = y_train[-5000:]
y_train = y_train[:-5000]


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Standardize the data
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


pca = PCA(n_components=100)  # Set the number of components to keep
training_features = pca.fit_transform(X_train)  # Fit PCA on the standardized data and transform
val_features = pca.transform(X_val)
test_features = pca.transform(X_test)

ks = [i for i in range(50, 1050, 50)]

#results_df = pd.DataFrame(columns=["Neighbours", "Bits", "MAP_10000", "time_nearest_neigbours", "time_Laplacian", "time_eigenvectors", "time_mlp"]+[f"P@{k}" for k in ks])
results_df = pd.read_csv(r'C:\Users\Test\Desktop\p7\Spectral\results\spectral_results.csv')

for bits in [32, 64, 128]:
    for neighbours in tqdm(range(1, 10)):

        start = time.time()
        nbrs = NearestNeighbors(n_neighbors=neighbours).fit(training_features)
        # Find the nearest neighbors
        distances, indices = nbrs.kneighbors(training_features)

        # Create an adjacency matrix
        n_samples = training_features.shape[0]
        adjacency_matrix = np.zeros((n_samples, n_samples))

        # Populate the adjacency matrix
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors:
                adjacency_matrix[i, neighbor] = 1
                adjacency_matrix[neighbor, i] = 1  # Ensure symmetry for an undirected graph
        nn_time = time.time() - start

        start = time.time()
        dim=adjacency_matrix.shape[0]
        adjacency_matrix = adjacency_matrix - np.identity(dim)
        D = np.zeros([dim,dim])
        for i in range(dim):
            D[i,i] = adjacency_matrix[i].sum()
        L = D- adjacency_matrix
        L= csr_matrix(L)
        laplace_time = time.time() - start

        start = time.time()
        eigenvalues, eigenvectors = eigsh(L, k=bits+1, which="SM")
        eigenvectors = eigenvectors[:,1:] #remove 0 eigenvalue
        eigenvectors_bin = np.where(eigenvectors > 0, 1, 0)
        eigen_time = time.time() - start

        start = time.time()
        clf = MLPClassifier(hidden_layer_sizes=(100), max_iter=1000).fit(training_features, eigenvectors_bin)
        val_hashes = clf.predict(val_features)
        mlp_time = time.time() - start

        map = mean_average_precision(val_hashes, eigenvectors_bin, y_val, y_train)

        p_at_ks = p_at_k(val_hashes, eigenvectors_bin, y_val, y_train, ks)

        results_df.loc[results_df.shape[0]] = (neighbours, bits, map, nn_time, laplace_time, eigen_time, mlp_time) + p_at_ks

        results_df.to_csv(r'C:\Users\Test\Desktop\p7\Spectral\results\spectral_results.csv', index=False)