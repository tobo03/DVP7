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
from sklearn.cluster import KMeans
import sys


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

def calculate_top_k_distances(anchors, data_points, k=20):
    # Convert anchors and data_points to numpy arrays for easy computation
    anchors = np.array(anchors)
    data_points = np.array(data_points)
    top_k_distances = np.zeros((data_points.shape[0], k))
    top_k_indices = np.zeros((data_points.shape[0], k), dtype=int)

    for i, data_point in enumerate(data_points):
        # Compute Euclidean distances between this data point and all anchors
        distances = np.sqrt(np.sum((anchors - data_point) ** 2, axis=1))

        k_smallest_indices = np.argpartition(distances, k)[:k]
        k_smallest_distances = distances[k_smallest_indices]
        sorted_indices = np.argsort(k_smallest_distances)
        top_k_distances[i] = k_smallest_distances[sorted_indices]
        top_k_indices[i] = k_smallest_indices[sorted_indices]

    return top_k_distances, top_k_indices

def anchor_adjecency(data, n_anchors, s, t):
    # Step 1: Anchor selection using k-means
    kmeans = KMeans(n_clusters=n_anchors, random_state=42)
    kmeans.fit(data)
    anchors = kmeans.cluster_centers_  # Get the anchor points

    Z = np.zeros([data.shape[0], n_anchors])
    dist,indices = calculate_top_k_distances(anchors, data, k=s)

    for i in range(data.shape[0]):
        for j, index in enumerate(indices[i]):
            top = np.exp(-dist[i][j]/t)
            bottom = 0
            for k in range(indices.shape[1]):
                bottom += np.exp(-dist[i][k]/t)
            Z[i,index] = top/bottom

    L = (1/np.sqrt(Z.sum(axis=0)))*np.identity(Z.shape[1])

    M = L @ Z.T @ Z @ L

    return M, Z, L



root = ''
sys.path.append(root)

X_train = np.load(r"C:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Cifar\X_hpo_Cifar.npy" ) # Shape = (40000, 4096)
y_train = np.load(r"C:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Cifar\y_hpo_Cifar.npy") # Shape = (40000,)
X_val = np.load( r"C:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Cifar\X_val_Cifar.npy" ) # Shape = (40000, 4096)
y_val = np.load(r"C:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Cifar\y_val_Cifar.npy") # Shape = (40000,)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Standardize the data
X_val = scaler.transform(X_val)


pca = PCA(n_components=128)  # Set the number of components to keep
training_features = pca.fit_transform(X_train)  # Fit PCA on the standardized data and transform
val_features = pca.transform(X_val)


results_df = pd.DataFrame(columns=["Anchors", "Top k", "Bits", "MAP"])
#results_df = pd.read_csv(r'C:\Users\Test\Desktop\p7\Spectral\results\spectral_results.csv')
train_len = len(training_features)

p = int(train_len/100)
for n_anchors in range(p, p*5 , p):
    print(n_anchors)
    for s in tqdm(range(int(p/10), int(p/2), 10)):

        M, Z, L = anchor_adjecency(training_features, n_anchors= n_anchors, s=s, t=1)

        eigenvalues, eigenvectors = eigsh(M, k=49,which="LM") # overvej max_iter, tolerance?

        eigenvalues = eigenvalues[:-1]
        eigenvectors = eigenvectors[:,:-1]

        for bits in [12, 24, 32, 48]:
            eigenvalues = eigenvalues[-bits:]
            eigenvectors = eigenvectors[:,-bits:]

            S = np.flip(1/np.sqrt(eigenvalues))*np.identity(eigenvalues.shape[0])
            V = eigenvectors

            Y = np.sqrt(training_features.shape[0]) * Z @ L @ V @ S

            threshold1 = 0
            eigenvectors_bin = np.where(Y > threshold1, 1, 0)

            clf = MLPClassifier(hidden_layer_sizes=(100), max_iter=1000).fit(training_features, eigenvectors_bin)
            val_hashes = clf.predict(val_features)

            map = mean_average_precision(val_hashes, eigenvectors_bin, y_val, y_train)

            results_df.loc[results_df.shape[0]] = (train_len/n_anchors, train_len / s, bits, map)

            results_df.to_csv(r'C:\Users\Test\Desktop\p7\Spectral\results\spectral_results_cifar10.csv', index=False)

print("cifar10 done")






root = ''
sys.path.append(root)

X_train = np.load(r"C:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Nus Wide\X_hpo_Nus.npy") # Shape = (40000, 4096)
y_train = np.load(r"C:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Nus Wide\y_hpo_Nus.npy") # Shape = (40000,)
X_val = np.load(r"C:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Nus Wide\X_val_Nus.npy") # Shape = (40000, 4096)
y_val = np.load(r"C:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Nus Wide\y_val_Nus.npy") # Shape = (40000,)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Standardize the data
X_val = scaler.transform(X_val)


pca = PCA(n_components=128)  # Set the number of components to keep
training_features = pca.fit_transform(X_train)  # Fit PCA on the standardized data and transform
val_features = pca.transform(X_val)


results_df = pd.DataFrame(columns=["Anchors", "Top k", "Bits", "MAP"])
#results_df = pd.read_csv(r'C:\Users\Test\Desktop\p7\Spectral\results\spectral_results.csv')

for n_anchors in range(100, 1000, 50):
    print(n_anchors)
    for s in tqdm(range(10, 100, 10)):

        M, Z, L = anchor_adjecency(training_features, n_anchors= n_anchors, s=s, t=1)

        eigenvalues, eigenvectors = eigsh(M, k=49,which="LM") # overvej max_iter, tolerance?

        eigenvalues = eigenvalues[:-1]
        eigenvectors = eigenvectors[:,:-1]

        for bits in [12, 24, 32, 48]:
            eigenvalues = eigenvalues[-bits:]
            eigenvectors = eigenvectors[:,-bits:]

            S = np.flip(1/np.sqrt(eigenvalues))*np.identity(eigenvalues.shape[0])
            V = eigenvectors

            Y = np.sqrt(training_features.shape[0]) * Z @ L @ V @ S

            threshold1 = 0
            eigenvectors_bin = np.where(Y > threshold1, 1, 0)

            clf = MLPClassifier(hidden_layer_sizes=(100), max_iter=1000).fit(training_features, eigenvectors_bin)
            val_hashes = clf.predict(val_features)

            map = mean_average_precision(val_hashes, eigenvectors_bin, y_val, y_train)

            results_df.loc[results_df.shape[0]] = (n_anchors, s, bits, map)

            results_df.to_csv(r'C:\Users\Test\Desktop\p7\Spectral\results\spectral_results_nuswide.csv', index=False)




print("nuswide done")

root = ''
sys.path.append(root)

X_train = np.load(r"C:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Imagenet\X_hpo_Img.npy") # Shape = (40000, 4096)
y_train = np.load(r"C:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Imagenet\y_hpo_Img.npy") # Shape = (40000,)
X_val = np.load(r"C:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Imagenet\X_val_Img.npy") # Shape = (40000, 4096)
y_val = np.load(r"C:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Imagenet\y_val_Img.npy") # Shape = (40000,)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Standardize the data
X_val = scaler.transform(X_val)


pca = PCA(n_components=128)  # Set the number of components to keep
training_features = pca.fit_transform(X_train)  # Fit PCA on the standardized data and transform
val_features = pca.transform(X_val)


results_df = pd.DataFrame(columns=["Anchors", "Top k", "Bits", "MAP"])
#results_df = pd.read_csv(r'C:\Users\Test\Desktop\p7\Spectral\results\spectral_results.csv')

for n_anchors in range(100, 1000, 50):
    print(n_anchors)
    for s in tqdm(range(10, 100, 10)):

        M, Z, L = anchor_adjecency(training_features, n_anchors= n_anchors, s=s, t=1)

        eigenvalues, eigenvectors = eigsh(M, k=49,which="LM") # overvej max_iter, tolerance?

        eigenvalues = eigenvalues[:-1]
        eigenvectors = eigenvectors[:,:-1]

        for bits in [12, 24, 32, 48]:
            eigenvalues = eigenvalues[-bits:]
            eigenvectors = eigenvectors[:,-bits:]

            S = np.flip(1/np.sqrt(eigenvalues))*np.identity(eigenvalues.shape[0])
            V = eigenvectors

            Y = np.sqrt(training_features.shape[0]) * Z @ L @ V @ S

            threshold1 = 0
            eigenvectors_bin = np.where(Y > threshold1, 1, 0)

            clf = MLPClassifier(hidden_layer_sizes=(100), max_iter=1000).fit(training_features, eigenvectors_bin)
            val_hashes = clf.predict(val_features)

            map = mean_average_precision(val_hashes, eigenvectors_bin, y_val, y_train)

            results_df.loc[results_df.shape[0]] = (n_anchors, s, bits, map)

            results_df.to_csv(r'C:\Users\Test\Desktop\p7\Spectral\results\spectral_results_imagenet.csv', index=False)