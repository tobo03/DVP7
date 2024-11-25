import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.sparse.linalg import eigsh
#from pretrainedModel import pretrainedModel
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
import optuna
import os


warnings.filterwarnings("ignore")

# Master file for hyperparameter tuning

#MAP
def one_hot_encode(a):
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    return b

def mean_average_precision(test_hashes, training_hashes, test_labels, training_labels):
    aps = []
    if len(training_labels.shape) == 1:
        training_labels = one_hot_encode(training_labels)
        test_labels = one_hot_encode(test_labels)
    for i, test_hash in enumerate(tqdm(test_hashes)):
        label = test_labels[i]
        distances = np.abs(training_hashes - test_hashes[i]).sum(axis=1)
        tp = np.where((training_labels*label).sum(axis=1)>0, 1, 0)
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

# ...
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

# Anch adj for spectral. 
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

def objective(trial):
    n_anchors = trial.suggest_int("n_anchors",int(p/2), int(p * 3) , step= int(p/10))
    s = trial.suggest_int("s", int(p/50), int(p/2), int(p/50))
    #n_anchors = trial.suggest_int("n_anchors",50, 150 , step= 50) Test thingies
    #s = trial.suggest_int("s", 5, 25, 5)

    M, Z, L = anchor_adjecency(training_features, n_anchors=n_anchors, s=s, t=1)
    try:
        eigenvalues, eigenvectors = eigsh(M, k=49, which="LM")
    except RuntimeError:
        return float("-inf")  # Return a poor score if eigsh fails

    eigenvalues = eigenvalues[:-1]
    eigenvectors = eigenvectors[:, :-1]
    bits = 32  # Fix one bit length for evaluation

    eigenvalues_selected = eigenvalues[-bits:]
    eigenvectors_selected = eigenvectors[:, -bits:]

    S = np.flip(1 / np.sqrt(eigenvalues_selected)) * np.identity(eigenvalues_selected.shape[0])
    V = eigenvectors_selected

    Y = np.sqrt(training_features.shape[0]) * Z @ L @ V @ S
    eigenvectors_bin = np.where(Y > 0, 1, 0)

    clf = MLPClassifier(hidden_layer_sizes=(100), max_iter=1000).fit(training_features, eigenvectors_bin)
    val_hashes = clf.predict(val_features)

    map_score = mean_average_precision(val_hashes, eigenvectors_bin, y_val, y_hpo)
    # Save trial data
    results.append({
        "n_anchors": n_anchors,
        "s": s,
        "MAP": map_score
    })
    return map_score

#study = optuna.create_study(direction="maximize")
#study.optimize(objective, n_trials=40)  # Run for 20 trials, do more if desired
#print("Best hyperparameters:", study.best_params)

dir_list = [r"c:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Cifar",r"C:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Nus Wide",r"C:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Imagenet"]
for i in range(3): 
    data_dir = dir_list[i]
    names = ["Cifar","Nus_Wide","Imagenet"]
    # Iterate over the files in the directory
    for filename in os.listdir(dir_list[i]):
        if "X_hpo" in filename:
            X_hpo = np.load(os.path.join(data_dir, filename))
            print(f"Loaded {filename} into X_hpo")
        elif "y_hpo" in filename:
            y_hpo = np.load(os.path.join(data_dir, filename))
            print(f"Loaded {filename} into y_hpo")
        elif "X_val" in filename:
            X_val = np.load(os.path.join(data_dir, filename))
            print(f"Loaded {filename} into X_val")
        elif "y_val" in filename:
            y_val = np.load(os.path.join(data_dir, filename))
            print(f"Loaded {filename} into y_val")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_hpo)  # Standardize the data
    X_val = scaler.transform(X_val)
    pca = PCA(n_components=128)  # Set the number of components to keep
    training_features = pca.fit_transform(X_hpo)  # Fit PCA on the standardized data and transform
    val_features = pca.transform(X_val)

    p = len(training_features)/100
    results = []
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)  # Run for 20 trials, do more if desired
    print("Best hyperparameters:", study.best_params)
    results_df = pd.DataFrame(results)
    results_df.to_csv(fr"spectral_results_{names[i]}_HPO.csv")
# BTW, jeg brugte ikke chatten særligt meget 