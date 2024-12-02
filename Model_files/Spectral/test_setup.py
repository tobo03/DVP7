import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.sparse.linalg import eigsh
#from pretrainedModel import pretrainedModel
#from tensorflow import keras
from PIL import Image
from sklearn.preprocessing import StandardScaler
#import torch
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
import metrics # add .final her
import din_mor # Kig ovenover 
from sklearn.model_selection import train_test_split
import datetime


def calculate_top_k_distances(anchors, data_points, k=20):
    # Convert anchors and data_points to numpy arrays for easy computation
    #st = time.time()
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
    #end = time.time()
   # print(f"Calculate top k distances time elapsed: {end - st}") 
    return top_k_distances, top_k_indices

def anchor_adjecency(data, n_anchors, s, t):
    # Step 1: Anchor selection using k-means
    print(":    doing kmeans    :")
   # st = time.time()
    kmeans = KMeans(n_clusters=n_anchors, random_state=42,max_iter = 10)
    kmeans.fit(data)
    anchors = kmeans.cluster_centers_  # Get the anchor points
   # end = time.time()
   # print(f"time elapsed k_means: {end-st}")
    Z = np.zeros([data.shape[0], n_anchors])
    dist,indices = calculate_top_k_distances(anchors, data, k=s)
   # st = time.time()
    for i in range(data.shape[0]):
        for j, index in enumerate(indices[i]):
            top = np.exp(-dist[i][j]/t)
            bottom = 0
            for k in range(indices.shape[1]):
                bottom += np.exp(-dist[i][k]/t)
            Z[i,index] = top/bottom
   # end = time.time()
   # print(f"time elapsed for exp dist {end - st}")
    L = (1/np.sqrt(Z.sum(axis=0)))*np.identity(Z.shape[1])

    M = L @ Z.T @ Z @ L

    return M, Z, L


def train_spectral(training_features, n_anchors, s, bits):
    print("Finding anchors")
    M, Z, L = anchor_adjecency(training_features, n_anchors= n_anchors, s=s, t=1)
    print("Finding eigenvalues")
    eigenvalues, eigenvectors = eigsh(M, k=bits+1, which="LM", maxiter = 1000,tol = 1*10**-4) # overvej maxiter, tolerance? tolerance =  1*10**-4
    eigenvalues = eigenvalues[:-1]
    eigenvectors = eigenvectors[:,:-1]
    print("Eigenvalues found")
    S = np.flip(1/np.sqrt(eigenvalues))*np.identity(eigenvalues.shape[0])
    V = eigenvectors
    Y = np.sqrt(training_features.shape[0]) * Z @ L @ V @ S
    threshold1 = 0
    eigenvectors_bin = np.where(Y > threshold1, 1, 0)
    print("running MLP")
    clf = MLPClassifier(hidden_layer_sizes=(100), max_iter=1000).fit(training_features, eigenvectors_bin)
    return clf


spectral_hpo_dic = {12:{"Cifar":[0.026, 0.0046], "Nus_Wide":[0.005, 0.0034], "Imagenet":[0.005, 0.0002]},
                   24:{"Cifar":[0.024, 0.0024], "Nus_Wide":[0.014, 0.0032], "Imagenet":[0.006, 0.0012]},
                   32:{"Cifar":[0.026, 0.002], "Nus_Wide":[0.015, 0.0032], "Imagenet":[0.005, 0.0002]},
                   48:{"Cifar":[0.03, 0.0014], "Nus_Wide":[0.018, 0.0016], "Imagenet":[0.006, 0.0012]}}
        
results_df = pd.DataFrame(columns=["dataset", "bits", "training_time", "query_time", "map", "p@k_1", "p@k_2", "p@k_3", "p@0", "p@1", "p@2", "p@3"])

model_name = "spectral"

bits = [12, 24, 32, 48]

dir_list = [r"c:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Cifar",r"C:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Nus Wide",r"C:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Imagenet"]
for i in range(1): # QUick fix to run it just once 
    i = 2
   
    print(datetime.datetime.now())
    data_dir = dir_list[i]
    names = ["Cifar","Nus_Wide","Imagenet"]
    # Iterate over the files in the directory
    for filename in os.listdir(dir_list[i]):
        if "X_train" in filename:
            database = np.load(os.path.join(data_dir, filename))
            #database = database[:55000]
            print(f"db len: {len(database)}")
            print(f"Loaded {filename} into X_train")
        elif "y_train" in filename:
            database_labels = np.load(os.path.join(data_dir, filename))
            #database_labels = database_labels[:55000]
            print(f"Loaded {filename} into y_train")
        elif "X_test" in filename:
            query = np.load(os.path.join(data_dir, filename))
            query = query[:10000] # burde måske randomsample 
            print(f"query len: {len(query)}")
            print(f"Loaded {filename} into X_test")
        elif "y_test" in filename:
            query_labels = np.load(os.path.join(data_dir, filename))
            query_labels = query_labels[:10000] # Burde måske randomsample...
            #print(np.sum(query_labels,axis = 0))
            #print(query_labels[:5])

            print(f"Loaded {filename} into y_test")
            

    if i ==2:
        ks = [10000, 50000, 100000]
        _, training_set, _, training_labels = train_test_split(database, database_labels, train_size=1-0.20, random_state=42, stratify=database_labels)
        _ = 0
        database = training_set  # new
        database_labels = training_labels # new 
    else: 
        ks = [1000, 5000, 10000]
        training_set = database
        training_labels = database_labels

    dists = [0,1,2,3]

    for bit in bits:
        pr_df = pd.DataFrame(columns=["precision", "recall"])
        n_anchors = round((training_set.shape[0]*spectral_hpo_dic[bit][names[i]][0])/2) # fjernede /3 for få anchors ( vigtig note) : /3 virkede bedre... og hurtigere

        print(f"n_anchors: {n_anchors}")
        s = round((training_set.shape[0]*spectral_hpo_dic[bit][names[i]][1])/2) # Fjernede /3 midlertidig 
        print(names[i], bit)
        training_time_start = time.time()
        #train model
        model = train_spectral(training_set, n_anchors, s, bit)
        training_time = time.time() - training_time_start 

        #run database through model
        print("model_training done")
        database_hashes = model.predict(database)
        #run query set through model
        query_hashes = model.predict(query)

        print("map")
        st = time.time()
        map_score = metrics.meanAveragePrecisionOptimized(query_hashes, database_hashes, query_labels, database_labels)
        print(f"time optimized map {time.time() - st}")
        print(map_score)

        #st = time.time()
       # map_score_2 = metrics.meanAveragePrecision(query_hashes, database_hashes, query_labels, database_labels)
      # print(f"time normal map {time.time() - st}")
       # print(map_score_2)

        print("p@k")
        p_at_k = metrics.p_at_k(query_hashes, database_hashes, query_labels, database_labels, ks)
        print("p@d")
        p_at_dist = metrics.p_at_dist(query_hashes, database_hashes, query_labels, database_labels, dists) # add optimized!!! 
        x = din_mor # Giver med vilje error så du ændrer p_at_dist, p_at_k etc.
        print("precision recall")
        recall, precision = metrics.interpolated_pr_curve(query_hashes, database_hashes, query_labels, database_labels, num_points=100) # add optimized 
            
        print("query time")
        query_hash = query_hashes[22]
        query_time_start = time.time()
        # run single image thorugh model
        _ = metrics.query(query_hash, database_hashes)
        query_time = time.time() - query_time_start
            
        results_df.loc[results_df.shape[0]] = (names[i], bit, training_time, query_time, map_score) + p_at_k + p_at_dist
        results_df.to_csv(f"{model_name}_{names[i]}_testing.csv", index=False) # addet csv

        pr_df = pd.DataFrame(data=np.array([precision, recall]).T, columns=["precision", "recall"])
        pr_df.to_csv(f"{model_name}_{names[i]}_{bit}_pr_curve.csv") # addet csv 
        print(datetime.datetime.now())

        
