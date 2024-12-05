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
import metrics_final # add 
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
        
#results_df = pd.DataFrame(columns=["dataset", "bits", "training_time", "query_time", "map", "p@k_1", "p@k_2", "p@k_3", "p@0", "p@1", "p@2", "p@3"])
results_df = pd.read_csv(r"C:\Users\Test\Desktop\Local_folder\spectral_Imagenet_testing.csv")

model_name = "dtsh"

bits = [32, 48] #  udkommenteret.

names = ["Cifar","Nus_Wide","Imagenet"]

def load_data(i):
    dir_list = [r"c:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Cifar",r"C:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Nus Wide",r"C:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Imagenet"]
    print(datetime.datetime.now())
    data_dir = dir_list[i]
    names = ["Cifar","Nus_Wide","Imagenet"]
    # Iterate over the files in the directory
    for filename in os.listdir(dir_list[i]):
        if "X_train" in filename:
            training = np.load(os.path.join(data_dir, filename))
            #database = database[:55000]
            print(f"Loaded {filename} into X_train")
        elif "y_train" in filename:
            training_labels = np.load(os.path.join(data_dir, filename))
            #database_labels = database_labels[:55000]
            print(f"Loaded {filename} into y_train")
        elif "X_test" in filename:
            query = np.load(os.path.join(data_dir, filename)) 
            print(f"query len: {len(query)}")
            print(f"Loaded {filename} into X_test")
        elif "y_test" in filename:
            query_labels = np.load(os.path.join(data_dir, filename))
            #print(np.sum(query_labels,axis = 0))
            #print(query_labels[:5])

            print(f"Loaded {filename} into y_test")
    return training, training_labels, query, query_labels


for i in range(1): # QUick fix to run it just once (Imagenet)
    i = 2

    database, database_labels, query, query_labels = load_data(i)
            

    if i ==2:
        ks = [10000, 50000, 100000]
        _, database, _, database_labels = train_test_split(database, database_labels, train_size=1-0.10, random_state=42, stratify=database_labels)
        _ = 0
      

    else: 
        ks = [1000, 5000, 10000]
        #database = training
        #database_labels = training_labels

    dists = [0,1,2,3]

    for bit in bits:
        pr_df = pd.DataFrame(columns=["precision", "recall"])
        n_anchors = round((database.shape[0]*spectral_hpo_dic[bit][names[i]][0])/1.5) # fjernede /3 for få anchors ( vigtig note) : /3 virkede bedre... og hurtigere
        """
        Traceback (most recent call last):
              File "C:\Users\Test\Desktop\Local_folder\test_setup.py", line 156, in <module>
                n_anchors = round((database.shape[0]*spectral_hpo_dic[bit][names[i]][0])/1.5) # fjernede /3 for få anchors ( vigtig note) : /3 virkede bedre... og hurtigere
            AttributeError: 'int' object has no attribute 'shape'
            PS C:\Users\Test\Desktop\Local_folder>

        """
        print(f"n_anchors: {n_anchors}") 
        s = round((database.shape[0]*spectral_hpo_dic[bit][names[i]][1])/1.5) # Fjernede /3 midlertidig 
        print(names[i], bit)
        training_time_start = time.time()
        #train model
        model = train_spectral(database, n_anchors, s, bit)
        training_time = time.time() - training_time_start 

        #run database through model
        print("model_training done")
        database_hashes = model.predict(database)
        #run query set through model
        query_hashes = model.predict(query)

        if i != 2:
            print("map")
            map_score = metrics_final.meanAveragePrecisionOptimized(query_hashes, database_hashes, query_labels, database_labels)

            print("p@k")
            p_at_k = metrics_final.p_at_k_optimized(query_hashes, database_hashes, query_labels, database_labels, ks)
            print("p@d")
            p_at_dist = metrics_final.p_at_dist_optimized(query_hashes, database_hashes, query_labels, database_labels, dists) # add optimized!!! 
            print("precision recall")
            _, pr_query_hashes, _, pr_query_labels = train_test_split(query_hashes, query_labels, train_size=1-0.10, random_state=42, stratify=query_labels)
            recall, precision = metrics_final.interpolated_pr_curve_optimized(pr_query_hashes, database_hashes, pr_query_labels, database_labels, num_points=100) # add optimized 

            print("query time")
        

        elif i == 2:
            np.save(fr"C:\Users\Test\Desktop\Local_folder\Hash_codes_db\database_{model_name}_{bit}.npy",database_hashes)
            np.save(fr"C:\Users\Test\Desktop\Local_folder\Hash_codes_db\database_labels_{model_name}_{bit}.npy",database_labels)
            np.save(fr"C:\Users\Test\Desktop\Local_folder\Hash_codes_query\query_{model_name}_{bit}.npy",query_hashes)
            np.save(fr"C:\Users\Test\Desktop\Local_folder\Hash_codes_query\query_labels_{model_name}_{bit}.npy",query_labels)


        database, database_labels, query, query_labels = load_data(i)
        database = model.predict(database)

        query_image = query[22].reshape(1,-1) # Der var fejl her... stod query[22]

        query_time_start = time.time()
        # run single image thorugh model
        query_hash = model.predict(query_image)
        images = metrics_final.query_optimized(query_hash, database) 
        query_time = time.time() - query_time_start
        database = 0
        if i ==2 and bit==48:
            images = images[:5]
            np.save(fr"C:\Users\Test\Desktop\Local_folder\Result_folder\query_{model_name}_5_images.npy", images)
            np.save(fr"C:\Users\Test\Desktop\Local_folder\Result_folder\Hash_codes_query_{model_name}_all_images.numpy",query_hashes)

        if i == 0 and bit == 48:
           np.save(fr"C:\Users\Test\Desktop\Local_folder\Result_folder\tsne_{model_name}_hashes.npy",query_hashes)

        # save down the database and 


        results_df.loc[results_df.shape[0]] = (names[i], bit, training_time, query_time, map_score) + p_at_k + p_at_dist
        results_df.to_csv(f"{model_name}_{names[i]}_testing.csv", index=False) # addet csv

        pr_df = pd.DataFrame(data=np.array([precision, recall]).T, columns=["precision", "recall"])
        pr_df.to_csv(f"{model_name}_{names[i]}_{bit}_pr_curve.csv") # addet csv 
        print(datetime.datetime.now())

        
