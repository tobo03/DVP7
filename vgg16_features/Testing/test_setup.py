import metrics
import pandas as pd
import numpy as np
import time
import os

results_df = pd.DataFrame(columns=["model", "dataset", "training_time", "query_time", "map", "p@k_1", "p@k_2", "p@k_3", "p@0", "p@1", "p@2", "p@3"])




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
    
    if i ==2:
        ks = [10000, 50000, 100000]
    
    else: 
        ks = [1000, 5000, 10000]

    dists = [0,1,2,3]
    precision_list = []
    pr_df = pd.DataFrame(columns=["precision", "recall"])


    for i in range(10):
        training_time_start = time.time()
        #training
        training_time = time.time() - training_time_start 

        #run database through model
        #run query set through model

        map_score = metrics.meanAveragePrecision(query_hashes, database_hashes, query_labels, database_labels)
        p_at_k = metrics.p_at_k(query_hashes, database_hashes, query_labels, database_labels, ks)
        p_at_dist = metrics.p_at_dist(query_hashes, database_hashes, query_labels, database_labels, dists)
        recall, precision = metrics.interpolated_pr_curve(query_hashes, database_hashes, query_labels, database_labels, num_points=100)
        precision_list.append(precision)

        query_time_start = time.time()
        # run single image thorugh model
        _ = metrics.query(query_hash, database_hashes)
        query_time = time.time() - query_time_start
        
        results_df

        
