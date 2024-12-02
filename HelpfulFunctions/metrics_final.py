import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

def one_hot_encode(a):
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    return b

def meanAveragePrecision(test_hashes, training_hashes, test_labels, training_labels):
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

def meanAveragePrecisionOptimized(test_hashes, training_hashes, test_labels, training_labels):
    aps = []
    if len(training_labels.shape) == 1:
        training_labels = one_hot_encode(training_labels)
        test_labels = one_hot_encode(test_labels)

    for i, test_hash in enumerate(tqdm(test_hashes)):
        label = test_labels[i]
        # Compute distances
        distances = np.abs(training_hashes - test_hash).sum(axis=1)
        # Determine relevance
        tp = (training_labels @ label) > 0
        # Sort by distance
        #sorted_indices = np.argsort(distances)
        sorted_indices = np.lexsort((np.arange(len(distances)), distances)) # ny attempt. 
        sorted_relevance = tp[sorted_indices]
        # Compute cumulative precision
        relevant_indices = np.where(sorted_relevance)[0] + 1
        precision_at_k = np.arange(1, len(relevant_indices) + 1) / relevant_indices
        aps.append(precision_at_k.mean() if len(precision_at_k) > 0 else 0.0)

    return np.mean(aps)

def query(test_hash, training_hashes):
    distances = np.abs(training_hashes - test_hash).sum(axis=1)
    hash_df = pd.DataFrame({"distances":distances}).reset_index()
    hash_df = hash_df.sort_values(["distances", "index"]).reset_index(drop=True)
    return np.array(hash_df["index"])

def query_optimized(test_hash, training_hashes):
    distances = np.abs(training_hashes - test_hash).sum(axis=1)
    #sorted_indices = np.lexsort((np.arange(len(distances)), distances))
    return np.lexsort((np.arange(len(distances)), distances)) #Virker???

def p_at_k(test_hashes, training_hashes, test_labels, training_labels, ks):
    k_dic = {k:[] for k in ks}
    if len(training_labels.shape) == 1:
        training_labels = one_hot_encode(training_labels)
        test_labels = one_hot_encode(test_labels)
    for i, test_hash in tqdm(enumerate(test_hashes)):
        label = test_labels[i]
        distances = np.abs(training_hashes - test_hashes[i]).sum(axis=1)
        tp = np.where((training_labels*label).sum(axis=1)>0, 1, 0)
        hash_df = pd.DataFrame({"distances":distances, "tp":tp}).reset_index()
        hash_df = hash_df.sort_values(["distances", "index"]).reset_index(drop=True)
        for k in ks:
            df_temp = hash_df[:k]
            patk = df_temp["tp"].sum()/k
            k_dic[k].append(patk)
    return tuple([np.array(k_dic[k]).mean() for k in ks])

def p_at_k_optimized(test_hashes, training_hashes, test_labels, training_labels, ks): 
    k_dic = {k: [] for k in ks}
    max_k = max(ks)
    
    # One-hot encode labels if necessary
    if len(training_labels.shape) == 1:
        training_labels = one_hot_encode(training_labels)
        test_labels = one_hot_encode(test_labels)
    for i, test_hash in tqdm(enumerate(test_hashes)):
        label = test_labels[i]
        # Compute Hamming distances
        distances = np.abs(training_hashes - test_hash).sum(axis=1)
        # True positives
        tp = (training_labels @ label) > 0
        # Use lexsort for deterministic sorting
        indices = np.arange(len(distances))
        sorted_indices = np.lexsort((indices, distances))
        # Extract top max_k indices
        top_k_indices = sorted_indices[:max_k]
        # For each k, compute precision
        for k in ks:
            relevant_tp = tp[top_k_indices[:k]]  # Top-k true positives
            precision_at_k = relevant_tp.sum() / k
            k_dic[k].append(precision_at_k)
    
    return tuple(np.mean(k_dic[k]) for k in ks)

def average_pr_curve(precision_list, recall_list, num_points=100):
    recall_levels = np.linspace(0, 1, num_points)
    interpolated_precisions = []
    for precision, recall in zip(precision_list, recall_list):
        sorted_indices = np.argsort(recall)
        recall_sorted = recall[sorted_indices]
        precision_sorted = precision[sorted_indices]
        interpolated_precision = np.interp(recall_levels, recall_sorted, precision_sorted, left=0, right=0)
        interpolated_precisions.append(interpolated_precision)
    avg_precision = np.mean(interpolated_precisions, axis=0)
    return recall_levels, avg_precision

def interpolated_pr_curve(test_hashes, training_hashes, test_labels, training_labels, num_points=100):
    precision_list = []
    recall_list = []
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
        hash_df["tp"] = hash_df["tp"].cumsum()
        hash_df["index"] = hash_df["index"] +1 
        precision = np.array(hash_df["tp"]) / np.array(hash_df["index"])
        recall = np.array(hash_df["tp"]) / np.array(hash_df["tp"].max())
        precision_list.append(precision)
        recall_list.append(recall)

    recall_levels, avg_precision = average_pr_curve(precision_list, recall_list, num_points=num_points)
    
    return recall_levels, avg_precision

def interpolated_pr_curve_optimized(test_hashes, training_hashes, test_labels, training_labels, num_points=100): # InterPol???
    precision_list = []
    recall_list = []

    # One-hot encode labels if necessary
    if len(training_labels.shape) == 1:
        training_labels = one_hot_encode(training_labels)
        test_labels = one_hot_encode(test_labels)
    
    for i, test_hash in tqdm(enumerate(test_hashes)):
        label = test_labels[i]
        # Compute Hamming distances
        distances = np.abs(training_hashes - test_hash).sum(axis=1)
        # True positives
        tp = (training_labels @ label) > 0
        # Use lexsort for deterministic sorting
        indices = np.arange(len(distances))
        sorted_indices = np.lexsort((indices, distances))
        # Sorted true positives
        sorted_tp = tp[sorted_indices]
        # Compute cumulative true positives
        cumulative_tp = np.cumsum(sorted_tp)
        # Number of retrieved items
        num_retrieved = np.arange(1, len(sorted_tp) + 1)
        # Precision and recall
        precision = cumulative_tp / num_retrieved
        recall = cumulative_tp / cumulative_tp[-1]
        precision_list.append(precision)
        recall_list.append(recall)
    
    # Compute interpolated average precision at recall levels
    recall_levels, avg_precision = average_pr_curve(precision_list, recall_list, num_points=num_points)
    
    return recall_levels, avg_precision    

def p_at_dist(test_hashes, training_hashes, test_labels, training_labels, dists):
    k_dic = {k:[] for k in dists}
    if len(training_labels.shape) == 1:
        training_labels = one_hot_encode(training_labels)
        test_labels = one_hot_encode(test_labels)
    for i, test_hash in tqdm(enumerate(test_hashes)):
        label = test_labels[i]
        distances = np.abs(training_hashes - test_hashes[i]).sum(axis=1)
        tp = np.where((training_labels*label).sum(axis=1)>0, 1, 0)
        hash_df = pd.DataFrame({"distances":distances, "tp":tp}).reset_index()
        hash_df = hash_df.sort_values(["distances", "index"]).reset_index(drop=True)
        for k in dists:
            df_temp = hash_df[hash_df["distances"]<=k]
            patk = df_temp["tp"].sum()/df_temp.shape[0] if df_temp.shape[0]>0 else 0
            k_dic[k].append(patk)
    return tuple([np.array(k_dic[k]).mean() for k in dists])

def p_at_dist_optimized(test_hashes, training_hashes, test_labels, training_labels, dists):
    k_dic = {k: [] for k in dists}
    # One-hot encode labels if necessary
    if len(training_labels.shape) == 1:
        training_labels = one_hot_encode(training_labels)
        test_labels = one_hot_encode(test_labels)
    # Maximum distance to consider
    max_dist = max(dists)
    for i, test_hash in tqdm(enumerate(test_hashes)):
        label = test_labels[i]
        # Compute Hamming distances
        distances = np.abs(training_hashes - test_hash).sum(axis=1)
        tp = (training_labels @ label) > 0
        # Filter indices by distance thresholds
        within_max_dist = distances <= max_dist
        filtered_distances = distances[within_max_dist]
        filtered_tp = tp[within_max_dist]
        indices = np.arange(len(filtered_distances))
        sorted_indices = np.lexsort((indices, filtered_distances))
        sorted_distances = filtered_distances[sorted_indices]
        sorted_tp = filtered_tp[sorted_indices]
        for k in dists:
            in_dist = sorted_distances <= k
            num_in_dist = np.sum(in_dist)
            if num_in_dist > 0:
                patk = sorted_tp[in_dist].sum() / num_in_dist
            else:
                patk = 0
            k_dic[k].append(patk)
    return tuple(np.mean(k_dic[k]) for k in dists)

