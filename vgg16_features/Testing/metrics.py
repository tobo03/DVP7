import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import pairwise_distances

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


def p_at_k(test_hashes, training_hashes, test_labels, training_labels, ks):
    k_dic = {k:[] for k in ks}
    if len(training_labels.shape) == 1:
        training_labels = one_hot_encode(training_labels)
        test_labels = one_hot_encode(test_labels)
    for i, test_hash in enumerate(test_hashes):
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

def interpolated_pr_curve(test_hashes, training_hashes, test_labels, training_labels):
    precision_list = []
    recall_list = []
    for i, test_hash in enumerate(tqdm(test_hashes)):
        label = test_labels[i]
        distances = np.abs(training_hashes - test_hashes[i]).sum(axis=1)
        tp = np.where(training_labels==label, 1, 0)
        hash_df = pd.DataFrame({"distances":distances, "tp":tp}).reset_index()
        hash_df = hash_df.sort_values(["distances", "index"]).reset_index(drop=True)
        hash_df = hash_df.drop(["index", "distances"], axis=1).reset_index()
        hash_df["tp"] = hash_df["tp"].cumsum()
        hash_df["index"] = hash_df["index"] +1 
        precision = np.array(hash_df["tp"]) / np.array(hash_df["index"])
        recall = np.array(hash_df["tp"]) / np.array(hash_df["tp"].max())
        precision_list.append(precision)
        recall_list.append(recall)

    recall_levels, avg_precision = average_pr_curve(precision_list, recall_list)
    
    return recall_levels, avg_precision


def precision_at_radius(test_hash_codes, train_hash_codes, test_labels, train_labels, radius=2):
    """
    Computes precision at a given Hamming radius.
    
    Parameters:
    - test_hash_codes: numpy array of shape (n_test_samples, n_bits), hash codes for test samples
    - train_hash_codes: numpy array of shape (n_train_samples, n_bits), hash codes for train samples
    - test_labels: numpy array of shape (n_test_samples,), labels for test samples
    - train_labels: numpy array of shape (n_train_samples,), labels for train samples
    - radius: int, the Hamming distance threshold to consider as a match
    
    Returns:
    - precision: float, the precision at the given radius
    """
    # Compute Hamming distance between each test and train hash code
    hamming_distances = pairwise_distances(test_hash_codes, train_hash_codes, metric='hamming') * test_hash_codes.shape[1]
    
    correct_retrievals = 0
    total_retrievals = 0
    
    for i in range(len(test_hash_codes)):
        # Get indices of all train samples within the given Hamming radius
        within_radius = np.where(hamming_distances[i] <= radius)[0]
        
        # Count the number of correct matches
        correct_matches = np.sum(train_labels[within_radius] == test_labels[i])
        
        # Update totals for precision calculation
        correct_retrievals += correct_matches
        total_retrievals += len(within_radius)
    
    precision = correct_retrievals / total_retrievals if total_retrievals > 0 else 0
    return precision