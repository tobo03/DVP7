import numpy as np
import tqdm

def meanAveragePrecision(test_hashes, training_hashes, test_labels, training_labels, compare_n):
    aps = []
    training_hashes = training_hashes[:compare_n,:]
    training_labels = training_labels[:compare_n]
    for i, test_hash in enumerate(tqdm(test_hashes)):
        label = test_labels[i]
        distances = np.abs(training_hashes - test_hashes[i]).sum(axis=1)
        tp = np.where(training_labels==label, 1, 0)
        hash_df = pd.DataFrame({"distances":distances, "tp":tp}).reset_index()
        hash_df = hash_df.drop(index=i)
        hash_df = hash_df.sort_values(["distances", "index"]).reset_index(drop=True)
        hash_df = hash_df.drop(["index", "distances"], axis=1).reset_index()
        hash_df = hash_df[hash_df["tp"]==1]
        hash_df["tp"] = hash_df["tp"].cumsum()
        hash_df["index"] = hash_df["index"] +1 
        precision = np.array(hash_df["tp"]) / np.array(hash_df["index"])
        ap = precision.mean()
        aps.append(ap)
    
    return np.array(aps).mean()
