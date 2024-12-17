import numpy as np
import metrics_final
import pandas as pd
from sklearn.model_selection import train_test_split

dists = [0,1,2,3]
ks = [10000, 50000, 100000]
model = "dtsh"

results_df = pd.DataFrame(columns=["bits", "map", "p@k_1", "p@k_2", "p@k_3", "p@0", "p@1", "p@2", "p@3"])

for bit in [12, 24, 32, 48]:
    print(bit)
    database_hashes = np.load(f"/Users/kristiansjorslevnielsen/Documents/DVP7/DeepHashing/Pointwise/imagenet_tests/saved_hashes/database_{model}_{bit}.npy").astype(int)
    database_labels = np.load(f"/Users/kristiansjorslevnielsen/Documents/DVP7/DeepHashing/Pointwise/imagenet_tests/saved_hashes/database_labels_{model}_{bit}.npy").astype(int)
    query_hashes = np.load(f"/Users/kristiansjorslevnielsen/Documents/DVP7/DeepHashing/Pointwise/imagenet_tests/saved_hashes/query_{model}_{bit}.npy").astype(int)
    query_labels = np.load(f"/Users/kristiansjorslevnielsen/Documents/DVP7/DeepHashing/Pointwise/imagenet_tests/saved_hashes/query_labels_{model}_{bit}.npy").astype(int)


    print(database_hashes.shape)
    print(database_labels.shape)
    print(query_labels.shape)
    print("map")
    #map_score = metrics_final.meanAveragePrecisionOptimized(query_hashes, database_hashes, query_labels, database_labels)

    print("p@k")
    #p_at_k = metrics_final.p_at_k_optimized(query_hashes, database_hashes, query_labels, database_labels, ks)
    print("p@d")
    #p_at_dist = metrics_final.p_at_dist_optimized(query_hashes, database_hashes, query_labels, database_labels, dists) # add optimized!!! 
    _, pr_query_hashes, _, pr_query_labels = train_test_split(query_hashes, query_labels, train_size=1-0.10, random_state=42, stratify=query_labels)
    recall, precision = metrics_final.interpolated_pr_curve_optimized(pr_query_hashes, database_hashes, pr_query_labels, database_labels, num_points=100) # add optimized 
    pr_df = pd.DataFrame(data=np.array([precision, recall]).T, columns=["precision", "recall"])
    pr_df.to_csv(f"{model}_imagenet_{bit}_pr_curve.csv") # addet csv 

    #results_df.loc[results_df.shape[0]] = (bit, map_score) + p_at_k + p_at_dist
    #results_df.to_csv(f"{model}_imagenet_results.csv", index = False)

