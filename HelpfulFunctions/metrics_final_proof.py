import metrics_final
import numpy as np
import time
import datetime




# Example usage
test_hashes = np.random.randint(0, 2, (10000, 32))  # 100 queries, 32-bit hashes
training_hashes = np.random.randint(0, 2, (10000, 32))  # 1000 samples, 32-bit hashes
test_labels = np.random.randint(0, 100, 10000)  # 5 classes
training_labels = np.random.randint(0, 100, 10000)
ks = [1, 5, 10, 20]
dists = [1,2,3,4]
#test_hashes, training_hashes, test_labels, training_labels, ks

# Starter med det langsomme for at ensure fair competition
st = time.time()
x1 = metrics_final.p_at_k(test_hashes,training_hashes,test_labels,training_labels,ks)
x2 = metrics_final.query(test_hashes[22],training_hashes)
x3 = metrics_final.meanAveragePrecision(test_hashes,training_hashes,test_labels,training_labels)
x4,x5 = metrics_final.interpolated_pr_curve(test_hashes,training_hashes,test_labels,training_labels) 
x6 = metrics_final.p_at_dist(test_hashes, training_hashes, test_labels, training_labels, dists)
print(f"time elapsed normal: {time.time()- st} , for all queries")

st = time.time()
z1 = metrics_final.p_at_k_optimized(test_hashes,training_hashes,test_labels,training_labels,ks)
z2 = metrics_final.query_optimized(test_hashes[22],training_hashes)
z3 = metrics_final.meanAveragePrecisionOptimized(test_hashes,training_hashes,test_labels,training_labels)
z4,z5 = metrics_final.interpolated_pr_curve_optimized(test_hashes, training_hashes, test_labels, training_labels) #samme output 
z6 = metrics_final.p_at_dist_optimized(test_hashes, training_hashes, test_labels, training_labels, dists)
print(f"time elapsed optimized: {time.time()- st} , for all queries")
#print(z)

#print(x)
print(x1 == z1) # Yes it is :)
print(x2 == z2) # Also true ( for this dummy set)
print(x3 == z3) # Yes it is :)
print(x4 == z4) # Also true ( for this dummy set)
print(x5 == z5) # Yes it is :)
print(x6 == z6) # Yes it is :)



