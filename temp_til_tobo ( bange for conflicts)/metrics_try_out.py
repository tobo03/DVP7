import metrics_new
import numpy as np
import time
import datetime




# Example usage
test_hashes = np.random.randint(0, 2, (5000, 32))  # 100 queries, 32-bit hashes
training_hashes = np.random.randint(0, 2, (100000, 32))  # 1000 samples, 32-bit hashes
test_labels = np.random.randint(0, 5, 5000)  # 5 classes
training_labels = np.random.randint(0, 5, 100000)
ks = [1, 5, 10, 20]

test_hashes, training_hashes, test_labels, training_labels, ks

st = time.time()
z = metrics_new.p_at_k_optimized(test_hashes,training_hashes,test_labels,training_labels,ks)
print(f"time elapsed optimized: {time.time()- st} , with p @k: {z}")

st = time.time()
x = metrics_new.p_at_k(test_hashes,training_hashes,test_labels,training_labels,ks)
print(f"time elapsed normal: {time.time()- st} , with p @k: {x}")