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
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import random
import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple
from tqdm import tqdm
from scipy.spatial import distance_matrix
import pandas as pd

torch.cuda.empty_cache()



dtsh_hpo_dic = {12:{"Cifar":[0.5, 5, 0.00010, 0.00010], "Nus_Wide":[0.5, 5, 0.00001, 0.00001], "Imagenet":[0.5, 5, 0.00001, 0.00001]},
                   24:{"Cifar":[1, 5, 0.00010, 0.00010], "Nus_Wide":[0.5, 3, 0.00010, 0.00001], "Imagenet":[0.5, 5, 0.00010, 0.00001]},
                   32:{"Cifar":[0.5, 5, 0.00010, 0.00010], "Nus_Wide":[0.5, 3, 0.00010, 0.00001], "Imagenet":[0.5, 5, 0.00010, 0.00010]},
                   48:{"Cifar":[2, 5, 0.00001, 0.00001], "Nus_Wide":[0.5, 3, 0.00001, 0.00001], "Imagenet":[0.5, 5, 0.00001, 0.00010]}}

def get_dataloader(X_train, y_train, one_hot=True, batchSize = 100):
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)

    if one_hot:
        y_train = torch.nn.functional.one_hot(y_train)

    train_data = []
    for i in range(len(X_train)):
        train_data.append([X_train[i], y_train[i]])
    dataloader = DataLoader(train_data, batch_size=batchSize, shuffle=True)


    return dataloader

def earlyStop(LossList, n = 10):
    bestVal = min(LossList)

    bestVal_i = LossList.index(bestVal)

    if bestVal_i < len(LossList) - n: return True

def train_dtsh(X_train, y_train, LAMBDA, ALPHA, lr, weight_decay, bits, do_one_hot=True, earlyStop_num=20):
    data = {}




    model = nn.Sequential(  nn.Linear(4096,256),
                            nn.ReLU(),
                            nn.Linear(256, bits),
                            )
    model.to(device)

    criterion = DTSHLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr , weight_decay=weight_decay)

    dataloader = get_dataloader(X_train, y_train, one_hot=do_one_hot)
    historical_lostList = []
    for i in range(1500):
        loss_list = []
        for j,batch  in enumerate(dataloader):
            X_batch = batch[0]
            y_batch = batch[1]

            optimizer.zero_grad()

            u = model(X_batch)
            loss = criterion(u, y_batch.float(), LAMBDA=LAMBDA, ALPHA=ALPHA)
            loss.backward()
            optimizer.step()

            loss_list.append( float(loss) )
        
        
        mean_loss = sum(loss_list) / len(loss_list)
        if i % 10 == 1:
            print(i, mean_loss)
        historical_lostList.append(mean_loss)

        if earlyStop(historical_lostList, n = earlyStop_num): 
            print(i, mean_loss)
            print("Early Stop!!!")
            data["earlyStop"] = True
            break
    
    return model

class DTSHLoss(torch.nn.Module):
    def __init__(self):
        super(DTSHLoss, self).__init__()

    def forward(self, u, y, LAMBDA=1, ALPHA=1):
        #LAMBDA = 1
        #ALPHA  = 1

        inner_product = u @ u.t()   # Similarity Matrix
        s = y @ y.t() > 0           # A matrix that show if the two idexes are the same or not
        count = 0

        loss1 = 0
        for row in range(s.shape[0]):
            # if has positive pairs and negative pairs
            if s[row].sum() != 0 and (~s[row]).sum() != 0:
                count += 1
                theta_positive = inner_product[row][s[row] == 1]                
                theta_negative = inner_product[row][s[row] == 0]

                triple = (theta_positive.unsqueeze(1) - theta_negative.unsqueeze(0) - ALPHA ).clamp(min=-100,max=50)
                loss1 += -(triple - torch.log(1 + torch.exp(triple))).mean()

        if count != 0:
            loss1 = loss1 / count
        else:
            loss1 = 0

        loss2 = LAMBDA * (u - u.sign()).pow(2).mean()

        return loss1 + loss2


results_df = pd.DataFrame(columns=["dataset", "bits", "training_time", "query_time", "map", "p@k_1", "p@k_2", "p@k_3", "p@0", "p@1", "p@2", "p@3"])

model_name = "subic"

bits = [12, 24,32, 48] #  udkommenteret.

names = ["Cifar","Nus_Wide","Imagenet"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(i):
    dir_list = [r"c:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Cifar",r"C:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Nus Wide",r"C:\Users\Test\Desktop\FINAL P7 FEATURES !!!!\Imagenet"]
    print(datetime.datetime.now())
    data_dir = dir_list[i]
    names = ["Cifar","Nus_Wide","Imagenet"]
    # Iterate over the files in the directory
    for filename in os.listdir(dir_list[i]):
        if "X_train" in filename:
            training = np.load(os.path.join(data_dir, filename))
            training = training[:40000]
            #database = database[:55000]
            print(f"Loaded {filename} into X_train")
        elif "y_train" in filename:
            training_labels = np.load(os.path.join(data_dir, filename))
            training_labels = training_labels[:40000]
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


for i in range(2): # QUick fix to run it just once (Image)
    i = 1

    dists = [0,1,2,3]

    for bit in bits:
        with torch.no_grad():
            torch.cuda.empty_cache()
        training, training_labels, query, query_labels = load_data(i)
            

        if i ==2:
            ks = [10000, 50000, 100000]
            _, training, _, training_labels = train_test_split(training, training_labels, train_size=1-0.10, random_state=42, stratify=training_labels)
            _ = 0
            print(training.shape)
      

        else: 
            ks = [1000, 5000, 10000]
        #database = training
        #database_labels = training_labels


        pr_df = pd.DataFrame(columns=["precision", "recall"])

        print(names[i], bit)
        training_time_start = time.time()
        #train model
        model = DPSH(device, training, training_labels, bit, 300, 128, dpsh_hpo_dic[bit][names[i]][1], dpsh_hpo_dic[bit][names[i]][2], dpsh_hpo_dic[bit][names[i]][0])
        training_time = time.time() - training_time_start 

        #run database through model
        print("model_training done")


        if i != 2:
            _, training_hashes = model.forward(training, use_one_hot=True)
            training_hashes = training_hashes.cpu().detach().numpy()
            #run query set through model
            _, query_hashes = model.forward(query, use_one_hot=True)
            query_hashes = query_hashes.cpu().detach().numpy()
            print("map")
            map_score = metrics_final.meanAveragePrecisionOptimized(query_hashes, training_hashes, query_labels, training_labels)
            print(map_score)
            print("p@k")
            p_at_k = metrics_final.p_at_k_optimized(query_hashes, training_hashes, query_labels, training_labels, ks)
            print("p@d")
            p_at_dist = metrics_final.p_at_dist_optimized(query_hashes, training_hashes, query_labels, training_labels, dists) # add optimized!!! 
        if i == 0:
            print("precision recall")
            _, pr_query_hashes, _, pr_query_labels = train_test_split(query_hashes, query_labels, train_size=1-0.10, random_state=42, stratify=query_labels)
            recall, precision = metrics_final.interpolated_pr_curve_optimized(pr_query_hashes, training_hashes, pr_query_labels, training_labels, num_points=100) # add optimized 
        if i == 1:

            recall, precision = metrics_final.interpolated_pr_curve_optimized(query_hashes, training_hashes, query_labels, training_labels, num_points=100) # add optimized 
            print("query time")
        


        print("query time")
        
        if i == 2:
            model = model.to("cpu")
            model.device = torch.device("cpu")
            # ide: split db op i 2 -> join dem senere, var også en
            _,training_hashes = model.forward(training, use_one_hot=True)
            training_hashes = training_hashes.detach().numpy()
            _, query_hashes = model.forward(query, use_one_hot=True)
            _=0
            query_hashes = query_hashes.detach().numpy()
            print(len(training_hashes))
            np.save(fr"C:\Users\Test\Desktop\Local_folder\Saved_hashes\database_{model_name}_{bit}.npy",training_hashes)
            np.save(fr"C:\Users\Test\Desktop\Local_folder\Saved_hashes\database_labels_{model_name}_{bit}.npy",training_labels)
            np.save(fr"C:\Users\Test\Desktop\Local_folder\Saved_hashes\query_{model_name}_{bit}.npy",query_hashes)
            np.save(fr"C:\Users\Test\Desktop\Local_folder\Saved_hashes\query_labels_{model_name}_{bit}.npy",query_labels)
            map_score = np.nan
            p_at_k = tuple([np.nan for _ in range(len(ks))])
            p_at_dist = tuple([np.nan for _ in range(len(dists))])
            recall, precision = np.nan, np.nan
            database_hashes = 0

            database, database_labels, query, query_labels = load_data(i)



            x = time.time()
            database = torch.tensor(database, dtype=torch.float32)  # Adjust dtype if needed
        
            batch_size = 128 
            db_split1_loader = DataLoader(database, batch_size=batch_size, shuffle=False)
            db_h1_all = []
            database = 0
            with torch.no_grad():  # No gradients needed for inference
                device_loc = torch.device("cpu")
                for batch in db_split1_loader:
                    batch = batch.to(device_loc)  # Move batch to the appropriate device
                    _, db_h1 = model.forward(batch, use_one_hot=True)  # Forward pass
                    db_h1_all.append(db_h1)

            db_h1_all= torch.cat(db_h1_all, dim=0)  # Combine results from all batches
            database_hashes= db_h1_all.numpy()
            print(x - time.time())


            query_batch = []
            query_loader = DataLoader(query, batch_size=batch_size, shuffle=False)
            with torch.no_grad():  # No gradients needed for inference
                device_loc = torch.device("cpu")
                for batch in query_loader:
                    batch = batch.to(device_loc)  # Move batch to the appropriate device
                    _, q = model.forward(batch, use_one_hot=True)  # Forward pass
                    query_batch.append(q)

            query_hashes = torch.cat(query_batch, dim=0)  # Combine results from all batches
            query_hashes  = query_hashes.numpy()



            
            query_image = query[22]
            query_time_start = time.time()
            # run single image thorugh model

            t, query_hash = model.forward(query_image, use_one_hot=True)
            #t = 0s
                
            query_hash = query_hash.cpu().detach().numpy()
            images = metrics_final.query_optimized(query_hash, database_hashes)
            query_time = time.time() - query_time_start
            database = 0

        else:
            database, database_labels, query, query_labels = load_data(i)
            t, database = model.forward(database, use_one_hot=True)
            t=0
            database = database.cpu().detach().numpy()
            query_image = query[22]
            query_time_start = time.time()
            # run single image thorugh model
            t, query_hash = model.forward(query_image, use_one_hot=True)
            query_hash = query_hash.cpu().detach().numpy()
            images = metrics_final.query_optimized(query_hash, database)
            query_time = time.time() - query_time_start
            database = 0






        if i ==2 and bit==48:
            images = images[:5]
            np.save(fr"C:\Users\Test\Desktop\Local_folder\Result_folder\query_{model_name}_5_images.npy", images)
         

        if i == 0 and bit == 48:
            np.save(fr"C:\Users\Test\Desktop\Local_folder\Result_folder\tsne_{model_name}_hashes.npy",query_hashes)
            #results_df.loc[results_df.shape[0]] = (names[i], bit, training_time, query_time, map_score) + p_at_k + p_at_dist
            #results_df.to_csv(f"{model_name}_{names[i]}_testing.csv", index=False) # addet csv


        if i ==0:

            #pr_df = pd.DataFrame(data=np.array([precision, recall]).T, columns=["precision", "recall"])
            #pr_df.to_csv(f"{model_name}_{names[i]}_{bit}_pr_curve.csv") # addet csv 
            print(datetime.datetime.now())
        with torch.no_grad():
            torch.cuda.empty_cache()

        