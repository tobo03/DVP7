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



triplet_hpo_dic ={
    12: {"Cifar":   {"alpha": 3, "lr":0.0001, "wd": 0.000010, "bits" : 12}, 
         "Nus_Wide":{"alpha": 3, "lr":0.0001, "wd": 0.00001, "bits" : 12}, 
         "Imagenet":{"alpha": 5, "lr":0.00010, "wd": 0.00001, "bits" : 12}},

    24: {"Cifar":   {"alpha": 3, "lr":0.0001, "wd": 0.000100, "bits" : 24}, 
         "Nus_Wide":{"alpha": 5, "lr":0.0001, "wd": 0.00001, "bits" : 24}, 
         "Imagenet":{"alpha": 5, "lr":0.00010, "wd": 0.00001, "bits" : 24}},

    32: {"Cifar":   {"alpha": 3, "lr":0.0001, "wd": 0.000001, "bits" : 32}, 
         "Nus_Wide":{"alpha": 5, "lr":0.0001, "wd": 0.00001, "bits" : 32}, 
         "Imagenet":{"alpha": 5, "lr":0.00010, "wd": 0.00001, "bits" : 32}},

    48: {"Cifar":   {"alpha": 5, "lr":0.0001, "wd": 0.000100, "bits" : 48}, 
         "Nus_Wide":{"alpha": 3, "lr":0.0001, "wd": 0.00010, "bits" : 48}, 
         "Imagenet":{"alpha": 1, "lr":0.00001, "wd": 0.00010, "bits" : 48}},
        }


def get_dataloader(device, X_train, y_train, one_hot=True, batchSize = 100):
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)

    if len(y_train.shape)==1:
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

def train_dtsh(device, X_train, y_train, ALPHA, lr, weight_decay, bits, do_one_hot=True, earlyStop_num=20, time_stop=60*60*2):
    data = {}
    start_time = time.time()



    model = nn.Sequential(  nn.Linear(4096,256),
                            nn.ReLU(),
                            nn.Linear(256, bits),
                            nn.Sigmoid()
                            )
    model.to(device)

    criterion = TripletLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr , weight_decay=weight_decay)

    dataloader = get_dataloader(device, X_train, y_train, one_hot=do_one_hot)
    historical_lostList = []
    for i in range(300):
        loss_list = []
        for j,batch  in enumerate(dataloader):
            X_batch = batch[0].to(device)
            y_batch = batch[1].to(device)

            optimizer.zero_grad()

            u = model(X_batch)
            loss = criterion(u, y_batch.float(), ALPHA=ALPHA)
            loss.backward()
            optimizer.step()

            loss_list.append( float(loss) )
        
        
        mean_loss = sum(loss_list) / len(loss_list)
        print(f"Epoch: {i+1}/{300}, loss: {mean_loss}")
        historical_lostList.append(mean_loss)

        if earlyStop(historical_lostList, n = earlyStop_num): 
            print(i, mean_loss)
            print("Early Stop!!!")
            data["earlyStop"] = True
            break
        
        if time.time() - start_time > time_stop:
            print("time_stop")
            break
    
    
    return model

class TripletLoss(torch.nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, u, y, ALPHA=1):
        #LAMBDA = 1
        #ALPHA  = 1

        inner_product = torch.cdist(u, u, p=2) 
        s = y @ y.t() > 0           # A matrix that show if the two idexes are the same or not

        loss1 = torch.tensor(0.0, requires_grad=True) + 0
        for row in range(s.shape[0]):
            # if has positive pairs and negative pairs
            if s[row].sum() != 0 and (~s[row]).sum() != 0:
                
                theta_negative = inner_product[row][s[row] == 0]
                
                theta_positive = inner_product[row][s[row] == 1]
                theta_positive = theta_positive[theta_positive != 0] # remove the anchor

                for p in theta_positive: 
                    n_i = torch.logical_and( (p < theta_negative), (theta_negative < p + ALPHA) )
                    
                    if sum(n_i) != 0:
                        n = torch.min( theta_negative[n_i] )

                        loss1 += (p - n + ALPHA).clamp(min=0)
        

        return loss1

results_df = pd.DataFrame(columns=["dataset", "bits", "training_time", "query_time", "map", "p@k_1", "p@k_2", "p@k_3", "p@0", "p@1", "p@2", "p@3"])

model_name = "triplet"

bits = [12, 24,32, 48] #  udkommenteret.

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
            #training = training[:40000]
            #database = database[:55000]
            print(f"Loaded {filename} into X_train")
        elif "y_train" in filename:
            training_labels = np.load(os.path.join(data_dir, filename))
            #training_labels = training_labels[:40000]
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


for i in range(3): # QUick fix to run it just once (Image)

    dists = [0,1,2,3]

    for bit in bits:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cpu")
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
        model = train_dtsh(device=device, X_train=training, y_train=training_labels, ALPHA=triplet_hpo_dic[bit][names[i]]["alpha"], lr=triplet_hpo_dic[bit][names[i]]["lr"], weight_decay=triplet_hpo_dic[bit][names[i]]["wd"], bits=bit, do_one_hot=True, earlyStop_num=20)
        training_time = time.time() - training_time_start 

        #run database through model
        print("model_training done")


        if i != 2:
            training_hashes = model(torch.tensor(training, dtype=torch.float32).to(device))
            training_hashes = training_hashes.cpu().detach().numpy()
            #run query set through model
            query_hashes = model(torch.tensor(query, dtype=torch.float32).to(device))
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
            #model = model.to("cpu")
            #model.device = torch.device("cpu")
            # ide: split db op i 2 -> join dem senere, var også en
            training_hashes = model(torch.tensor(training, dtype=torch.float32).to(device))
            training_hashes = training_hashes.cpu().detach().numpy()
            query_hashes = model(torch.tensor(query, dtype=torch.float32).to(device))
            
            query_hashes = query_hashes.cpu().detach().numpy()
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
                    db_h1 = model(torch.tensor(batch, dtype=torch.float32).to(device))  # Forward pass
                    db_h1_all.append(db_h1)

            db_h1_all= torch.cat(db_h1_all, dim=0)  # Combine results from all batches
            database_hashes= db_h1_all.cpu().numpy()
            print(x - time.time())


            query_batch = []
            query_loader = DataLoader(query, batch_size=batch_size, shuffle=False)
            with torch.no_grad():  # No gradients needed for inference
                device_loc = torch.device("cpu")
                for batch in query_loader:
                    batch = batch.to(device_loc)  # Move batch to the appropriate device
                    q = model(torch.tensor(batch, dtype=torch.float32).to(device))  # Forward pass
                    query_batch.append(q)

            query_hashes = torch.cat(query_batch, dim=0)  # Combine results from all batches
            query_hashes  = query_hashes.cpu().numpy()



            
            query_image = query[22]
            query_time_start = time.time()
            # run single image thorugh model

            query_hash = model(torch.tensor(query_image, dtype=torch.float32).to(device))
            #t = 0s
                
            query_hash = query_hash.cpu().detach().numpy()
            images = metrics_final.query_optimized(query_hash, database_hashes)
            query_time = time.time() - query_time_start
            database = 0

        else:
            database, database_labels, query, query_labels = load_data(i)
            database = model(torch.tensor(database, dtype=torch.float32).to(device))
            t=0
            database = database.cpu().detach().numpy()
            query_image = query[22]
            query_time_start = time.time()
            # run single image thorugh model
            query_hash = model(torch.tensor(query_image, dtype=torch.float32).to(device))
            query_hash = query_hash.cpu().detach().numpy()
            images = metrics_final.query_optimized(query_hash, database)
            query_time = time.time() - query_time_start
            database = 0






        if i ==2 and bit==48:
            images = images[:5]
            np.save(fr"C:\Users\Test\Desktop\Local_folder\Result_folder\query_{model_name}_5_images.npy", images)
         

        if i == 0 and bit == 48:
            np.save(fr"C:\Users\Test\Desktop\Local_folder\Result_folder\tsne_{model_name}_hashes.npy",query_hashes)


        results_df.loc[results_df.shape[0]] = (names[i], bit, training_time, query_time, map_score) + p_at_k + p_at_dist
        results_df.to_csv(f"{model_name}_{names[i]}_testing.csv", index=False) # addet csv


        if i !=2:

            pr_df = pd.DataFrame(data=np.array([precision, recall]).T, columns=["precision", "recall"])
            pr_df.to_csv(f"{model_name}_{names[i]}_{bit}_pr_curve.csv") # addet csv 
            print(datetime.datetime.now())
        with torch.no_grad():
            torch.cuda.empty_cache()

        
        
