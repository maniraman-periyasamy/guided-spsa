from pathlib import Path
from typing import Any
import numpy as np
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, normalize
import os
import pickle
import random
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticData(Dataset[Any]):
    idx: int  # requested data index
    x: torch.Tensor
    y: torch.Tensor

    def __init__(self, data: np.ndarray, targets: np.ndarray):
        if len(data) != len(targets):
            raise ValueError(
                "data and targets must be the same length. "
                f"{len(data)} != {len(targets)}"
            )

        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.get_x(idx)
        y = self.get_y(idx)
        return x, y

    def get_x(self, idx: int):
        self.idx = idx
        self.preprocess_x()
        return self.x

    def preprocess_x(self):
        self.x = self.data[self.idx].copy().astype(np.float32)
        self.x = torch.tensor(self.x)

    def get_y(self, idx: int):
        self.idx = idx
        self.preprocess_y()
        return self.y

    def preprocess_y(self):
        self.y = self.targets[self.idx].copy().astype(np.float32)
        self.y = torch.tensor(self.y)


def process_dataset(x,y , validation_size, test_size):

    
    for j in range(x.shape[1]):
        x[:, j] = ((x[:, j]-np.min(x[:, j]))/(np.max(x[:, j])-np.min(x[:, j])))*np.pi
    
    y = y.reshape((y.size, 1))
    y = ((y-np.min(y))/(np.max(y)-np.min(y)))*2 - 1
   
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_size, random_state=23)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size, random_state=6)

    return x_train, y_train, x_val, y_val, x_test, y_test

def process_dataset_classification(x,y , validation_size, test_size):

    
    for j in range(x.shape[1]):
        x[:, j] = ((x[:, j]-np.min(x[:, j]))/(np.max(x[:, j])-np.min(x[:, j])))*np.pi
    

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_size, random_state=23)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size, random_state=6)

    return x_train, y_train, x_val, y_val, x_test, y_test



def generate_dataloader( task: str, name: str, dataset_size: int, n_features: int, noise: float, 
    test_size: float = 0.1, validation_size: float = 0.2, batch_size: int = 1, shuffle: bool = True) -> DataLoader[Any]:
    
    if task == "regression":
        if name == "friedman1":
            if n_features <5:
                raise ValueError
            if os.path.isfile("dataset/regression/friedman1/"+str(dataset_size)+".pkl"):
                with open("dataset/regression/friedman1/"+str(dataset_size)+".pkl", "rb") as f:
                        x_train, y_train, x_val, y_val, x_test, y_test = pickle.load(f)
            else:
                if not os.path.exists("dataset/regression/friedman1/"):
                    os.makedirs("dataset/regression/friedman1/")
                    
                x,y = dt.make_friedman1(n_samples=1000, n_features=n_features, noise=noise)
                x_train, y_train, x_val, y_val, x_test, y_test = process_dataset(x, y, validation_size, test_size)

                with open("dataset/regression/friedman1/"+str(1000)+".pkl", "wb") as f:
                    pickle.dump((x_train,y_train, x_val, y_val, x_test, y_test),f)

                train_index = random.sample(range(len(x_train)), 500)
                val_index = random.sample(range(len(x_val)), 100)
                with open("dataset/regression/friedman1/"+str(500)+".pkl", "wb") as f:
                    pickle.dump((x_train[train_index, :], y_train[train_index], x_val[val_index, :], y_val[val_index],
                                x_test, y_test),f)

                train_index = random.sample(range(len(x_train)), 100)
                val_index = random.sample(range(len(x_val)), 20)
                with open("dataset/regression/friedman1/"+str(100)+".pkl", "wb") as f:
                    pickle.dump((x_train[train_index, :], y_train[train_index], x_val[val_index, :], y_val[val_index],
                                x_test, y_test),f)

                train_index = random.sample(range(len(x_train)), 50)
                val_index = random.sample(range(len(x_val)), 10)
                with open("dataset/regression/friedman1/"+str(50)+".pkl", "wb") as f:
                    pickle.dump((x_train[train_index, :], y_train[train_index], x_val[val_index, :], y_val[val_index],
                                x_test, y_test),f)

        elif name == "boston-housing":
            if os.path.isfile("dataset/regression/boston-housing/"+str(dataset_size)+".pkl"):
                with open("dataset/regression/boston-housing/"+str(dataset_size)+".pkl", "rb") as f:
                        x_train, y_train, x_val, y_val, x_test, y_test = pickle.load(f)
            else:
                if not os.path.exists("dataset/regression/boston-housing/"):
                    os.makedirs("dataset/regression/boston-housing/")
                
                column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
                data = pd.read_csv('dataset/regression/boston-housing/housing.csv', header=None, delimiter=r"\s+", names=column_names)
                x_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
                y = data['NOX'].to_numpy()
                x = data[x_columns].to_numpy()

                x_train, y_train, x_val, y_val, x_test, y_test = process_dataset(x, y, validation_size, test_size)
                with open("dataset/regression/boston-housing/"+str(500)+".pkl", "wb") as f:
                    pickle.dump((x_train, y_train, x_val, y_val,x_test, y_test),f)

                train_index = random.sample(range(len(x_train)), 50)
                val_index = random.sample(range(len(x_val)), 10)
                with open("dataset/regression/boston-housing/"+str(50)+".pkl", "wb") as f:
                    pickle.dump((x_train[train_index, :], y_train[train_index], x_val[val_index, :], y_val[val_index],
                                x_test, y_test),f)

        elif name == "friedman2":

            if os.path.isfile("dataset/regression/friedman2/"+str(dataset_size)+".pkl"):
                with open("dataset/regression/friedman2/"+str(dataset_size)+".pkl", "rb") as f:
                        x_train, y_train, x_val, y_val, x_test, y_test = pickle.load(f)
            else:
                if not os.path.exists("dataset/regression/friedman2/"):
                    os.makedirs("dataset/regression/friedman2/")
                    
                x,y = dt.make_friedman2(n_samples=1000,noise=noise)
                x_train, y_train, x_val, y_val, x_test, y_test = process_dataset(x, y, validation_size, test_size)

                with open("dataset/regression/friedman2/"+str(1000)+".pkl", "wb") as f:
                    pickle.dump((x_train,y_train, x_val, y_val, x_test, y_test),f)

                train_index = random.sample(range(len(x_train)), 500)
                val_index = random.sample(range(len(x_val)), 100)
                with open("dataset/regression/friedman2/"+str(500)+".pkl", "wb") as f:
                    pickle.dump((x_train[train_index, :], y_train[train_index], x_val[val_index, :], y_val[val_index],
                                x_test, y_test),f)

                train_index = random.sample(range(len(x_train)), 100)
                val_index = random.sample(range(len(x_val)), 20)
                with open("dataset/regression/friedman2/"+str(100)+".pkl", "wb") as f:
                    pickle.dump((x_train[train_index, :], y_train[train_index], x_val[val_index, :], y_val[val_index],
                                x_test, y_test),f)

                train_index = random.sample(range(len(x_train)), 50)
                val_index = random.sample(range(len(x_val)), 10)
                with open("dataset/regression/friedman2/"+str(50)+".pkl", "wb") as f:
                    pickle.dump((x_train[train_index, :], y_train[train_index], x_val[val_index, :], y_val[val_index],
                                x_test, y_test),f)

        elif name == "friedman3":

            if os.path.isfile("dataset/regression/friedman3/"+str(dataset_size)+".pkl"):
                with open("dataset/regression/friedman3/"+str(dataset_size)+".pkl", "rb") as f:
                        x_train, y_train, x_val, y_val, x_test, y_test = pickle.load(f)
            else:
                if not os.path.exists("dataset/regression/friedman3/"):
                    os.makedirs("dataset/regression/friedman3/")
                    
                x,y = dt.make_friedman3(n_samples=1000,noise=noise)
                x_train, y_train, x_val, y_val, x_test, y_test = process_dataset(x, y, validation_size, test_size)

                with open("dataset/regression/friedman3/"+str(1000)+".pkl", "wb") as f:
                    pickle.dump((x_train,y_train, x_val, y_val, x_test, y_test),f)

                train_index = random.sample(range(len(x_train)), 500)
                val_index = random.sample(range(len(x_val)), 100)
                with open("dataset/regression/friedman3/"+str(500)+".pkl", "wb") as f:
                    pickle.dump((x_train[train_index, :], y_train[train_index], x_val[val_index, :], y_val[val_index],
                                x_test, y_test),f)

                train_index = random.sample(range(len(x_train)), 100)
                val_index = random.sample(range(len(x_val)), 20)
                with open("dataset/regression/friedman3/"+str(100)+".pkl", "wb") as f:
                    pickle.dump((x_train[train_index, :], y_train[train_index], x_val[val_index, :], y_val[val_index],
                                x_test, y_test),f)

                train_index = random.sample(range(len(x_train)), 50)
                val_index = random.sample(range(len(x_val)), 10)
                with open("dataset/regression/friedman3/"+str(50)+".pkl", "wb") as f:
                    pickle.dump((x_train[train_index, :], y_train[train_index], x_val[val_index, :], y_val[val_index],
                                x_test, y_test),f)

        elif name == "CCPP":

            if os.path.isfile("dataset/regression/CCPP/"+str(dataset_size)+".pkl"):
                with open("dataset/regression/CCPP/"+str(dataset_size)+".pkl", "rb") as f:
                        x_train, y_train, x_val, y_val, x_test, y_test = pickle.load(f)
            else:
                if not os.path.exists("dataset/regression/CCPP/"):
                    os.makedirs("dataset/regression/CCPP/")
                    
                df = pd.read_excel("dataset/source_files/CCPP/Folds5x2_pp.xlsx")
                x = df.to_numpy()
                y = x[:,4]
                x = x[:,:4]
                x_train, y_train, x_val, y_val, x_test, y_test = process_dataset(x, y, validation_size, test_size)

                with open("dataset/regression/CCPP/"+str(1000)+".pkl", "wb") as f:
                    pickle.dump((x_train,y_train, x_val, y_val, x_test, y_test),f)

                train_index = random.sample(range(len(x_train)), 500)
                val_index = random.sample(range(len(x_val)), 100)
                with open("dataset/regression/CCPP/"+str(500)+".pkl", "wb") as f:
                    pickle.dump((x_train[train_index, :], y_train[train_index], x_val[val_index, :], y_val[val_index],
                                x_test, y_test),f)

                train_index = random.sample(range(len(x_train)), 100)
                val_index = random.sample(range(len(x_val)), 20)
                with open("dataset/regression/CCPP/"+str(100)+".pkl", "wb") as f:
                    pickle.dump((x_train[train_index, :], y_train[train_index], x_val[val_index, :], y_val[val_index],
                                x_test, y_test),f)

                train_index = random.sample(range(len(x_train)), 50)
                val_index = random.sample(range(len(x_val)), 10)
                with open("dataset/regression/CCPP/"+str(50)+".pkl", "wb") as f:
                    pickle.dump((x_train[train_index, :], y_train[train_index], x_val[val_index, :], y_val[val_index],
                                x_test, y_test),f)
        
        else:

            if os.path.isfile("dataset/regression/general/"+str(dataset_size)+".pkl"):
                with open("dataset/regression/general/"+str(dataset_size)+".pkl", "rb") as f:
                        x_train, y_train, x_val, y_val, x_test, y_test = pickle.load(f)
            else:
                if not os.path.exists("dataset/regression/general/"):
                    os.makedirs("dataset/regression/general/")
                    
                x,y = dt.make_regression(n_samples=1000,n_features=n_features,noise=noise)
                x_train, y_train, x_val, y_val, x_test, y_test = process_dataset(x, y, validation_size, test_size)

                with open("dataset/regression/general/"+str(1000)+".pkl", "wb") as f:
                    pickle.dump((x_train,y_train, x_val, y_val, x_test, y_test),f)

                train_index = random.sample(range(len(x_train)), 500)
                val_index = random.sample(range(len(x_val)), 100)
                with open("dataset/regression/general/"+str(500)+".pkl", "wb") as f:
                    pickle.dump((x_train[train_index, :], y_train[train_index], x_val[val_index, :], y_val[val_index],
                                x_test, y_test),f)

                train_index = random.sample(range(len(x_train)), 100)
                val_index = random.sample(range(len(x_val)), 20)
                with open("dataset/regression/general/"+str(100)+".pkl", "wb") as f:
                    pickle.dump((x_train[train_index, :], y_train[train_index], x_val[val_index, :], y_val[val_index],
                                x_test, y_test),f)

                train_index = random.sample(range(len(x_train)), 50)
                val_index = random.sample(range(len(x_val)), 10)
                with open("dataset/regression/general/"+str(50)+".pkl", "wb") as f:
                    pickle.dump((x_train[train_index, :], y_train[train_index], x_val[val_index, :], y_val[val_index],
                                x_test, y_test),f)

    elif task == "classification":
        if name == "unique":
            if os.path.isfile("dataset/classification/unique/"+str(dataset_size)+".pkl"):
                with open("dataset/classification/unique/"+str(dataset_size)+".pkl", "rb") as f:
                        x, y = pickle.load(f)
            else:
                if not os.path.exists("dataset/classification/unique/"):
                    os.makedirs("dataset/classification/unique/")
                    
                #x,y = dt.make_friedman1(n_samples=dataset_size, n_features=n_features, noise=noise)
                x,y = dt.make_classification(n_samples=dataset_size, n_features=n_features, n_classes=n_features, n_clusters_per_class=1)
                x = ((x-np.min(x))/(np.max(x)-np.min(x)))*np.pi/2
                onehot_encoder = OneHotEncoder(sparse=False)
                y = y.reshape(len(y), 1)
                y = onehot_encoder.fit_transform(y)
                
                with open("dataset/classification/unique/"+str(dataset_size)+".pkl", "wb") as f:
                    pickle.dump((x,y),f)
        
        elif name == "iris":
            dataset_size = 150
            if os.path.isfile("dataset/classification/iris/"+str(dataset_size)+".pkl"):
                with open("dataset/classification/iris/"+str(dataset_size)+".pkl", "rb") as f:
                        x_train, y_train, x_val, y_val, x_test, y_test = pickle.load(f)
            else:
                if not os.path.exists("dataset/classification/iris/"):
                    os.makedirs("dataset/classification/iris/")
                    
                #x,y = dt.make_friedman1(n_samples=dataset_size, n_features=n_features, noise=noise)
                x,y = dt.load_iris(return_X_y=True)
                x_train, y_train, x_val, y_val, x_test, y_test = process_dataset_classification(x, y, validation_size, test_size)
                num_classes = np.unique(y_train)
                #y = y.reshape(len(y), 1)
                y_train = np.eye(len(num_classes))[y_train]
                y_val = np.eye(len(num_classes))[y_val]
                y_test = np.eye(len(num_classes))[y_test]
                
                with open("dataset/classification/iris/"+str(dataset_size)+".pkl", "wb") as f:
                    pickle.dump((x_train, y_train, x_val, y_val, x_test, y_test),f)

        
    
    

    train_loader = DataLoader(
        dataset=SyntheticData(x_train, y_train),
        batch_size=batch_size,
        shuffle=shuffle
    )

    validation_loader = DataLoader(
        dataset=SyntheticData(x_val, y_val),
        batch_size=batch_size,
        shuffle=shuffle
    )

    test_loader = DataLoader(
        dataset=SyntheticData(x_test[:100], y_test[:100]),
        batch_size=batch_size,
        shuffle=shuffle
    )

    return train_loader, test_loader, validation_loader