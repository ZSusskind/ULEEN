#!/usr/bin/false

################################################################################
# get_datasets.py
# Author: Zachary Susskind (ZSusskind@utexas.edu)
#
# Retrieves thermometer encoding thresholds (and performs thermometer encoding)
# for a specified dataset.
#
#
# MIT License
# 
# Copyright (c) 2024 The University of Texas at Austin
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################

import torch
from torch.utils.data import TensorDataset
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import pickle
from lz4 import frame
import lzma
import openml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import norm

dataset_ids = {
    "gesture_phase": 4538,
    "eye_movements": 1044,
    "covertype": 1596
}

def get_thresholds(T_X_train, encoding_bits, encoding="gaussian"):
    if encoding == "mean": # Compare against mean value
        train_mean = T_X_train.mean(axis=0)
        thresholds = train_mean.unsqueeze(-1)\
            .repeat_interleave(encoding_bits, -1)
    elif encoding == "linear":
        train_min = T_X_train.amin(axis=0)
        train_max = T_X_train.amax(axis=0)
        train_spread = train_max - train_min
        fractions = [(i+1)/(encoding_bits+1)\
                     for i in range(encoding_bits)]
        thresholds = torch.stack(
            tuple(train_min + train_spread*f for f in fractions),
            axis=-1)
    elif encoding == "gaussian":
        train_mean = T_X_train.mean(axis=0)
        train_std = T_X_train.std(axis=0)
        std_skews = torch.Tensor([norm.ppf((i+1)/(encoding_bits+1))\
                                  for i in range(encoding_bits)])
        thresholds = ((std_skews.view(-1, 1).repeat(1, T_X_train.shape[1])\
                       * train_std) + train_mean).T
    elif encoding == "percentile":
        quantiles = [(i+1)/(encoding_bits+1) for i in range(encoding_bits)]
        thresholds = torch.stack(
            tuple(torch.quantile(T_X_train, q, axis=0) for q in quantiles),
            axis=-1)
    else:
        raise ValueError(f"Unknown encoding \"{encoding}\"")
    return thresholds

def get_dataset(dset_name, encoding_bits, encoding="gaussian",
                total_inputs=None):
    dset_name = dset_name.lower()
    assert((encoding_bits is None) or (total_inputs is None))
    if dset_name in dataset_ids:
        dataset = openml.datasets.get_dataset(dataset_ids[dset_name])
        X, Y, _, _ = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute)
        features = X.astype("float").values
        labels = Y.astype("category").cat.codes.values
        X_train, X_test, Y_train, Y_test = train_test_split(features,
                                                            labels,
                                                            train_size=0.8,
                                                            random_state=0)
    elif dset_name in ["mnist", "mnist_aug", "fashionmnist", "fashionmnist_aug"]:
        if dset_name.split("_")[0] == "mnist":
            base_dset = dsets.MNIST
        else:
            base_dset = dsets.FashionMNIST
        train_dataset = base_dset(root='./../data',
                train=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: torch.flatten(x))]),
                download=True)
        test_dataset = base_dset(root='./../data',
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: torch.flatten(x))]))
        X_train = train_dataset.data.reshape(len(train_dataset), -1).numpy()
        Y_train = train_dataset.targets.numpy()
        X_test = test_dataset.data.reshape(len(test_dataset), -1).numpy()
        Y_test = test_dataset.targets.numpy()
        if dset_name in ["mnist_aug", "fashionmnist_aug"]:
            padded_data = np.pad(X_train.reshape(-1, 28, 28), ((0, 0), (1, 1), (1, 1)))
            augmented_data = []
            for y in range(-1, 2):
                for x in range(-1, 2):
                    augmented_data.append(padded_data[:,y+1:y+29,x+1:x+29])
            augmented_X_train = np.concatenate(augmented_data, axis=0).reshape(9*X_train.shape[0], -1)
            augmented_Y_train = np.tile(Y_train, 9)
            data_order = np.arange(augmented_X_train.shape[0])
            np.random.seed(0)
            np.random.shuffle(data_order)
            X_train = augmented_X_train[data_order]
            Y_train = augmented_Y_train[data_order]
    elif dset_name == "cifar10":
        train_dataset = dsets.CIFAR10(root='./../data',
                train=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: torch.flatten(x))]),
                download=True)
        test_dataset = dsets.CIFAR10(root='./../data',
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: torch.flatten(x))]))
        X_train = train_dataset.data.reshape(len(train_dataset), -1)
        Y_train = np.array(train_dataset.targets)
        X_test = test_dataset.data.reshape(len(test_dataset), -1)
        Y_test = np.array(test_dataset.targets)
    elif dset_name in ["toy_admos", "kws"]:
        with open(f"./datasets/{dset_name}/train.pickle", "rb") as f:
            train_dataset = pickle.load(f)
        with open(f"./datasets/{dset_name}/test.pickle", "rb") as f:
            test_dataset = pickle.load(f)
        X_train = train_dataset[0].view(train_dataset[0].shape[0], -1).numpy()
        Y_train = train_dataset[1].numpy()
        X_test = test_dataset[0].view(test_dataset[0].shape[0], -1).numpy()
        Y_test = test_dataset[1].numpy()
    elif dset_name == "vww":
        with frame.open(f"./datasets/{dset_name}/train.pickle.lz4", "rb") as f:
            train_dataset = pickle.load(f)
        with frame.open(f"./datasets/{dset_name}/test.pickle.lz4", "rb") as f:
            test_dataset = pickle.load(f)
        X_train = train_dataset[0].view(train_dataset[0].shape[0], -1).numpy()
        Y_train = train_dataset[1].numpy()
        X_test = test_dataset[0].view(test_dataset[0].shape[0], -1).numpy()
        Y_test = test_dataset[1].numpy()
    elif dset_name.startswith("bot_iot"):
        dname = "./datasets/bot_iot"
        fext = dset_name[7:]
        with frame.open(f"{dname}/train{fext}.pickle.lz4", "rb") as f:
            X_train, Y_train = pickle.load(f)
        with frame.open(f"{dname}/test{fext}.pickle.lz4", "rb") as f:
            X_test, Y_test = pickle.load(f)
    elif dset_name.startswith("unswnb15"):
        dname = "./datasets/unswnb15"
        fext = dset_name[8:]
        with frame.open(f"{dname}/train{fext}.pickle.lz4", "rb") as f:
            X_train, Y_train = pickle.load(f)
        with frame.open(f"{dname}/test{fext}.pickle.lz4", "rb") as f:
            X_test, Y_test = pickle.load(f)
    else:
        raise ValueError(f"Unknown dataset {dset_name}")

    if encoding_bits is None and encoding is not None:
        assert((total_inputs / X_train.shape[1]).is_integer())
        encoding_bits = total_inputs // X_train.shape[1]
        print("Encoding bits:", encoding_bits)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train,
                                                      Y_train,
                                                      train_size=0.9,
                                                      random_state=42)
    
    T_X_train = torch.Tensor(X_train)
    T_X_val = torch.Tensor(X_val)
    T_X_test = torch.Tensor(X_test)
    if encoding is not None:
        print(f"Using {encoding} encoding with {encoding_bits} encoding bits")
        thresholds = get_thresholds(T_X_train, encoding_bits, encoding)
            
        # Binarize with thermometer encoding
        B_X_train = (T_X_train.unsqueeze(2) >= thresholds).byte()\
            .reshape(T_X_train.shape[0], -1)
        B_X_val = (T_X_val.unsqueeze(2) >= thresholds).byte()\
            .reshape(T_X_val.shape[0], -1)
        B_X_test = (T_X_test.unsqueeze(2) >= thresholds).byte()\
            .reshape(T_X_test.shape[0], -1)
        
        train_set = TensorDataset(B_X_train, torch.LongTensor(Y_train))
        val_set = TensorDataset(B_X_val, torch.LongTensor(Y_val))
        test_set = TensorDataset(B_X_test, torch.LongTensor(Y_test))
    else:
        print("Skipping thermometer encoding")
        train_set = TensorDataset(T_X_train, torch.LongTensor(Y_train))
        val_set = TensorDataset(T_X_val, torch.LongTensor(Y_val))
        test_set = TensorDataset(T_X_test, torch.LongTensor(Y_test))

    return train_set, val_set, test_set
