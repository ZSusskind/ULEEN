#!/usr/bin/env python3

################################################################################
# train_model.py
# Author: Zachary Susskind (ZSusskind@utexas.edu)
#
# Handles training for ULEEN models
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
import torch.nn as nn

import sys
import json
import signal
import pickle
import lzma
import numpy as np
import pandas as pd

from time import perf_counter

from model import BackpropWiSARD, BackpropMultiWiSARD
from get_datasets import get_dataset

default_device = "cuda" if torch.cuda.is_available() else "cpu"

np.random.seed(0)
torch.manual_seed(0)

# Run inference using dataset (validation or test set)
def run_inference(model, dset_loader, collect_submodel_accuracies=False):
    total = 0
    correct = 0
    device = next(model.parameters()).device
    if collect_submodel_accuracies:
        submodel_correct = torch.zeros(len(model.models), device=device)
    for features, labels in dset_loader:
        features, labels = features.to(device), labels.to(device)
        model_results = model(features)
        if model_results.ndim == 3:
            outputs = model_results.sum(axis=0)
        else:
            outputs = model_results

        _, predicted = torch.max(outputs.data, axis=1)
        if collect_submodel_accuracies:
            submodel_predicted = torch.argmax(model_results.data,
                                              axis=2).detach()
        total += labels.size(0)
        correct += (predicted == labels).sum()
        if collect_submodel_accuracies:
            submodel_correct += (submodel_predicted == labels).sum(axis=1)

    if collect_submodel_accuracies:
        return total, correct, submodel_correct
    else:
        return total, correct

Abort_Training = False
def sigint_handler(signum, frame):
    global Abort_Training
    if not Abort_Training:
        print("Will abort training at end of epoch")
        Abort_Training = True
    else:
        sys.exit("Quitting immediately on second SIGINT")

# Train pre-specified model
def train_model(
    model, train_loader, val_loader, num_epochs=100,
    learning_rate=1e-3, decay_lr=False, device=None
):
    if device is None:
        device = default_device

    global Abort_Training
    Abort_Training = False
    old_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, sigint_handler)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    if decay_lr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=(num_epochs*3)//10)

    for epoch in range(num_epochs):
        start_time = perf_counter()

        train_total = 0
        train_correct = 0
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            #outputs = model(features)
            model_results = model(features)
            outputs = model_results.sum(axis=0)
          
            _, predicted = torch.max(outputs.data, 1)

            train_total += labels.size(0)
            train_correct += (predicted == labels).sum()

            #loss = criterion(outputs, labels)
            loss = sum(criterion(o, labels) for o in model_results)
            loss.backward()
            optimizer.step()
            model.clamp()

        model.eval()
        val_total, val_correct, val_submodel_corect =\
            run_inference(model, val_loader, True)
        end_time = perf_counter()

        print(f"At end of epoch {epoch}: "\
                f"Train set: Correct: {train_correct}/{train_total} "\
                f"({round(((100*train_correct)/train_total).item(), 3)}%); "\
                f"Validation set: Correct: {val_correct}/{val_total} "\
                f"({round(((100*val_correct)/val_total).item(), 3)}%)")
        submodel_accuracies = [f"{round(100*(c/val_total).item(), 3)}%"\
                               for c in val_submodel_corect]
        print(f"  Submodel accuracies: {' '.join(submodel_accuracies)}; Time elapsed: {round(end_time-start_time, 2)}")
        
        if decay_lr:
            scheduler.step()
        
        if Abort_Training:
            break
    
    model.eval()
    signal.signal(signal.SIGINT, old_handler)
    
    return model

def compute_model_size(inputs, classes, configs):
    total_model_size = 0
    for filter_inputs, filter_entries, filter_hash_functions in configs:
        filter_entries = min(filter_entries, 2**filter_inputs)
        filters_per_discriminator = int(np.ceil(inputs / filter_inputs))
        total_filters = filters_per_discriminator * classes
        total_model_size += total_filters * filter_entries
    model_size_kib = round(total_model_size / (2**13), 2)
    model_size_kp = round(total_model_size / (10**3), 2)
    print(f"Total model size: {model_size_kib} KiB / {model_size_kp} kParams")
    return total_model_size

def create_model(ds_name, configs, encoding_bits, model_fname, num_epochs=100,
        learning_rate=1e-3, decay_lr=False, encoding="gaussian", dropout_p=0.0,
        num_workers=4, batch_size=None):

    print(f"dropout_p: {dropout_p}")

    train_set, val_set, test_set = get_dataset(ds_name, encoding_bits,\
                                               encoding=encoding)
    inputs = train_set.tensors[0].shape[1]
    classes = (train_set.tensors[1].amax() + 1).item()

    print(f"Num inputs/classes: {inputs}({inputs//encoding_bits})/{classes}")
    print("Dump configs:", end="\n  ")
    print("\n  ".join(str(c) for c in configs))
    print(f"Batch size: {batch_size}")
    size = compute_model_size(inputs, classes, configs)
    
    train_data = train_set.tensors[0]
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True)
 
    model = BackpropMultiWiSARD(inputs, classes, configs,\
                                encoding_bits, dropout_p)
    model = train_model(model, train_loader, val_loader,
                        num_epochs, learning_rate, decay_lr)
    test_total, test_correct = run_inference(model, test_loader)
    print(f"Test set: Correct: {test_correct}/{test_total} "\
          f"({round(((100*test_correct)/test_total).item(), 3)}%)")
    test_total, test_correct = run_inference(model, test_loader)

    print(f"Test set: Correct: {test_correct}/{test_total} "\
          f"({round(((100*test_correct)/test_total).item(), 3)}%)")
    model = model.to("cpu")
    torch.save(model, model_fname)

    return model, size, (test_correct/test_total).item()

if __name__ == "__main__":
    settings_f = sys.argv[1]
    with open(settings_f, "r") as f:
        settings = json.load(f)

    dataset = settings["dataset"]
    configs = settings["configs"]
    encoding_bits = settings["encoding_bits"]
    model_fname = settings.get("output", "model.pt")
    num_epochs = settings.get("epochs", 100)
    learning_rate = settings.get("lr", 1e-3)
    decay_lr = settings.get("decay_lr", False)
    encoding = settings.get("encoding", "gaussian")
    dropout_p = settings.get("dropout_p", 0.0)
    batch_size = int(settings.get("batch_size", 32))

    model, size, acc = create_model(dataset, configs, encoding_bits,
                                    model_fname, num_epochs, learning_rate,
                                    decay_lr, encoding, dropout_p,
                                    batch_size=batch_size)
