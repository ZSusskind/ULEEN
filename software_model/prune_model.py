#!/usr/bin/env python3

################################################################################
# prune_model.py
# Author: Zachary Susskind (ZSusskind@utexas.edu)
#
# Implements uniform or non-uniform pruning for ULEEN models
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
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import os
import sys
import numpy as np

from model import BackpropWiSARD, BackpropMultiWiSARD
from get_datasets import get_dataset
from train_model import train_model, run_inference

default_device = "cuda" if torch.cuda.is_available() else "cpu"

class BiasModel(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(classes))

    def forward(self, x):
        return x + self.bias

def run_bias_inference(model, dset_loader):
    total = 0
    correct = 0
    device = next(model.parameters()).device
    for features, labels in dset_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    return total, correct

# Train pre-specified model
def train_bias_model(model, train_loader, val_loader, num_epochs=2, learning_rate=1e-3, device=None):
    if device is None:
        device = default_device

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

    for epoch in range(num_epochs):
        train_total = 0
        train_correct = 0
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            features = Variable(features)
            labels = Variable(labels)

            optimizer.zero_grad()
            outputs = model(features)
          
            _, predicted = torch.max(outputs.data, 1)

            train_total += labels.size(0)
            train_correct += (predicted == labels).sum()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_total, val_correct = run_bias_inference(model, val_loader)
        print(f"At end of epoch {epoch}: "\
                f"Train set: Correct: {train_correct}/{train_total} ({round(((100*train_correct)/train_total).item(), 3)}%); "\
                f"Validation set: Correct: {val_correct}/{val_total} ({round(((100*val_correct)/val_total).item(), 3)}%)")
    
    model.eval()
    model = model.to("cpu")
    return model

def process_dset(model, dset_loader, device=None, ensemble=False):
    if device is None:
        device = default_device

    results = None
    labels = torch.empty(len(dset_loader.dataset), dtype=torch.long)
    model = model.to(device)
    model.eval()
    idx = 0
    for features, l in dset_loader:
        features = features.to(device)
        if ensemble:
            model_results = model(features)
            outputs = model_results.sum(axis=0)
        else:
            outputs = model(features)
        if idx == 0:
            results = torch.empty(len(dset_loader.dataset), *outputs.shape[1:])
        results[idx:idx+features.shape[0]] = outputs.detach().cpu()
        labels[idx:idx+features.shape[0]] = l.cpu()
        idx += features.shape[0]

    return torch.utils.data.TensorDataset(results, labels)

def compute_new_model_size(model, ratio):
    total_model_size = 0
    for m in model.models:
        eff_filters_per_discriminator = m.filters_per_discriminator - int(ratio * m.filters_per_discriminator)
        total_filters = eff_filters_per_discriminator * m.classes
        total_model_size += total_filters * m.filter_entries
    model_size_kib = round(total_model_size / (2**13), 2)
    model_size_kp = round(total_model_size / (10**3), 2)
    print(f"Target model size: {model_size_kib} KiB / {model_size_kp} kParams")

def prune_model(model, ds_name, encoding_bits, ratio=0.5, device=None, uniform_pruning=True):
    if device is None:
        device = default_device

    batch_size = 32
    
    compute_new_model_size(model, ratio)

    train_set, val_set, test_set = get_dataset(ds_name, encoding_bits)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size)

    model = model.to(device)
    model.eval()
    test_total, test_correct = run_inference(model, test_loader)
    print(f"Before pruning: Test set: Correct: {test_correct}/{test_total} ({round(((100*test_correct)/test_total).item(), 3)}%)")

    classes = model.models[0].classes

    print("Compute scores")
    scores = [torch.zeros((classes, m.filters_per_discriminator)) for m in model.models]
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        for model_idx in range(len(model.models)):
            responses = model.models[model_idx].get_filter_responses(features)
            for sample_idx in range(features.shape[0]):
                label = labels[sample_idx]
                responses[sample_idx][label] *= -(classes-1)
            scores[model_idx] -= responses.sum(axis=0).cpu().detach()
    if uniform_pruning:
        for model_idx in range(len(model.models)):
            scores[model_idx] = scores[model_idx].amax(axis=0)
    print("Zero filters")
    for model_idx in range(len(model.models)):
        prune_count = int(ratio * model.models[model_idx].filters_per_discriminator)
        prune_idxs = (-scores[model_idx]).topk(prune_count).indices
        if uniform_pruning:
            # Shrink the model
            m = model.models[model_idx]
            keep_idxs = torch.LongTensor([i for i in range(m.filters_per_discriminator) if i not in prune_idxs])
            input_order = m.input_order
            new_input_order = torch.cat([input_order[m.filter_inputs*i:m.filter_inputs*(i+1)] for i in keep_idxs])
            new_data = m.table[:,keep_idxs].clone()
            if hasattr(m, "bleach_weights"):
                new_bleach_weights = m.bleach_weights[:,keep_idxs]
            with torch.no_grad():
                m.input_order.resize_(len(new_input_order))
                # PyTorch refuses to resize tensors which require grad - so we use a hack
                m.table.requires_grad = False
                m.table.resize_(classes, len(keep_idxs), m.filter_entries)
                print(classes, len(keep_idxs), m.filter_entries)
                m.table.requires_grad = True
                m.mask.resize_(classes, len(keep_idxs))
                if hasattr(m, "bleach_weights"):
                    m.bleach_weights.resize_(classes, len(keep_idxs), m.filter_entries)
                m.input_order.data = new_input_order.to(device)
                m.table = nn.Parameter(new_data.to(device))
                m.mask = nn.Parameter(torch.ones((classes, len(keep_idxs))).to(device), requires_grad=False)
                m.filters_per_discriminator = len(keep_idxs)
                if hasattr(m, "bleach_weights"):
                    m.bleach_weights = nn.Parameter(new_bleach_weights, requires_grad=False)
            #model.models[model_idx].mask[:,prune_idxs] = 0 # Workaround...
        else:
            for i in range(classes):
                model.models[model_idx].mask[i,prune_idxs[i]] = 0
                model.models[model_idx].pruned = True

    test_total, test_correct = run_inference(model, test_loader)
    print(f"After pruning: Test set: Correct: {test_correct}/{test_total} ({round(((100*test_correct)/test_total).item(), 3)}%)")
    
    print("Learn biases")
    for model_idx in range(len(model.models)):
        print(f"{model_idx+1}/{len(model.models)}")
        model_train_dset = process_dset(model.models[model_idx], train_loader, device=default_device)
        model_val_dset = process_dset(model.models[model_idx], val_loader, device=default_device)
        model_train_loader = torch.utils.data.DataLoader(dataset=model_train_dset, batch_size=batch_size)
        model_val_loader = torch.utils.data.DataLoader(dataset=model_val_dset, batch_size=batch_size)
        bias_model = BiasModel(classes)
        bias_model = train_bias_model(bias_model, model_train_loader, model_val_loader)
        model.models[model_idx].bias.data = bias_model.bias.data.detach().to(device)
  
    test_total, test_correct = run_inference(model, test_loader)
    print(f"Before fine-tuning: Test set: Correct: {test_correct}/{test_total} ({round(((100*test_correct)/test_total).item(), 3)}%)")

    print("Fine-tune models")
    #for m in model.models:
    #    m.bias.requires_grad = True
    model = train_model(model, train_loader, val_loader, num_epochs=30)
    
    test_total, test_correct = run_inference(model, test_loader)
    print(f"After fine-tuning: Test set: Correct: {test_correct}/{test_total} ({round(((100*test_correct)/test_total).item(), 3)}%)")
    return model

if __name__ == "__main__":
    model_fname, dset_name, ratio = sys.argv[1], sys.argv[2], float(sys.argv[3])
    ratio = min(max(ratio, 0.0), 1.0)
    out_fname = os.path.splitext(model_fname)[0] + f"_pruned_{str(ratio).replace('.', '_')}.pt"
    print(f"Out filename is {out_fname}")
    model = torch.load(model_fname)
    pruned_model = prune_model(model, dset_name, model.encoding_bits, ratio=ratio, uniform_pruning=True)
    torch.save(pruned_model, out_fname)
