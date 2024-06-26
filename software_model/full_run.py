#!/usr/bin/env python3

################################################################################
# full_run.py
# Author: Zachary Susskind (ZSusskind@utexas.edu)
#
# Performs the three steps in the ULEEN model training process (initial
# training, pruning/fine-tuning, and "finalization" for inference) in one fell 
# swoop. Takes a JSON config file as input.
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

import os
import sys
import json

from train_model import create_model
from prune_model import prune_model
from finalize_model import main as finalize_main

def main(config_json_fname):
    with open(config_json_fname, "r") as f:
        config = json.load(f)

    print("Step 1/3: Train ULEEN model")
    model_config = config["model"]
    dataset = model_config["dataset"]
    configs = model_config["configs"]
    encoding_bits = model_config["encoding_bits"]
    model_fname = model_config.get("output", "model.pt")
    num_epochs = model_config.get("epochs", 100)
    learning_rate = model_config.get("lr", 1e-3)
    decay_lr = model_config.get("decay_lr", False)
    encoding = model_config.get("encoding", "gaussian")
    dropout_p = model_config.get("dropout_p", 0.0)
    batch_size = int(model_config.get("batch_size", 32))
    create_model(dataset, configs, encoding_bits,
                 model_fname, num_epochs, learning_rate,
                 decay_lr, encoding, dropout_p,
                 batch_size=batch_size)

    print("Step 2/3: Prune ULEEN model")
    pruning_ratio = config.get("pruning_ratio", 0.0)
    assert 0.0 <= pruning_ratio < 1.0
    uniform_pruning = config.get("uniform_pruning", True)
    if pruning_ratio == 0.0:
        print("Skipping pruning since provided ratio is 0.0")
        pruned_fname = model_fname
    else:
        pruned_fname = os.path.splitext(model_fname)[0] + \
            f"_pruned_{str(pruning_ratio).replace('.', '_')}.pt"
        model = torch.load(model_fname)
        pruned_model = prune_model(model, dataset, encoding_bits,
                                   ratio=pruning_ratio,
                                   uniform_pruning=uniform_pruning)
        torch.save(pruned_model, pruned_fname)

    print("Step 3/3: Finalize pruned model")
    compress = config.get("compress_input")
    finalize_main(pruned_fname, dataset, compress)

if __name__ == "__main__":
    main(sys.argv[1])
