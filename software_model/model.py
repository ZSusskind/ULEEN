#!/usr/bin/false

################################################################################
# model.py
# Author: Zachary Susskind (ZSusskind@utexas.edu)
#
# Implementation of the actual ULEEN model.
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

import os
import numpy as np

# The H3 hash is pretty slow to evaluate in Python, so instead I use a libtorch
# C++ extension, which is compiled here if it's not already up-to-date
from torch.utils.cpp_extension import load
base_dir = os.path.dirname(__file__)
build_dir = os.path.join(base_dir, "./build")
if not os.path.exists(build_dir):
    os.mkdir(build_dir)
print("Building CPP sources (if needed); this may take a while")
h3_hash_cpp = load(name="h3_hash_cpp", sources=[os.path.join(base_dir, "./cpp/h3_hash.cpp")], build_directory=build_dir)

# Computes hash functions within the H3 family of integer-integer hashing 
# functions, as described by Carter and Wegman in the paper
# "Universal Classes of Hash Functions" (1979)
# Inputs:
#  x:         A 2D tensor (dxb) consisting of d b-bit values to be hashed,
#             expressed as bitvectors
#  hash_vals: A 2D tensor (hxb) consisting of h sets of b int64s, representing
#             random constants to compute h unique hashes
# Returns:   A 2D tensor (dxh) of int64s, representing the results of the h hash
#            functions on the d input values
def h3_hash(x, hash_vals):
    return h3_hash_cpp.h3_hash(x, hash_vals)[0]

# Generates random constants for H3 hash functions
# Inputs:
#  filter_inputs:  Number of inputs to each Bloom filter (b)
#  filter_entries: Number of entries in each Bloom filter's LUT
#  filter_hash_functions:  Number of unique sets of H3 parameters to generate
# Returns: A 2D tensor (hxb) of random int64s
def generate_h3_values(filter_inputs, filter_entries, filter_hash_functions):
    assert(np.log2(filter_entries).is_integer())
    shape = (filter_hash_functions, filter_inputs)
    values = torch.from_numpy(np.random.randint(0, filter_entries, shape))
    return values

# Performs sign-based binarization, using the straight-through estimator
# Derived from the method described in "Binarized Neural Networks" by 
# Hubara et al. (NeurIPS 2016)
class BinarizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        outp = ((inp >= 0).float() * 2) - 1 # -1 / 1 binarization
        return outp
   
    @staticmethod
    def backward(ctx, grad_outp):
        # Straight-through estimator
        # Note that the gradient cancellation used for BNNs is not needed
        # for ULEEN, since table entries are explicitly confined to the range
        # [-1, +1] during training
        return grad_outp

# Implementation of a single ULEEN submodel
class BackpropWiSARD(nn.Module):
    def __init__(
        self, inputs, classes, filter_inputs, filter_entries,
        filter_hash_functions, dropout_p=0.0
    ):
        super().__init__()

        # The commented block below is vestigial from an exploration into
        # ULEEN models with very small n, where using Bloom filters didn't
        # make logical sense. It's still there if you want to play with it,
        # but the RTL doesn't currently support it.
        self.use_luts = False
        """
        if (filter_entries >= (1 << filter_inputs)):
            print("Using LUTs instead of Bloom filters for config with "\
                  f"{filter_inputs} filter inputs and "\
                  f"{filter_entries} filter entries")
            filter_entries = (1 << filter_inputs)
            filter_hash_functions = 1
            self.use_luts = True
        else:
            self.use_luts = False
        """

        self.inputs = inputs
        self.classes = classes
        self.filter_inputs = filter_inputs
        self.filter_entries = filter_entries
        self.filter_hash_functions = filter_hash_functions
       
        # Get input information and number of filters
        # Total number of inputs to the model, padded to an integer # of filters
        input_bits = int(np.ceil(inputs/filter_inputs))\
                     * filter_inputs 
        # Extra bits to make model inputs an integer multiple of filter inputs
        self.null_bits = input_bits - inputs
        self.filters_per_discriminator = input_bits // filter_inputs
       
        # Initialize table tensor (3D tensor - discriminator x filter x entry)
        self.table = nn.Parameter(torch.Tensor(classes,
                                               self.filters_per_discriminator,
                                               filter_entries))
        nn.init.uniform_(self.table, -1, 1)

        if self.use_luts:
            self.bit_scalars = nn.Parameter((1 << torch.arange(filter_inputs)),
                                            requires_grad=False)
        else:
            self.hash_values = nn.Parameter(
                generate_h3_values(filter_inputs,
                                   filter_entries,
                                   filter_hash_functions),
                requires_grad=False)

        input_order = np.arange(input_bits).astype(int)
        np.random.shuffle(input_order)
        self.input_order = nn.Parameter(torch.from_numpy(input_order).long(),
                                        requires_grad=False)

        if dropout_p > 0.0:
            self.dropout = nn.Dropout(p=dropout_p)
        else:
            self.dropout = nn.Identity()

        # For working with pruned models
        self.pruned = False
        self.bias = nn.Parameter(torch.zeros(classes), requires_grad=False)
        self.mask = nn.Parameter(torch.ones((classes,
                                             self.filters_per_discriminator)),
                                 requires_grad=False)

    def get_filter_responses(self, x_b, raw=False):
        batch_size = x_b.shape[0]
        
        # Pad inputs to integer multiple of unit size and reorder
        padded = nn.functional.pad(x_b, (0, self.null_bits))
        mapped = padded[:,self.input_order]

        # Hash filter inputs (using H3)
        hash_inputs = mapped.view(batch_size*self.filters_per_discriminator,
                                  self.filter_inputs)
        if self.use_luts:
            hash_outputs = (self.bit_scalars * hash_inputs).sum(axis=1,
                                                                keepdim=True)
        else:
            hash_outputs = h3_hash(hash_inputs, self.hash_values)

        # Perform table lookup - this requires some weird data munging
        #  1. Reshape hash outputs into 3D tensor with dimensions
        #     discriminator x filter x (batch * hashfunction), duplicating
        #     (expanding) along discriminator axis
        filter_inputs = hash_outputs\
                .view(batch_size,
                      self.filters_per_discriminator,
                      self.filter_hash_functions)\
                .permute(1, 0, 2)\
                .reshape(1, self.filters_per_discriminator, -1)\
                .expand(self.classes, -1, -1)
        #  2. Perform high-dimensional table lookup using gather operation
        flat_lookup = self.table.gather(2, filter_inputs)
        #  3. Restructure 3D discriminator x filter x (batch * hashfunction)
        #     tensor into 4D batch x discriminator x filter x hashfunction
        lookup = flat_lookup\
                .view(self.classes,
                      self.filters_per_discriminator,
                      batch_size,
                      self.filter_hash_functions)\
                .permute(2, 0, 1, 3)

        if raw:
            return lookup

        # 4. Binarize and reduce along hashfunction dimension
        bin_lookup = BinarizeFunction.apply(lookup) # Binarize to -1/1
        reduced = bin_lookup.amin(axis=-1) 
        return self.dropout(reduced)

    def forward(self, x_b):
        responses = self.get_filter_responses(x_b)
        if self.pruned:
            masked_responses = responses * self.mask
        else:
            masked_responses = responses
        # Calculate activations for each discriminaton
        activations = masked_responses.sum(axis=2) + self.bias 
        return activations

    def clamp(self): # Clamp data to [-1, 1]
        self.table.data.clamp_(-1, 1)

# Ensemble of BackpropWiSARD models (i.e., ULEEN submodels)
# Returns results as a 3D tensor (models x batch x discriminators)
class BackpropMultiWiSARD(nn.Module):
    # Configs is a list of 3-entry tuples:
    #  filter_inputs, filter_entries, filter_hash_functions
    def __init__(self, inputs, classes, configs, encoding_bits, dropout_p=0.0):
        super().__init__()
        self.encoding_bits = encoding_bits
        self.models = nn.ModuleList()
        for config in configs:
            self.models.append(BackpropWiSARD(inputs,
                                              classes,
                                              *config,
                                              dropout_p))

    def forward(self, x_b):
        batch_size = x_b.shape[0]
        
        # Run inference for each component submodel
        model_results = torch.stack([m(x_b) for m in self.models]) 

        return model_results
    
    def clamp(self):
        for m in self.models:
            m.clamp()
