#!/usr/bin/env python3

################################################################################
# finalize_model.py
# Author: Zachary Susskind (ZSusskind@utexas.edu)
#
# Given a .pt file representing a trained (and optionally pruned) ULEEN model,
# this script converts it into a minimal (.pickle.lzma) format which is more
# easily processed by the RTL generation scripts. It also converts the dataset
# into a format which can be used by the RTL testbench, optionally with the
# unary->binary compression scheme described in the paper.
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
import numpy as np
from numba import jit

import os
import sys
import pickle
import lzma

from model import BinarizeFunction
from get_datasets import get_dataset, get_thresholds

# Computes H3 hash functions for a mapped input or batch of inputs
@jit(nopython=True)
def h3_hash(xv, m):
    if xv.ndim == 1:
        selected_entries = xv * m # np.where is unsupported in Numba
        reduction_result = np.zeros(m.shape[0], dtype=np.int64) # ".reduce" is unsupported in Numba
        for i in range(m.shape[1]):
            reduction_result ^= selected_entries[:,i]
    else:
        selected_entries = np.expand_dims(xv, 1) * m # np.where is unsupported in Numba
        reduction_result = np.zeros((xv.shape[0], m.shape[0]), dtype=np.int64) # ".reduce" is unsupported in Numba
        for i in range(m.shape[1]):
            reduction_result ^= selected_entries[:,:,i]
    return reduction_result

class BloomFilter:
    def __init__(self, data, hash_values):
        assert(isinstance(data, np.ndarray))
        assert(isinstance(hash_values, np.ndarray))
        self.data, self.hash_values = data, hash_values

    # Implementation of the check_membership function
    # Coding in this style (as a static method) is necessary to use Numba for JIT compilation
    @staticmethod
    def __check_membership(xv, hash_values, data):
        hash_results = h3_hash(xv, hash_values)
        if xv.ndim == 1:
            return int(data[hash_results].all())
        else:
            return data[hash_results.flatten()].reshape(hash_results.shape).all(axis=1).astype(np.int64)

    def check_membership(self, xv):
        return BloomFilter.__check_membership(xv, self.hash_values, self.data)

class Discriminator:
    def __init__(self, data, mask, hash_values):
        self.nonsparse_filters = {}
        self.num_filters = len(data)
        for i in range(self.num_filters):
            if mask[i]:
                self.nonsparse_filters[i] = BloomFilter(data[i], hash_values)

    def predict(self, xv):
        if xv.ndim == 1:
            filter_inputs = xv.reshape(self.num_filters, -1) # Divide the inputs between the filters
            response = 0
        else:
            filter_inputs = np.transpose(xv.reshape(xv.shape[0], self.num_filters, -1), (1, 0, 2))
            response = np.zeros(xv.shape[0], dtype=int)
        for idx, inp in enumerate(filter_inputs):
            if idx in self.nonsparse_filters:
                response += self.nonsparse_filters[idx].check_membership(inp)
        return response

class WiSARD:
    def __init__(self, data, mask, pad, hash_values, input_order):
        self.num_classes, _, self.unit_entries = data.shape
        self.unit_hashes, self.unit_inputs = hash_values.shape
        self.num_filters = mask.shape[1]
        self.discriminators = [Discriminator(data[i], mask[i], hash_values) for i in range(self.num_classes)]
        self.input_order = input_order
        self.pad = pad

    def predict(self, xv):
        if xv.ndim == 1:
            padded = np.pad(xv, (0, self.pad))
            mapped = padded[self.input_order]
        else:
            padded = np.pad(xv, ((0, 0), (0, self.pad)))
            mapped = padded[:,self.input_order]
        responses = np.array([d.predict(mapped) for d in self.discriminators], dtype=int)
        return responses

class EnsembleWiSARD:
    def __init__(self, data, mask, pad, hash_values, input_order, bias, num_inputs, encoding_bits):
        self.num_models = len(data)
        self.wisard_models = [WiSARD(data[i], mask[i], pad[i], hash_values[i], input_order[i]) for i in range(self.num_models)]
        self.bias = bias
        self.num_inputs = num_inputs
        self.encoding_bits = encoding_bits

    def predict(self, xv):
        responses = np.array([w.predict(xv) for w in self.wisard_models], dtype=int)
        total_responses = responses.sum(axis=0)
        if xv.ndim == 2:
            total_responses = total_responses.T
        if self.bias is not None:
            total_responses += self.bias
        if xv.ndim == 1:
            return np.argmax(total_responses)
        else:
            return np.argmax(total_responses, axis=1)

def get_unused_inputs(model, inputs):
    used = np.zeros(inputs)
    encoding_bits = model.encoding_bits
    max_input = 0
    for m in model.models:
        max_input = max(max_input, m.input_order.max().item())
        flat_mask = m.mask.any(axis=0)
        for i in range(m.filters_per_discriminator):
            if flat_mask[i]:
                for j in range(i*m.filter_inputs, (i+1)*m.filter_inputs):
                    input_idx = m.input_order[j].item() // encoding_bits
                    if input_idx < inputs: # Need to worry about zero padding at end...
                        used[input_idx] = 1

    # TODO: Handling unused inputs is currently buggy, so this is a workaround
    used = np.ones(inputs)
    unused_inputs = np.where(used == 0)[0]
    input_remap = np.zeros(max_input+1, dtype=int)
    input_idx = 0
    remap_idx = 0
    for i in used:
        for j in range(encoding_bits):
            if i:
                input_remap[input_idx] = remap_idx
                remap_idx += 1
            else:
                input_remap[input_idx] = -1
            input_idx += 1
    for i in range(inputs*encoding_bits, max_input+1): # Handle zero pad inputs
        input_remap[input_idx] = remap_idx
        input_idx += 1
        remap_idx += 1
    return unused_inputs, input_remap


def finalize_model(model, inputs, unused_inputs, input_remap, adjust_bias = True):
    ensemble_data = [((BinarizeFunction.apply(m.table).detach()+1)/2).numpy().astype(bool) for m in model.models]
    ensemble_mask = [m.mask.numpy() for m in model.models]
    ensemble_pad = [m.null_bits for m in model.models]
    ensemble_hash_values = [m.hash_values.numpy() for m in model.models]
    ensemble_input_order = [input_remap[m.input_order.cpu().numpy()] for m in model.models]
    if adjust_bias:
        # for conversion from {-1, 1} binarization in training to {0, 1} binarization in inference
        # t = p + n; a = p - n + b
        # a = 2p + (b - t)
        adj_biases = [(m.bias.numpy() - m.filters_per_discriminator)/2 for m in model.models]
        ensemble_bias = sum(adj_biases).round().astype(int)
    else:
        ensemble_bias = sum([m.bias.numpy() for m in model.models]).round().astype(int)
    ensemble_bias -= ensemble_bias.min()
    encoding_bits = model.encoding_bits
    ensemble_model = EnsembleWiSARD(\
            ensemble_data, ensemble_mask, ensemble_pad,\
            ensemble_hash_values, ensemble_input_order, ensemble_bias,\
            inputs, encoding_bits)
    return ensemble_model

def thermometer_encode_dataset(dset, thresholds, unused_inputs, compressed=True):
    inputs = thresholds.shape[0] - len(unused_inputs)
    kept_inputs = torch.Tensor([i for i in range(thresholds.shape[0]) if i not in unused_inputs]).long()
    kept_thresholds = thresholds[kept_inputs]
    encoding_bits = thresholds.shape[1]
    if compressed:
        compressed_encoding_bits = int(np.ceil(np.log2(encoding_bits+1)))
        bytes_per_sample = int(np.ceil((inputs * compressed_encoding_bits) / 8))
    else:
        bytes_per_sample = int(np.ceil((inputs * encoding_bits) / 8))
    encoded_dset = np.array(bytes_per_sample).astype(">u4").tobytes() # Header - big-endian 32-bit value
    encoded_dset += np.array(compressed).astype("u1").tobytes() # Indicates whether dataset is compressed
    encoded_dset += np.array(encoding_bits).astype("u1").tobytes()
    for data, label in dset:
        duplicated = data.view(-1, 1)[kept_inputs].expand(-1, encoding_bits)
        if compressed:
            quantized = (duplicated >= kept_thresholds).sum(axis=1).numpy().astype("u1")
            binarized = np.flip(np.unpackbits(quantized.reshape(-1, 1), axis=1)[:,-compressed_encoding_bits:], axis=0).flatten()
        else:
            binarized = np.flip((duplicated >= kept_thresholds).numpy(), axis=(0,1)).flatten().astype(bool)
        encoded_dset += np.packbits(binarized).tobytes()
        encoded_dset += np.array(label).astype("u1").tobytes()
    return encoded_dset

def save_model(model, fname):
    submodel_info = []
    for m in model.wisard_models:
        submodel_info.append({
            "num_filters": m.num_filters,
            "num_filter_inputs": m.unit_inputs,
            "num_filter_entries": m.unit_entries,
            "num_filter_hashes": m.unit_hashes,
            "num_null_bits":m.pad,
            "nonsparse_filter_idxs": [sorted(list(d.nonsparse_filters.keys())) for d in m.discriminators],
            "input_order": list(m.input_order),
            "hash_values": next(iter(m.discriminators[0].nonsparse_filters.values())).hash_values
        })
    model_info = {
        "num_inputs": model.num_inputs * model.encoding_bits,
        "num_classes": len(model.wisard_models[0].discriminators),
        "bits_per_input": model.encoding_bits,
        "bias": list(model.bias),
        "submodel_info": submodel_info
    }
    state_dict = {
        "info": model_info,
        "model": model
    }

    with lzma.open(fname, "wb") as f:
        pickle.dump(state_dict, f)

def run_inference(model_fname, dset_fname, batch_size=8192):
    with lzma.open(model_fname, "rb") as f:
        model = pickle.load(f)["model"]
    num_inputs = model.num_inputs
    total = 0
    correct = 0
    samples = []
    labels = []
    with open(dset_fname, "rb") as ds:
        bytes_per_sample = int.from_bytes(ds.read(4), "big")
        compressed = bool.from_bytes(ds.read(1), "big")
        encoding_bits = int.from_bytes(ds.read(1), "big")
        if compressed:
            compressed_encoding_bits = int(np.ceil(np.log2(encoding_bits+1)))
        while True:
            sample_bytes = ds.read(bytes_per_sample)
            if len(sample_bytes) == 0:
                break # EOF
            sample_bits = np.unpackbits(np.frombuffer(sample_bytes, dtype="u1"))
            if compressed:
                trimmed_sample_bits = sample_bits[0:compressed_encoding_bits*num_inputs]
                sample_counts = np.flip(trimmed_sample_bits.reshape(-1, compressed_encoding_bits).dot(1 << np.arange(compressed_encoding_bits-1, -1, -1)))
                sample = (np.arange(0, encoding_bits).reshape(-1, 1).repeat(len(sample_counts), 1) < sample_counts).T.flatten()
            else:
                trimmed_sample_bits = sample_bits[0:encoding_bits*num_inputs]
                sample = np.flip(trimmed_sample_bits.reshape(-1, encoding_bits), axis=(0,1)).flatten().astype(bool)
            label = int.from_bytes(ds.read(1), "big")
            samples.append(sample)
            labels.append(label)
    samples = np.array(samples)
    labels = np.array(labels)
    if batch_size == 0:
        for sample, label in zip(samples, labels):
            prediction = model.predict(sample)
            if prediction == label:
                correct += 1
            total += 1
            if total % 1000 == 0:
                print(total)
    else:
        assert(batch_size > 0)
        batch_start = 0
        while(batch_start < len(samples)):
            batch_end = min(batch_start+batch_size, len(samples))
            #print(batch_start, batch_size, batch_end, len(samples))
            batch_samples = samples[batch_start:batch_end]
            batch_labels = labels[batch_start:batch_end]
            predictions = model.predict(batch_samples)
            for prediction, label in zip(predictions, batch_labels):
                if prediction == label:
                    correct += 1
            if batch_end // 1000 > batch_start // 1000:
                print(batch_end)
            total = batch_end
            batch_start = batch_end
    print(f"Correct: {correct}/{total} ({(100*correct)/total}%)")

def main(model_fname, dset_name, compress=None):
    model = torch.load(model_fname, map_location="cpu").to("cpu")
    if compress is None:
        compress = (model.encoding_bits > 2)

    train_dataset, _, test_dataset = get_dataset(dset_name, None, None)
    thresholds = get_thresholds(train_dataset.tensors[0], model.encoding_bits)
    inputs = test_dataset[0][0].shape[0]
    print("Get unused inputs")
    unused_inputs, input_remap = get_unused_inputs(model, inputs)
    print(f"Found {len(unused_inputs)} unused inputs")
    print("Finalize model")
    finalized_model = finalize_model(model, inputs, unused_inputs, input_remap)
    print("Convert dataset")
    encoded_dataset = thermometer_encode_dataset(test_dataset, thresholds,
                                                 unused_inputs, compress)
    model_out_fname = os.path.splitext(model_fname)[0] + "_finalized.pickle.lzma"
    print("Save results")
    save_model(finalized_model, model_out_fname)
    compressed_str = "_compressed" if compress else ""
    dset_out_fname = f"{dset_name}_encoded_{thresholds.shape[1]}b{compressed_str}.bds"
    with open(dset_out_fname, "wb") as f:
        f.write(encoded_dataset)
    print("Run inference (debug)")
    run_inference(model_out_fname, dset_out_fname)

if __name__ == "__main__":
    model_fname, dset_name = sys.argv[1:]
    main(model_fname, dset_name)

