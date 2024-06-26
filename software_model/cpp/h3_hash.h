// h3_hash.cpp
// Author: Zachary Susskind (ZSusskind@utexas.edu)

#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> h3_hash(
    torch::Tensor inp,
    torch::Tensor hash_vals
);
