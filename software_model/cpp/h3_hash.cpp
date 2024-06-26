// h3_hash.cpp
// Author: Zachary Susskind (ZSusskind@utexas.edu)

#include <torch/extension.h>

#include <vector>

#include "h3_hash.h"

using torch::Tensor;

// NOTE: This is just a utility function
// Python loop handling is slow, this function is a bottleneck, and
// profiling shows that H3 computation is a significant portion of training time

std::vector<at::Tensor> h3_hash(
    Tensor inp,
    Tensor hash_vals
) {
    const auto device = inp.device();

    // Choose between hash values and 0 based on input bits
    // This is done using a tensor product, which, oddly, seems to be faster
    // than using a conditional lookup (e.g. torch.where)
    Tensor selected_entries = torch::einsum("hb,db->bdh", {hash_vals, inp});

    // Perform an XOR reduction along the input axis (b dimension)
    Tensor reduction_result = torch::zeros(
        {inp.size(0), hash_vals.size(0)},
        torch::dtype(torch::kInt64).device(device));
    for (size_t i = 0; i < hash_vals.size(1); i++) {
        reduction_result.bitwise_xor_(selected_entries[i]); // In-place XOR
    }

    return { reduction_result };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("h3_hash", &h3_hash, "Compute H3 hash function");
}
