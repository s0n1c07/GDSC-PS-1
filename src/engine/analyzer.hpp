#pragma once

#include "core/types.hpp"
#include <vector>
#include <cstdint>
#include <string>

namespace neuralscope {

/// Computes activation statistics from raw float data.
/// Also runs anomaly detection rules.
struct ActivationStats {
    float   min_val   = 0.0f;
    float   max_val   = 0.0f;
    float   mean_val  = 0.0f;
    float   std_val   = 0.0f;
    float   sparsity  = 0.0f;    // fraction of near-zero values
    bool    has_nan   = false;
    bool    has_inf   = false;
    int64_t n_elements = 0;

    std::vector<AnomalyEntry> anomalies;
};

/// Analyze a buffer of float values and return statistics + anomalies.
/// The data vector should contain float values extracted from the tensor.
/// epsilon: threshold for sparsity (values with |x| < epsilon are "zero")
ActivationStats analyze_activations(
    const std::vector<float>& data,
    const std::string& layer_name,
    const std::string& timestamp,
    float epsilon = 1e-6f
);

/// Lightweight version that only computes sparsity (for large tensors
/// where full stats are too expensive).
float compute_sparsity(const std::vector<float>& data, float epsilon = 1e-6f);

} // namespace neuralscope
