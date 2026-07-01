#include "engine/analyzer.hpp"
#include "core/utils.hpp"

#include <cmath>
#include <algorithm>
#include <limits>

namespace neuralscope {

ActivationStats analyze_activations(
    const std::vector<float>& data,
    const std::string& layer_name,
    const std::string& timestamp,
    float epsilon)
{
    ActivationStats stats;
    stats.n_elements = static_cast<int64_t>(data.size());

    if (data.empty()) {
        return stats;
    }

    // Single-pass: min, max, mean, variance (Welford's algorithm), sparsity
    double sum      = 0.0;
    double sum_sq   = 0.0;   // For variance via E[X^2] - E[X]^2
    int64_t n_zero  = 0;
    float min_v     = std::numeric_limits<float>::max();
    float max_v     = std::numeric_limits<float>::lowest();

    for (size_t i = 0; i < data.size(); ++i) {
        float v = data[i];

        // NaN / Inf check
        if (std::isnan(v)) {
            stats.has_nan = true;
            continue;
        }
        if (std::isinf(v)) {
            stats.has_inf = true;
            continue;
        }

        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum += static_cast<double>(v);
        sum_sq += static_cast<double>(v) * static_cast<double>(v);

        if (std::fabs(v) < epsilon) {
            ++n_zero;
        }
    }

    int64_t valid_count = stats.n_elements - (stats.has_nan ? 1 : 0);
    if (valid_count > 0) {
        double mean = sum / static_cast<double>(valid_count);
        double var  = (sum_sq / static_cast<double>(valid_count)) - (mean * mean);
        if (var < 0.0) var = 0.0; // Numerical safety

        stats.min_val  = min_v;
        stats.max_val  = max_v;
        stats.mean_val = static_cast<float>(mean);
        stats.std_val  = static_cast<float>(std::sqrt(var));
        stats.sparsity = static_cast<float>(n_zero) / static_cast<float>(valid_count);
    }

    // ─── Anomaly detection rules ───────────────────────────────
    // Rule 1: NaN detected
    if (stats.has_nan) {
        stats.anomalies.push_back({
            timestamp,
            Severity::Critical,
            "NaN detected in " + layer_name
        });
    }

    // Rule 2: Inf detected
    if (stats.has_inf) {
        stats.anomalies.push_back({
            timestamp,
            Severity::Critical,
            "Inf detected in " + layer_name
        });
    }

    // Rule 3: Clipping risk (fp16 max = 65504)
    if (stats.max_val > 65504.0f || stats.min_val < -65504.0f) {
        stats.anomalies.push_back({
            timestamp,
            Severity::Critical,
            "Clipping risk in " + layer_name +
            ": Max=" + format_float(stats.max_val, 1)
        });
    }

    // Rule 4: Dead layer (very high sparsity)
    if (stats.sparsity > 0.90f) {
        stats.anomalies.push_back({
            timestamp,
            Severity::Warning,
            "Dead layer " + layer_name +
            ": Sparsity=" + format_float(stats.sparsity * 100.0f, 1) + "%"
        });
    }

    // Rule 5: Outlier mean
    if (std::fabs(stats.mean_val) > 100.0f) {
        stats.anomalies.push_back({
            timestamp,
            Severity::Warning,
            "Outlier mean in " + layer_name +
            ": Mean=" + format_float(stats.mean_val, 2)
        });
    }

    // Rule 6: Collapsed distribution
    if (stats.std_val < 1e-6f && valid_count > 1) {
        stats.anomalies.push_back({
            timestamp,
            Severity::Warning,
            "Collapsed distribution in " + layer_name +
            ": Std=" + format_float(stats.std_val, 8)
        });
    }

    return stats;
}

float compute_sparsity(const std::vector<float>& data, float epsilon) {
    if (data.empty()) return 0.0f;
    int64_t n_zero = 0;
    for (float v : data) {
        if (std::fabs(v) < epsilon) ++n_zero;
    }
    return static_cast<float>(n_zero) / static_cast<float>(data.size());
}

} // namespace neuralscope
