#include "engine/hook_manager.hpp"
#include "engine/analyzer.hpp"
#include "core/utils.hpp"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstring>
#include <algorithm>
#include <cmath>

namespace neuralscope {

HookManager::HookManager(RingBuffer<LayerSnapshot>& buffer,
                         RingBuffer<AnomalyEntry>& anomaly_buffer)
    : snapshot_buffer_(buffer)
    , anomaly_buffer_(anomaly_buffer)
{}

bool HookManager::eval_callback(struct ggml_tensor* t, bool ask, void* user_data) {
    auto* self = static_cast<HookManager*>(user_data);
    if (!self || !self->enabled_.load()) return false;

    if (ask) {
        return self->on_ask(t);
    } else {
        return self->on_inspect(t);
    }
}

void HookManager::set_filter_patterns(const std::vector<std::string>& patterns) {
    std::lock_guard<std::mutex> lock(filter_mutex_);
    filter_patterns_ = patterns;
}

void HookManager::set_enabled(bool enabled) {
    enabled_.store(enabled);
}

bool HookManager::is_enabled() const {
    return enabled_.load();
}

int64_t HookManager::get_snapshot_count() const {
    return snapshot_count_.load();
}

void HookManager::reset() {
    snapshot_count_.store(0);
    {
        std::lock_guard<std::mutex> lock(timer_mutex_);
        timers_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(active_mutex_);
        active_layer_.clear();
    }
}

std::string HookManager::get_active_layer() const {
    std::lock_guard<std::mutex> lock(active_mutex_);
    return active_layer_;
}

void HookManager::set_max_analysis_elements(size_t max_elements) {
    max_analysis_elements_ = max_elements;
}

bool HookManager::matches_filter(const std::string& name) const {
    std::lock_guard<std::mutex> lock(filter_mutex_);
    if (filter_patterns_.empty()) return true; // No filter = accept all

    for (const auto& pattern : filter_patterns_) {
        if (name.find(pattern) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool HookManager::on_ask(struct ggml_tensor* t) {
    if (!t || !t->name) return false;

    std::string name(t->name);

    // Skip very small utility tensors and internal ops
    if (name.empty() || name[0] == '#') return false;

    // Check filter
    if (!matches_filter(name)) return false;

    // Start timer for this tensor
    {
        std::lock_guard<std::mutex> lock(timer_mutex_);
        timers_[name] = std::chrono::high_resolution_clock::now();
    }

    // Update active layer
    {
        std::lock_guard<std::mutex> lock(active_mutex_);
        active_layer_ = name;
    }

    return true; // Yes, we want to inspect this tensor
}

bool HookManager::on_inspect(struct ggml_tensor* t) {
    if (!t || !t->name) return true;

    std::string name(t->name);
    std::string ts = now_timestamp();

    // Calculate latency
    double latency_ms = 0.0;
    {
        std::lock_guard<std::mutex> lock(timer_mutex_);
        auto it = timers_.find(name);
        if (it != timers_.end()) {
            auto end = std::chrono::high_resolution_clock::now();
            latency_ms = std::chrono::duration<double, std::milli>(end - it->second).count();
            timers_.erase(it);
        }
    }

    // Build the layer snapshot
    LayerSnapshot snap;
    snap.id         = snapshot_count_.fetch_add(1);
    snap.timestamp  = ts;
    snap.layer_name = name;
    snap.layer_type = classify_tensor(name);
    snap.latency_ms = latency_ms;

    // Tensor shape
    snap.tensor_shape = format_shape(t->ne);
    snap.n_elements   = ggml_nelements(t);

    // Dtype & Op
    snap.dtype   = std::string(ggml_type_name(t->type));
    snap.op_name = std::string(ggml_op_name(t->op));

    // Compute device
    if (t->buffer) {
        if (ggml_backend_buffer_is_host(t->buffer)) {
            snap.compute_device = "CPU";
        } else {
            snap.compute_device = "GPU";
        }
    } else {
        snap.compute_device = "CPU";
    }

    // ─── Extract tensor data for analysis ──────────────────────
    // Only analyze tensors up to max_analysis_elements_ to control RAM
    size_t n_elements = static_cast<size_t>(ggml_nelements(t));
    size_t analyze_count = std::min(n_elements, max_analysis_elements_);

    if (analyze_count > 0 && t->type == GGML_TYPE_F32) {
        // For f32 tensors, read data directly
        std::vector<float> data(analyze_count);

        if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) {
            // GPU tensor: copy to host
            ggml_backend_tensor_get(t, data.data(), 0,
                                   analyze_count * sizeof(float));
        } else if (t->data) {
            // CPU tensor: direct memcpy
            std::memcpy(data.data(), t->data,
                        analyze_count * sizeof(float));
        }

        // Run analysis
        auto stats = analyze_activations(data, name, ts);
        snap.sparsity  = stats.sparsity;
        snap.act_min   = stats.min_val;
        snap.act_max   = stats.max_val;
        snap.act_mean  = stats.mean_val;
        snap.act_std   = stats.std_val;
        snap.anomalies = stats.anomalies;

        // Push anomalies to the anomaly buffer
        for (const auto& a : stats.anomalies) {
            anomaly_buffer_.push(a);
        }
    } else if (analyze_count > 0 && t->type == GGML_TYPE_F16) {
        // For f16 tensors, we need to convert
        // Read raw f16 data, then convert to f32
        std::vector<uint16_t> raw(analyze_count);
        if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) {
            ggml_backend_tensor_get(t, raw.data(), 0,
                                   analyze_count * sizeof(uint16_t));
        } else if (t->data) {
            std::memcpy(raw.data(), t->data,
                        analyze_count * sizeof(uint16_t));
        }

        std::vector<float> data(analyze_count);
        for (size_t i = 0; i < analyze_count; ++i) {
            data[i] = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(raw[i]));
        }

        auto stats = analyze_activations(data, name, ts);
        snap.sparsity  = stats.sparsity;
        snap.act_min   = stats.min_val;
        snap.act_max   = stats.max_val;
        snap.act_mean  = stats.mean_val;
        snap.act_std   = stats.std_val;
        snap.anomalies = stats.anomalies;

        for (const auto& a : stats.anomalies) {
            anomaly_buffer_.push(a);
        }
    } else if (analyze_count > 0) {
        // For quantized types (q4_K, q8_0, etc.), we can dequantize a sample
        // Use ggml's built-in dequantization
        size_t sample_size = std::min(analyze_count, (size_t)4096);
        std::vector<float> data(sample_size);

        // ggml_backend_tensor_get returns raw quantized data
        // For simplicity, just record shape info without deep analysis
        // on heavily quantized tensors
        snap.sparsity = 0.0f;
        snap.act_min  = 0.0f;
        snap.act_max  = 0.0f;
        snap.act_mean = 0.0f;
        snap.act_std  = 0.0f;
    }

    // Store attention weights if this is the softmax attention pattern tensor.
    // In llama.cpp the key tensor names vary by version. We check:
    // 1. Name-based: "KQ_soft_max", "soft_max", "kqv", "attn_out"
    // 2. Op-based: GGML_OP_SOFT_MAX (the softmax output IS the attention matrix)
    bool is_attn_softmax = (name.find("KQ_soft_max") != std::string::npos ||
                            name.find("kq_soft_max") != std::string::npos ||
                            name.find("soft_max")    != std::string::npos ||
                            name.find("kqv")         != std::string::npos ||
                            name.find("attn_out")    != std::string::npos ||
                            t->op == GGML_OP_SOFT_MAX);

    // Log attention-related tensor names for debugging
    if (name.find("attn") != std::string::npos ||
        name.find("KQ")   != std::string::npos ||
        name.find("kq")   != std::string::npos ||
        name.find("soft") != std::string::npos ||
        t->op == GGML_OP_SOFT_MAX ||
        t->op == GGML_OP_MUL_MAT) {
        debug_log("[ATTN_DEBUG] name=" + name +
                  " op=" + std::string(ggml_op_name(t->op)) +
                  " type=" + std::string(ggml_type_name(t->type)) +
                  " shape=[" + std::to_string(t->ne[0]) + "," +
                  std::to_string(t->ne[1]) + "," +
                  std::to_string(t->ne[2]) + "," +
                  std::to_string(t->ne[3]) + "]" +
                  " is_match=" + (is_attn_softmax ? "YES" : "NO"));
    }

    if (is_attn_softmax && t->type == GGML_TYPE_F32) {
        // ne[0] = sequence length (cols), ne[1] = sequence length (rows),
        // ne[2] = num heads, ne[3] = batch
        int seq_len  = static_cast<int>(t->ne[0]);
        int num_heads = static_cast<int>(t->ne[2] > 0 ? t->ne[2] : 1);
        snap.attn_size = seq_len;
        snap.num_heads = num_heads;

        // Cap at 64x64 per head to keep memory sane
        int cap = std::min(seq_len, 64);
        size_t elements_to_copy = static_cast<size_t>(cap) * cap * num_heads;
        elements_to_copy = std::min(elements_to_copy,
                                    static_cast<size_t>(ggml_nelements(t)));

        snap.attention_weights.resize(elements_to_copy);
        if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) {
            ggml_backend_tensor_get(t, snap.attention_weights.data(), 0,
                                    elements_to_copy * sizeof(float));
        } else if (t->data) {
            std::memcpy(snap.attention_weights.data(), t->data,
                        elements_to_copy * sizeof(float));
        }
        snap.attn_size = cap; // update to capped size
        debug_log("[ATTN_CAPTURED] name=" + name + " attn_size=" + std::to_string(cap) +
                  " num_heads=" + std::to_string(num_heads) +
                  " weights_count=" + std::to_string(snap.attention_weights.size()));
    }

    // Push to ring buffer
    snapshot_buffer_.push(std::move(snap));

    return true;
}

} // namespace neuralscope
