#pragma once

#include "core/types.hpp"
#include "core/ring_buffer.hpp"

#include "ggml.h"
#include "ggml-backend.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

namespace neuralscope {

/// Manages ggml eval callbacks for non-invasive tensor inspection.
/// Hooks into the backend scheduler to intercept every tensor computation.
class HookManager {
public:
    HookManager(RingBuffer<LayerSnapshot>& buffer,
                RingBuffer<AnomalyEntry>& anomaly_buffer);

    /// Get the C-style callback function pointer for ggml.
    /// Pass this to ggml_backend_sched_set_eval_callback.
    static bool eval_callback(struct ggml_tensor* t, bool ask, void* user_data);

    /// Set a filter: only inspect tensors whose names match these patterns.
    /// Empty = inspect all tensors.
    void set_filter_patterns(const std::vector<std::string>& patterns);

    /// Enable/disable hooking at runtime.
    void set_enabled(bool enabled);
    bool is_enabled() const;

    /// Get the running snapshot counter.
    int64_t get_snapshot_count() const;

    /// Reset the counter and timers.
    void reset();

    /// Set the active layer name (for topology highlighting).
    std::string get_active_layer() const;

    /// Max tensor elements to fully analyze (larger tensors get sampled).
    void set_max_analysis_elements(size_t max_elements);

private:
    /// Called when ask=true: decide whether to inspect this tensor.
    bool on_ask(struct ggml_tensor* t);

    /// Called when ask=false: tensor data is ready, extract stats.
    bool on_inspect(struct ggml_tensor* t);

    /// Check if tensor name matches any filter pattern.
    bool matches_filter(const std::string& name) const;

    RingBuffer<LayerSnapshot>&  snapshot_buffer_;
    RingBuffer<AnomalyEntry>&   anomaly_buffer_;

    std::atomic<bool>           enabled_{true};
    std::atomic<int64_t>        snapshot_count_{0};
    size_t                      max_analysis_elements_ = 65536; // 64K elements max

    // Filter patterns (empty = accept all)
    std::vector<std::string>    filter_patterns_;
    mutable std::mutex          filter_mutex_;

    // Per-tensor timing: start time stored when ask=true
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> timers_;
    std::mutex                  timer_mutex_;

    // Active layer tracking
    mutable std::mutex          active_mutex_;
    std::string                 active_layer_;
};

} // namespace neuralscope
