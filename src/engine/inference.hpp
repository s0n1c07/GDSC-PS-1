#pragma once

#include "core/types.hpp"
#include "core/ring_buffer.hpp"
#include "engine/hook_manager.hpp"
#include "engine/model_loader.hpp"

#include "llama.h"

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace neuralscope {

/// Status of the inference engine.
enum class InferenceState {
    Idle,
    Loading,
    Ready,
    Running,
    Finished,
    Error
};

/// Runs model inference in a background thread while hooks capture data.
class InferenceEngine {
public:
    InferenceEngine(
        RingBuffer<LayerSnapshot>& snapshot_buffer,
        RingBuffer<AnomalyEntry>&  anomaly_buffer
    );

    ~InferenceEngine();

    /// Load a model from a GGUF file (runs in background thread).
    void load_model_async(const std::string& model_path, int n_ctx = 512,
                          int n_threads = 4, bool use_gpu = false);

    /// Start generating text from a prompt (runs in background thread).
    void run_inference(const std::string& prompt, int max_tokens = 128);

    /// Stop the current inference (thread-safe).
    void stop();

    /// Get current state.
    InferenceState get_state() const;

    /// Get the generated text so far.
    std::string get_generated_text() const;

    /// Get tokens per second.
    double get_tokens_per_second() const;

    /// Get model loading progress (0.0 - 1.0).
    float get_load_progress() const;

    /// Get the hook manager for callback setup.
    HookManager& get_hook_manager();

    /// Get the model handle (may be null).
    ModelHandle* get_model() const;

    /// Set a callback for UI refresh (called from worker thread).
    void set_refresh_callback(std::function<void()> cb);

    /// Get last error message.
    std::string get_error() const;

    /// Get current token labels (thread-safe).
    std::vector<std::string> get_token_labels() const;

private:
    void worker_load(const std::string& model_path, int n_ctx, int n_threads, bool use_gpu);
    void worker_inference(const std::string& prompt, int max_tokens);

    RingBuffer<LayerSnapshot>& snapshot_buffer_;
    RingBuffer<AnomalyEntry>&  anomaly_buffer_;

    HookManager                hook_manager_;
    std::unique_ptr<ModelHandle> model_;

    std::atomic<InferenceState> state_{InferenceState::Idle};
    std::atomic<bool>           stop_flag_{false};
    std::atomic<float>          load_progress_{0.0f};
    std::atomic<double>         tokens_per_second_{0.0};

    mutable std::mutex          text_mutex_;
    std::string                 generated_text_;

    mutable std::mutex          error_mutex_;
    std::string                 error_message_;

    mutable std::mutex          token_labels_mutex_;
    std::vector<std::string>    token_labels_;

    std::thread                 worker_thread_;

    std::function<void()>       refresh_callback_;
    std::mutex                  refresh_mutex_;

    void notify_refresh();
};

} // namespace neuralscope
