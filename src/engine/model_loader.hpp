#pragma once

#include "core/types.hpp"
#include "llama.h"

#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace neuralscope {

/// Holds a loaded model and its context.
struct ModelHandle {
    llama_model*    model   = nullptr;
    llama_context*  ctx     = nullptr;
    std::string     name;
    std::string     arch;
    int             n_layers = 0;
    int             n_heads  = 0;
    int             n_embd   = 0;
    int             n_vocab  = 0;
    int             n_ctx    = 0;

    /// Model topology tree for the TUI
    TopologyNode    topology;

    ~ModelHandle();
};

/// Load a GGUF model from disk.
/// Returns nullptr on failure.
/// progress_cb receives a value 0.0 - 1.0 during loading.
std::unique_ptr<ModelHandle> load_model(
    const std::string& model_path,
    int n_ctx = 512,
    int n_threads = 4,
    bool use_gpu = false,
    std::function<void(float)> progress_cb = nullptr,
    ggml_backend_sched_eval_callback cb_eval = nullptr,
    void* cb_eval_user_data = nullptr
);

/// Build a topology tree from the model's tensor names.
TopologyNode build_topology(llama_model* model);

/// Scan a directory for .gguf files.
std::vector<std::string> scan_model_files(const std::string& directory);

/// Scan a directory for .json capture files.
std::vector<std::string> scan_capture_files(const std::string& directory);

} // namespace neuralscope
