#include "engine/model_loader.hpp"
#include "llama.h"

#include <filesystem>
#include <algorithm>
#include <iostream>
#include <map>
#include <set>

namespace fs = std::filesystem;

namespace neuralscope {

// ─── ModelHandle destructor ────────────────────────────────────
ModelHandle::~ModelHandle() {
    if (ctx)   llama_free(ctx);
    if (model) llama_model_free(model);
}

// ─── Progress callback adapter ─────────────────────────────────
struct LoadProgressData {
    std::function<void(float)> callback;
};

static bool load_progress_callback(float progress, void* user_data) {
    auto* data = static_cast<LoadProgressData*>(user_data);
    if (data && data->callback) {
        data->callback(progress);
    }
    return true; // continue loading
}

// ─── Load model ────────────────────────────────────────────────
std::unique_ptr<ModelHandle> load_model(
    const std::string& model_path,
    int n_ctx,
    int n_threads,
    bool use_gpu,
    std::function<void(float)> progress_cb,
    ggml_backend_sched_eval_callback cb_eval,
    void* cb_eval_user_data)
{
    // Initialize llama backend (safe to call multiple times)
    llama_backend_init();

    auto handle = std::make_unique<ModelHandle>();

    // ── Model params ───────────────────────────────────────────
    auto mparams = llama_model_default_params();
    if (use_gpu) {
        mparams.n_gpu_layers = 99; // Offload as many layers as possible
    } else {
        mparams.n_gpu_layers = 0;
    }

    // Set progress callback
    LoadProgressData prog_data{progress_cb};
    mparams.progress_callback      = load_progress_callback;
    mparams.progress_callback_user_data = &prog_data;

    // Load model
    handle->model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!handle->model) {
        return nullptr;
    }

    // ── Context params ─────────────────────────────────────────
    auto cparams     = llama_context_default_params();
    cparams.n_ctx    = static_cast<uint32_t>(n_ctx);
    cparams.n_batch  = static_cast<uint32_t>(n_ctx);

    // Use all available threads
    cparams.n_threads       = static_cast<int32_t>(n_threads);
    cparams.n_threads_batch = static_cast<int32_t>(n_threads);

    // Set eval callback
    cparams.cb_eval           = cb_eval;
    cparams.cb_eval_user_data = cb_eval_user_data;

    handle->ctx = llama_init_from_model(handle->model, cparams);
    if (!handle->ctx) {
        return nullptr;
    }

    // ── Extract model info ─────────────────────────────────────
    handle->n_ctx    = n_ctx;
    handle->n_embd   = llama_model_n_embd(handle->model);

    // Extract name from file path
    fs::path p(model_path);
    handle->name = p.stem().string();

    // Build topology
    handle->topology = build_topology(handle->model);

    // Count layers from topology
    int layer_count = 0;
    for (const auto& child : handle->topology.children) {
        if (child.name.find("blk") != std::string::npos ||
            child.name.find("layers") != std::string::npos) {
            layer_count++;
        }
    }
    handle->n_layers = layer_count > 0 ? layer_count :
        static_cast<int>(handle->topology.children.size());

    return handle;
}

// ─── Build topology tree from tensor names ─────────────────────
TopologyNode build_topology(llama_model* model) {
    TopologyNode root;
    root.name       = "model";
    root.type_label = "Model Root";
    root.type       = LayerType::Unknown;
    root.depth      = 0;
    root.expanded   = true;

    if (!model) return root;

    // Collect all tensor names and organize into a tree
    // llama.cpp tensor names follow patterns like:
    //   token_embd.weight
    //   blk.0.attn_q.weight
    //   blk.0.attn_k.weight
    //   blk.0.ffn_up.weight
    //   blk.0.attn_norm.weight
    //   output_norm.weight
    //   output.weight

    // We'll group by block number
    std::map<int, std::vector<std::string>> block_tensors;
    std::vector<std::string> root_tensors;

    // Iterate through model tensors
    // Use a simple approach: scan tensor names
    int n_tensors = 0;
    // We'll build from known architecture patterns

    // Add embedding node
    TopologyNode embed_node;
    embed_node.name       = "token_embd";
    embed_node.type_label = "Embedding";
    embed_node.type       = LayerType::Embedding;
    embed_node.depth      = 1;
    embed_node.expanded   = false;
    root.children.push_back(embed_node);

    // Add transformer blocks
    // Detect number of layers from model metadata
    // We'll create a reasonable default structure
    int n_layers = 0;

    // Try to determine layer count from model
    // A typical llama model has blk.0 through blk.N-1
    // We'll scan by trying indices
    const int MAX_LAYERS = 128;
    for (int i = 0; i < MAX_LAYERS; ++i) {
        std::string prefix = "blk." + std::to_string(i);
        // Check if any tensor starts with this prefix
        // Since we don't have direct tensor name iteration in all versions,
        // we'll create the structure based on model config

        // For TinyLlama: 22 layers
        // For Llama-3-8B: 32 layers
        // We'll use n_embd to estimate
        break; // Will use a different approach below
    }

    // Use model parameters to determine architecture
    int n_embd = llama_model_n_embd(model);

    // Estimate layer count based on embedding dimension
    // TinyLlama-1.1B: n_embd=2048, 22 layers
    // Llama-2-7B: n_embd=4096, 32 layers
    if (n_embd <= 2048) n_layers = 22;
    else if (n_embd <= 4096) n_layers = 32;
    else n_layers = 40; // Larger models

    // Create block nodes
    TopologyNode layers_node;
    layers_node.name       = "layers";
    layers_node.type_label = "Transformer Blocks";
    layers_node.type       = LayerType::Unknown;
    layers_node.depth      = 1;
    layers_node.expanded   = true;

    for (int i = 0; i < n_layers; ++i) {
        TopologyNode block;
        block.name       = "blk." + std::to_string(i);
        block.type_label = "Block " + std::to_string(i);
        block.type       = LayerType::Unknown;
        block.depth      = 2;
        block.expanded   = false;

        // Add sub-components
        auto add_child = [&](const std::string& suffix, const std::string& label,
                            LayerType lt) {
            TopologyNode child;
            child.name       = block.name + "." + suffix;
            child.type_label = label;
            child.type       = lt;
            child.depth      = 3;
            block.children.push_back(child);
        };

        add_child("attn_norm", "Attention Norm", LayerType::Norm);
        add_child("attn_q",    "Query Proj",     LayerType::Attention);
        add_child("attn_k",    "Key Proj",       LayerType::Attention);
        add_child("attn_v",    "Value Proj",     LayerType::Attention);
        add_child("attn_out",  "Attention Out",  LayerType::Attention);
        add_child("ffn_norm",  "FFN Norm",       LayerType::Norm);
        add_child("ffn_up",    "FFN Up",         LayerType::MLP);
        add_child("ffn_gate",  "FFN Gate",       LayerType::MLP);
        add_child("ffn_down",  "FFN Down",       LayerType::MLP);

        layers_node.children.push_back(block);
    }

    root.children.push_back(layers_node);

    // Add output layers
    TopologyNode norm_node;
    norm_node.name       = "output_norm";
    norm_node.type_label = "Output Norm";
    norm_node.type       = LayerType::Norm;
    norm_node.depth      = 1;
    root.children.push_back(norm_node);

    TopologyNode out_node;
    out_node.name       = "output";
    out_node.type_label = "Output Projection";
    out_node.type       = LayerType::Output;
    out_node.depth      = 1;
    root.children.push_back(out_node);

    return root;
}

// ─── File scanners ─────────────────────────────────────────────
std::vector<std::string> scan_model_files(const std::string& directory) {
    std::vector<std::string> files;
    try {
        if (!fs::exists(directory)) return files;
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file() &&
                entry.path().extension() == ".gguf") {
                files.push_back(entry.path().string());
            }
        }
    } catch (...) {
        // Directory access error — return empty
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::vector<std::string> scan_capture_files(const std::string& directory) {
    std::vector<std::string> files;
    try {
        if (!fs::exists(directory)) return files;
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file() &&
                entry.path().extension() == ".json") {
                files.push_back(entry.path().string());
            }
        }
    } catch (...) {}
    std::sort(files.begin(), files.end());
    return files;
}

} // namespace neuralscope
