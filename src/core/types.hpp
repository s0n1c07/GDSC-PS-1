#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <mutex>

namespace neuralscope {

// ─── Layer classification ──────────────────────────────────────
enum class LayerType {
    Embedding,
    Attention,
    MLP,
    Norm,
    Output,
    Unknown
};

inline const char* layer_type_str(LayerType t) {
    switch (t) {
        case LayerType::Embedding: return "Embedding";
        case LayerType::Attention: return "Attention";
        case LayerType::MLP:       return "MLP";
        case LayerType::Norm:      return "LayerNorm";
        case LayerType::Output:    return "Output";
        default:                   return "Unknown";
    }
}

// Classify a ggml tensor name into a LayerType
inline LayerType classify_tensor(const std::string& name) {
    // Embedding layers
    if (name.find("embd") != std::string::npos ||
        name.find("embed") != std::string::npos ||
        name.find("tok_embd") != std::string::npos) {
        return LayerType::Embedding;
    }
    // Attention layers
    if (name.find("attn") != std::string::npos ||
        name.find("kqv") != std::string::npos ||
        name.find("attn_q") != std::string::npos ||
        name.find("attn_k") != std::string::npos ||
        name.find("attn_v") != std::string::npos) {
        return LayerType::Attention;
    }
    // MLP / Feed-forward layers
    if (name.find("ffn") != std::string::npos ||
        name.find("mlp") != std::string::npos ||
        name.find("ff_") != std::string::npos) {
        return LayerType::MLP;
    }
    // Normalization layers
    if (name.find("norm") != std::string::npos ||
        name.find("ln_") != std::string::npos) {
        return LayerType::Norm;
    }
    // Output projection
    if (name.find("output") != std::string::npos ||
        name.find("lm_head") != std::string::npos) {
        return LayerType::Output;
    }
    return LayerType::Unknown;
}

// ─── Anomaly severity ──────────────────────────────────────────
enum class Severity {
    Info,     // ℹ  (blue/dim)
    Warning,  // ⚠  (yellow)
    Critical  // ✖  (red)
};

inline const char* severity_icon(Severity s) {
    switch (s) {
        case Severity::Info:     return "\xe2\x84\xb9";  // ℹ
        case Severity::Warning:  return "\xe2\x9a\xa0";  // ⚠
        case Severity::Critical: return "\xe2\x9c\x96";  // ✖
        default:                 return "?";
    }
}

// ─── Anomaly entry ─────────────────────────────────────────────
struct AnomalyEntry {
    std::string timestamp;
    Severity    severity;
    std::string message;
};

// ─── Layer snapshot (one per tensor evaluation) ────────────────
struct LayerSnapshot {
    int64_t     id = 0;                // Auto-incrementing
    std::string timestamp;             // HH:MM:SS.mmm
    std::string layer_name;            // e.g., "blk.1.attn_q"
    LayerType   layer_type = LayerType::Unknown;
    std::string compute_device;        // "CPU" or "CUDA [GPU 0]"
    double      latency_ms = 0.0;      // Execution time in ms
    std::string tensor_shape;          // e.g., "[1, 32, 4096]"
    std::string dtype;                 // e.g., "q4_K", "f16", "f32"
    int64_t     n_elements = 0;        // Total number of elements

    // Activation statistics
    float       sparsity   = 0.0f;     // 0.0 - 1.0
    float       act_min    = 0.0f;
    float       act_max    = 0.0f;
    float       act_mean   = 0.0f;
    float       act_std    = 0.0f;

    // Attention weights (only for attention layers, flattened NxN)
    std::vector<float> attention_weights;
    int         attn_size  = 0;        // N for NxN attention matrix
    int         num_heads  = 0;

    // Detected anomalies for this tensor
    std::vector<AnomalyEntry> anomalies;
};

// ─── Model topology tree node ──────────────────────────────────
struct TopologyNode {
    std::string name;
    std::string type_label;  // Display label
    LayerType   type = LayerType::Unknown;
    int         depth = 0;
    bool        expanded = false;
    bool        selected = false;
    bool        active = false;        // Currently being processed
    std::vector<TopologyNode> children;
};

// ─── Application configuration ─────────────────────────────────
struct AppConfig {
    std::string model_path;
    std::string capture_path;          // For replay mode
    size_t      buffer_size = 256;
    int         n_ctx       = 512;     // Context size
    int         n_threads   = 4;       // Inference threads
    bool        replay_mode = false;   // True when loading a capture
    bool        use_gpu     = false;   // Use GPU acceleration if available
};

} // namespace neuralscope
