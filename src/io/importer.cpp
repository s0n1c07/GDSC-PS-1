#include "io/importer.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>

using json = nlohmann::json;

namespace neuralscope {

static LayerType parse_layer_type(const std::string& type_str) {
    if (type_str == "Embedding")  return LayerType::Embedding;
    if (type_str == "Attention")  return LayerType::Attention;
    if (type_str == "MLP")        return LayerType::MLP;
    if (type_str == "LayerNorm")  return LayerType::Norm;
    if (type_str == "Output")     return LayerType::Output;
    return LayerType::Unknown;
}

static Severity parse_severity(int sev) {
    switch (sev) {
        case 0:  return Severity::Info;
        case 1:  return Severity::Warning;
        case 2:  return Severity::Critical;
        default: return Severity::Info;
    }
}

std::vector<LayerSnapshot> import_capture(const std::string& filepath) {
    std::vector<LayerSnapshot> result;

    try {
        std::ifstream ifs(filepath);
        if (!ifs.is_open()) {
            return result;
        }

        json root = json::parse(ifs);

        if (!root.contains("snapshots") || !root["snapshots"].is_array()) {
            return result;
        }

        for (const auto& j : root["snapshots"]) {
            LayerSnapshot snap;

            snap.id             = j.value("id", (int64_t)0);
            snap.timestamp      = j.value("timestamp", "");
            snap.layer_name     = j.value("layer_name", "");
            snap.layer_type     = parse_layer_type(j.value("layer_type", ""));
            snap.compute_device = j.value("compute_device", "CPU");
            snap.latency_ms     = j.value("latency_ms", 0.0);
            snap.tensor_shape   = j.value("tensor_shape", "");
            snap.dtype          = j.value("dtype", "");
            snap.op_name        = j.value("op_name", "UNKNOWN");
            snap.n_elements     = j.value("n_elements", (int64_t)0);
            snap.sparsity       = j.value("sparsity", 0.0f);
            snap.act_min        = j.value("act_min", 0.0f);
            snap.act_max        = j.value("act_max", 0.0f);
            snap.act_mean       = j.value("act_mean", 0.0f);
            snap.act_std        = j.value("act_std", 0.0f);
            snap.attn_size      = j.value("attn_size", 0);
            snap.num_heads      = j.value("num_heads", 0);

            // Attention weights
            if (j.contains("attention_weights") &&
                j["attention_weights"].is_array()) {
                snap.attention_weights =
                    j["attention_weights"].get<std::vector<float>>();
            }

            // Anomalies
            if (j.contains("anomalies") && j["anomalies"].is_array()) {
                for (const auto& aj : j["anomalies"]) {
                    AnomalyEntry entry;
                    entry.timestamp = aj.value("timestamp", "");
                    entry.severity  = parse_severity(aj.value("severity", 0));
                    entry.message   = aj.value("message", "");
                    snap.anomalies.push_back(entry);
                }
            }

            result.push_back(std::move(snap));
        }

    } catch (const std::exception& e) {
        // Parse error — return whatever we have
        std::cerr << "Import error: " << e.what() << std::endl;
    }

    return result;
}

CaptureMetadata peek_capture(const std::string& filepath) {
    CaptureMetadata meta;

    try {
        std::ifstream ifs(filepath);
        if (!ifs.is_open()) return meta;

        json root = json::parse(ifs);

        meta.version     = root.value("version", "");
        meta.model       = root.value("model", "");
        meta.timestamp   = root.value("timestamp", "");
        meta.buffer_size = root.value("buffer_size", (size_t)0);
        meta.n_snapshots = root.value("n_snapshots", (size_t)0);
        meta.valid       = true;

    } catch (...) {
        meta.valid = false;
    }

    return meta;
}

} // namespace neuralscope
