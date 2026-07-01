#include "io/exporter.hpp"
#include "core/utils.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace neuralscope {

static std::string generate_filename(const std::string& prefix,
                                      const std::string& ext)
{
    auto now  = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);

    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &time);
#else
    localtime_r(&time, &tm_buf);
#endif

    std::ostringstream oss;
    oss << prefix << "_"
        << std::put_time(&tm_buf, "%Y%m%d_%H%M%S")
        << "." << ext;
    return oss.str();
}

static json snapshot_to_json(const LayerSnapshot& snap) {
    json j;
    j["id"]             = snap.id;
    j["timestamp"]      = snap.timestamp;
    j["layer_name"]     = snap.layer_name;
    j["layer_type"]     = layer_type_str(snap.layer_type);
    j["compute_device"] = snap.compute_device;
    j["latency_ms"]     = snap.latency_ms;
    j["tensor_shape"]   = snap.tensor_shape;
    j["dtype"]          = snap.dtype;
    j["n_elements"]     = snap.n_elements;
    j["sparsity"]       = snap.sparsity;
    j["act_min"]        = snap.act_min;
    j["act_max"]        = snap.act_max;
    j["act_mean"]       = snap.act_mean;
    j["act_std"]        = snap.act_std;
    j["attn_size"]      = snap.attn_size;
    j["num_heads"]      = snap.num_heads;

    // Attention weights (can be large, only include if present)
    if (!snap.attention_weights.empty()) {
        j["attention_weights"] = snap.attention_weights;
    }

    // Anomalies
    json anomalies = json::array();
    for (const auto& a : snap.anomalies) {
        json aj;
        aj["timestamp"] = a.timestamp;
        aj["severity"]  = static_cast<int>(a.severity);
        aj["message"]   = a.message;
        anomalies.push_back(aj);
    }
    j["anomalies"] = anomalies;

    return j;
}

std::string export_capture(
    const std::string& directory,
    const std::string& model_name,
    const std::vector<LayerSnapshot>& snapshots,
    size_t buffer_size)
{
    // Ensure directory exists
    fs::create_directories(directory);

    std::string filename = generate_filename("capture", "json");
    std::string filepath = (fs::path(directory) / filename).string();

    json root;
    root["version"]     = "1.0";
    root["model"]       = model_name;
    root["timestamp"]   = now_iso_timestamp();
    root["buffer_size"] = buffer_size;
    root["n_snapshots"] = snapshots.size();

    json snap_array = json::array();
    for (const auto& snap : snapshots) {
        snap_array.push_back(snapshot_to_json(snap));
    }
    root["snapshots"] = snap_array;

    // Write to file
    std::ofstream ofs(filepath);
    if (ofs.is_open()) {
        ofs << root.dump(2); // Pretty print with indent=2
        ofs.close();
    }

    return filepath;
}

std::string export_capture_csv(
    const std::string& directory,
    const std::string& model_name,
    const std::vector<LayerSnapshot>& snapshots)
{
    fs::create_directories(directory);

    std::string filename = generate_filename("capture", "csv");
    std::string filepath = (fs::path(directory) / filename).string();

    std::ofstream ofs(filepath);
    if (!ofs.is_open()) return "";

    // CSV header
    ofs << "id,timestamp,layer_name,layer_type,device,latency_ms,"
        << "tensor_shape,dtype,n_elements,sparsity,"
        << "act_min,act_max,act_mean,act_std\n";

    for (const auto& snap : snapshots) {
        ofs << snap.id << ","
            << snap.timestamp << ","
            << snap.layer_name << ","
            << layer_type_str(snap.layer_type) << ","
            << snap.compute_device << ","
            << snap.latency_ms << ","
            << "\"" << snap.tensor_shape << "\","
            << snap.dtype << ","
            << snap.n_elements << ","
            << snap.sparsity << ","
            << snap.act_min << ","
            << snap.act_max << ","
            << snap.act_mean << ","
            << snap.act_std << "\n";
    }

    ofs.close();
    return filepath;
}

} // namespace neuralscope
