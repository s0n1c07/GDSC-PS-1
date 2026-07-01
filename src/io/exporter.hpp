#pragma once

#include "core/types.hpp"
#include <string>
#include <vector>

namespace neuralscope {

/// Export ring buffer snapshots to a JSON file in the given directory.
/// Returns the full path to the created file.
std::string export_capture(
    const std::string& directory,
    const std::string& model_name,
    const std::vector<LayerSnapshot>& snapshots,
    size_t buffer_size
);

/// Export snapshots to a CSV file (metrics only, no tensor data).
std::string export_capture_csv(
    const std::string& directory,
    const std::string& model_name,
    const std::vector<LayerSnapshot>& snapshots
);

} // namespace neuralscope
