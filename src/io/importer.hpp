#pragma once

#include "core/types.hpp"
#include <string>
#include <vector>

namespace neuralscope {

/// Import snapshots from a previously saved JSON capture file.
/// Returns an empty vector on failure.
std::vector<LayerSnapshot> import_capture(const std::string& filepath);

/// Get metadata from a capture file without loading all snapshots.
struct CaptureMetadata {
    std::string version;
    std::string model;
    std::string timestamp;
    size_t      buffer_size = 0;
    size_t      n_snapshots = 0;
    bool        valid       = false;
};

CaptureMetadata peek_capture(const std::string& filepath);

} // namespace neuralscope
