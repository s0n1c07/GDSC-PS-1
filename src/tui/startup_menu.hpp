#pragma once

#include "core/types.hpp"
#include <ftxui/component/component.hpp>
#include <ftxui/dom/elements.hpp>
#include <functional>
#include <string>
#include <vector>

namespace neuralscope {

/// Interactive startup/configuration screen.
/// Allows model selection, buffer size config, and capture loading.
class StartupMenu {
public:
    StartupMenu();

    /// Set the list of available GGUF model files.
    void set_model_files(const std::vector<std::string>& files);

    /// Set the list of available capture files.
    void set_capture_files(const std::vector<std::string>& files);

    /// Set callback for when the user clicks "Launch".
    void set_on_launch(std::function<void(const AppConfig&)> cb);

    /// Get the FTXUI component.
    ftxui::Component component();

private:
    std::vector<std::string> model_files_;
    std::vector<std::string> capture_files_;

    // Model display names (basenames)
    std::vector<std::string> model_names_;
    std::vector<std::string> capture_names_;

    // Selection state
    int    selected_model_   = 0;
    int    selected_capture_ = -1;
    int    selected_buffer_  = 2;  // Index: 0=64, 1=128, 2=256, 3=512
    std::string model_path_input_;
    bool   use_capture_      = false;
    bool   use_gpu_          = false;

    std::function<void(const AppConfig&)> on_launch_;

    static constexpr size_t BUFFER_SIZES[] = {64, 128, 256, 512};
};

} // namespace neuralscope
