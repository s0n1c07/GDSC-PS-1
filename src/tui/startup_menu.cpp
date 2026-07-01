#include "tui/startup_menu.hpp"
#include "engine/model_loader.hpp"
#include "core/utils.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/dom/elements.hpp>

#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;
using namespace ftxui;

namespace neuralscope {

StartupMenu::StartupMenu() {}

void StartupMenu::set_model_files(const std::vector<std::string>& files) {
    model_files_ = files;
    model_names_.clear();
    for (const auto& f : files) {
        model_names_.push_back(fs::path(f).filename().string());
    }
    if (model_names_.empty()) {
        model_names_.push_back("(no .gguf files found in models/)");
    }
}

void StartupMenu::set_capture_files(const std::vector<std::string>& files) {
    capture_files_ = files;
    capture_names_.clear();
    for (const auto& f : files) {
        capture_names_.push_back(fs::path(f).filename().string());
    }
}

void StartupMenu::set_on_launch(std::function<void(const AppConfig&)> cb) {
    on_launch_ = std::move(cb);
}

Component StartupMenu::component() {
    std::ostringstream oss;
    oss << "[DEBUG-ADDR] StartupMenu::component() - this: " << this << ", model_names_: " << &model_names_;
    debug_log(oss.str());

    // Buffer size radio buttons
    static std::vector<std::string> buffer_options = {"64", "128", "256", "512"};

    auto model_menu = Menu(&model_names_, &selected_model_);

    auto buffer_radio = Radiobox(&buffer_options, &selected_buffer_);

    auto model_path_input = Input(&model_path_input_, "Or paste model path here...");

    auto launch_button = Button(" [ Launch Inspector ] ", [this] {
        if (!on_launch_) return;

        AppConfig config;
        config.buffer_size = BUFFER_SIZES[selected_buffer_];
        config.replay_mode = false;
        config.use_gpu     = use_gpu_;

        // Use typed path if provided, otherwise use selected file
        if (!model_path_input_.empty()) {
            config.model_path = model_path_input_;
        } else if (selected_model_ >= 0 &&
                   selected_model_ < static_cast<int>(model_files_.size())) {
            config.model_path = model_files_[selected_model_];
        }

        on_launch_(config);
    }, ButtonOption::Animated(Color::Cyan));

    auto replay_button = Button(" [ Load Capture ] ", [this] {
        if (!on_launch_) return;
        if (selected_capture_ < 0 ||
            selected_capture_ >= static_cast<int>(capture_files_.size())) {
            return;
        }

        AppConfig config;
        config.capture_path = capture_files_[selected_capture_];
        config.buffer_size  = BUFFER_SIZES[selected_buffer_];
        config.replay_mode  = true;

        on_launch_(config);
    }, ButtonOption::Animated(Color::Yellow));

    // Capture file selector (only if captures exist)
    Component capture_menu;
    if (!capture_names_.empty()) {
        capture_menu = Menu(&capture_names_, &selected_capture_);
    }

    auto gpu_checkbox = Checkbox("Use GPU Acceleration", &use_gpu_);

    // Compose everything into a vertical layout
    auto container = Container::Vertical({
        model_menu,
        model_path_input,
        gpu_checkbox,
        buffer_radio,
        launch_button,
    });

    if (capture_menu) {
        container->Add(capture_menu);
        container->Add(replay_button);
    }

    return Renderer(container, [this, model_menu, buffer_radio, model_path_input, launch_button, capture_menu, replay_button, gpu_checkbox] {
        debug_log("StartupMenu::render() starting");
        std::ostringstream oss;
        oss << "[DEBUG-ADDR] StartupMenu::render() lambda - this: " << this << ", model_names_: " << &model_names_;
        debug_log(oss.str());
        std::vector<Element> content;

        // Header / Banner
        debug_log("StartupMenu::render(): rendering Banner");
        content.push_back(separator());
        content.push_back(
            text("   NEURALSCOPE v1.0") |
            bold | color(Color::Cyan) | center);
        content.push_back(
            text("  Non-Invasive LLM Inspection TUI") |
            dim | center);
        content.push_back(separator());

        // Model selection section
        debug_log("StartupMenu::render(): rendering MODEL SELECTION text");
        content.push_back(text(""));
        content.push_back(
            text("  MODEL SELECTION") | bold | color(Color::Yellow));
        content.push_back(text(""));

        content.push_back(hbox({
            text("  Available Models: ") | bold,
        }));
        
        debug_log("StartupMenu::render(): rendering model_menu");
        if (model_menu) {
            content.push_back(
                model_menu->Render() | border | size(HEIGHT, LESS_THAN, 6) |
                size(WIDTH, LESS_THAN, 60));
        } else {
            debug_log("WARNING: model_menu is null!");
        }
        
        content.push_back(text(""));
        
        debug_log("StartupMenu::render(): rendering model_path_input");
        if (model_path_input) {
            content.push_back(hbox({
                text("  Manual Path: ") | bold,
                model_path_input->Render() | flex | borderStyled(ROUNDED),
            }));
        } else {
            debug_log("WARNING: model_path_input is null!");
        }

        // Buffer size
        debug_log("StartupMenu::render(): rendering BUFFER SIZE text");
        content.push_back(text(""));
        content.push_back(
            text("  BUFFER SIZE") | bold | color(Color::Yellow));
            
        debug_log("StartupMenu::render(): rendering buffer_radio");
        if (buffer_radio) {
            content.push_back(
                vbox({
                    text("Buffer Size:") | bold,
                    buffer_radio->Render(),
                    separatorEmpty(),
                    gpu_checkbox->Render(),
                    separatorEmpty(),
                    launch_button->Render() | center,
                }) | border);
        } else {
            debug_log("WARNING: buffer_radio is null!");
        }

        // System info
        debug_log("StartupMenu::render(): rendering SYSTEM INFO text");
        content.push_back(text(""));
        content.push_back(
            text("  SYSTEM INFO") | bold | color(Color::Yellow));
        content.push_back(hbox({
            text("  Device: ") | dim,
            text("CPU") | color(Color::Green),
            text("  |  OS: ") | dim,
            text("Windows") | color(Color::Green),
        }));

        content.push_back(text(""));
        content.push_back(separator());

        // Launch button
        debug_log("StartupMenu::render(): rendering launch_button");
        content.push_back(text(""));
        if (launch_button) {
            content.push_back(launch_button->Render() | center);
        } else {
            debug_log("WARNING: launch_button is null!");
        }

        // Capture replay section
        debug_log("StartupMenu::render(): checking capture_menu");
        if (capture_menu) {
            debug_log("StartupMenu::render(): rendering LOAD SAVED CAPTURE section");
            content.push_back(text(""));
            content.push_back(separator());
            content.push_back(
                text("  LOAD SAVED CAPTURE") | bold | color(Color::Yellow));
            content.push_back(
                capture_menu->Render() | border | size(HEIGHT, LESS_THAN, 5));
                
            debug_log("StartupMenu::render(): rendering replay_button");
            if (replay_button) {
                content.push_back(replay_button->Render() | center);
            } else {
                debug_log("WARNING: replay_button is null!");
            }
        }

        content.push_back(text(""));

        debug_log("StartupMenu::render() complete");
        return vbox(std::move(content)) |
               borderStyled(DOUBLE) | color(Color::Cyan) |
               size(WIDTH, LESS_THAN, 70) | center;
    });
}

} // namespace neuralscope
