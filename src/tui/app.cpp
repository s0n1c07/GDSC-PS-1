#include "tui/app.hpp"
#include "engine/model_loader.hpp"
#include "io/exporter.hpp"
#include "io/importer.hpp"
#include "core/utils.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

#include <filesystem>
#include <thread>

namespace fs = std::filesystem;
using namespace ftxui;

namespace neuralscope {

App::App()
    : screen_(ScreenInteractive::Fullscreen())
    , snapshot_buffer_(256)
    , anomaly_buffer_(100)
    , packet_stream_(snapshot_buffer_)
    , anomaly_ledger_(anomaly_buffer_)
{
    debug_log("App::App() constructor starting");
    screen_.TrackMouse(true);
    debug_log("App::App() constructor finished");
}

App::~App() {
    if (engine_) {
        engine_->stop();
    }
}

void App::run() {
    debug_log("App::run() started");

    // Start with the startup menu
    debug_log("App::run(): building startup screen");
    auto startup = build_startup_screen();
    debug_log("App::run(): building dashboard");
    auto dashboard = build_dashboard();

    // Top-level container that switches between startup and dashboard
    // Wrap in a renderer that picks the active screen
    auto main_component = Container::Tab({
        startup,
        dashboard
    }, &screen_index_);

    auto app_component = Renderer(main_component, [&, this] {
        debug_log("app_component render lambda triggered");
        if (!in_dashboard_) {
            debug_log("app_component: rendering startup");
            auto elem = startup->Render();
            debug_log("app_component: startup rendering complete");
            return elem;
        } else {
            debug_log("app_component: rendering dashboard");
            auto elem = dashboard->Render();
            debug_log("app_component: dashboard rendering complete");
            return elem;
        }
    });

    // Catch global keys
    debug_log("App::run(): setting up CatchEvent");
    app_component = CatchEvent(app_component, [&, this](Event event) -> bool {
        // Q to quit (only in dashboard)
        if (in_dashboard_ && event == Event::Character('q')) {
            screen_.Exit();
            return true;
        }
        // Escape to go back to startup
        if (in_dashboard_ && event == Event::Escape) {
            if (engine_) engine_->stop();
            in_dashboard_ = false;
            return true;
        }
        return false;
    });

    debug_log("App::run(): entering screen_.Loop(app_component)");
    screen_.Loop(app_component);
    debug_log("App::run(): screen_.Loop returned");
}

// ─── Startup screen ───────────────────────────────────────────
Component App::build_startup_screen() {
    // Scan for available models and captures
    std::string exe_dir = ".";
    auto model_files   = scan_model_files("models");
    auto capture_files = scan_capture_files("captures");

    startup_menu_.set_model_files(model_files);
    startup_menu_.set_capture_files(capture_files);

    startup_menu_.set_on_launch([this](const AppConfig& config) {
        on_launch(config);
    });

    return startup_menu_.component();
}

// ─── Dashboard ─────────────────────────────────────────────────
Component App::build_dashboard() {
    // Panel components
    auto topology_comp  = topology_tree_.component();
    auto stream_comp    = packet_stream_.component();
    auto heatmap_comp   = attention_heatmap_.component();
    auto metrics_comp   = metrics_panel_.component();
    auto anomaly_comp   = anomaly_ledger_.component();

    // Prompt input
    InputOption prompt_opt;
    prompt_opt.on_enter = [this] {
        if (engine_ && !prompt_text_.empty() &&
            (engine_->get_state() == InferenceState::Ready ||
             engine_->get_state() == InferenceState::Finished)) {
            std::string prompt = prompt_text_;
            prompt_text_.clear();
            engine_->run_inference(prompt);
        }
    };
    auto prompt_input = Input(&prompt_text_, "Type prompt and press Enter...", prompt_opt);

    // Export button functionality embedded in key handler

    auto row1_container = Container::Horizontal({
        topology_comp,
        stream_comp,
    });

    auto row3_container = Container::Horizontal({
        metrics_comp,
        anomaly_comp,
    });

    auto panels = Container::Vertical({
        row1_container,
        heatmap_comp,
        row3_container,
        prompt_input,
    });

    // Main dashboard renderer
    return CatchEvent(
        Renderer(panels, [this, topology_comp, stream_comp, heatmap_comp, metrics_comp, anomaly_comp, prompt_input] {
            // Refresh data from engine
            refresh_panels();

            // Panel border colors: active = cyan, inactive = gray
            auto border_for = [&](Component comp) {
                return comp->Focused()
                    ? color(Color::Cyan)
                    : color(Color::GrayDark);
            };

            // Build state string
            std::string state_str;
            if (engine_) {
                switch (engine_->get_state()) {
                    case InferenceState::Idle:     state_str = "Idle"; break;
                    case InferenceState::Loading:  state_str = "Loading..."; break;
                    case InferenceState::Ready:    state_str = "Ready"; break;
                    case InferenceState::Running:  state_str = "Generating"; break;
                    case InferenceState::Finished: state_str = "Done"; break;
                    case InferenceState::Error:
                        state_str = "Error: " + engine_->get_error();
                        break;
                }
            } else {
                state_str = config_.replay_mode ? "Replay Mode" : "No Engine";
            }

            // Throughput
            double tps = engine_ ? engine_->get_tokens_per_second() : 0.0;

            // Top status bar
            auto status_bar = hbox({
                text(" [Tab]") | bold | color(Color::Cyan),
                text(": Focus  ") | dim,
                text("[Q]") | bold | color(Color::Cyan),
                text(": Quit  ") | dim,
                text("[P]") | bold | color(Color::Cyan),
                text(": Pause  ") | dim,
                text("[E]") | bold | color(Color::Cyan),
                text(": Export  ") | dim,
                text("[Esc]") | bold | color(Color::Cyan),
                text(": Menu  ") | dim,
                filler(),
                text(" " + state_str + " ") | bold |
                    color(state_str.find("Error") != std::string::npos
                        ? Color::Red : Color::Green),
                text(" | ") | dim,
                text(format_float(static_cast<float>(tps), 1) + " tok/s ") |
                    bold | color(Color::GreenLight),
            }) | borderStyled(HEAVY);

            // Row 1: Topology (35%) + Packet Stream (65%)
            auto row1 = hbox({
                // Panel 1: Topology
                vbox({
                    text(" 1. MODEL TOPOLOGY ") | bold |
                        color(Color::Cyan),
                    topology_comp->Render(),
                }) | size(WIDTH, EQUAL, 40) | border_for(topology_comp),
                // Panel 2: Packet Stream
                vbox({
                    text(" 2. LIVE PACKET STREAM ") | bold |
                        color(Color::Cyan),
                    stream_comp->Render(),
                }) | flex | border_for(stream_comp),
            }) | size(HEIGHT, EQUAL, 12);

            // Row 2: Attention Heatmap (full width)
            auto row2 = vbox({
                text(" 3. ATTENTION MATRIX VISUALIZER ") | bold |
                    color(Color::Cyan),
                heatmap_comp->Render(),
            }) | borderStyled(ROUNDED) | border_for(heatmap_comp) |
                 size(HEIGHT, EQUAL, 12);

            // Row 3: Metrics (50%) + Anomaly Ledger (50%)
            auto row3 = hbox({
                // Panel 4: Metrics
                vbox({
                    text(" 4. RUNTIME METRICS ") | bold |
                        color(Color::Cyan),
                    metrics_comp->Render(),
                }) | flex | border_for(metrics_comp),
                // Panel 5: Anomaly Ledger
                vbox({
                    text(" 5. ANOMALY LEDGER ") | bold |
                        color(Color::Cyan),
                    anomaly_comp->Render(),
                }) | flex | border_for(anomaly_comp),
            });

            // Generated text display
            std::string gen_text = engine_
                ? engine_->get_generated_text() : "";
            auto gen_display = text("");
            if (!gen_text.empty()) {
                std::string display = gen_text;
                if (display.size() > 80) {
                    display = "..." + display.substr(display.size() - 77);
                }
                gen_display = text(" Output: " + display) |
                    color(Color::GreenLight) | dim;
            }

            // Prompt bar
            bool is_replay_mode = (engine_ == nullptr);
            auto prompt_bar = hbox({
                text(is_replay_mode ? " [REPLAY MODE] " : " > ") | bold | color(is_replay_mode ? Color::Red : Color::Cyan),
                (is_replay_mode ? text("Prompt disabled. Please load a .gguf model to run live inference.") | dim | flex : prompt_input->Render() | flex),
            }) | borderStyled(ROUNDED) |
                 ((prompt_input->Focused() && !is_replay_mode)
                    ? color(Color::Cyan)
                    : color(Color::GrayDark));

            return vbox({
                status_bar,
                row1,
                row2,
                row3,
                gen_display,
                prompt_bar,
            });
        }),
        [&, prompt_input](Event event) -> bool {
            // Do not trigger global hotkeys if the user is typing in the prompt
            if (prompt_input->Focused()) {
                return false;
            }

            // P: toggle pause on packet stream
            if (event == Event::Character('p') ||
                event == Event::Character('P')) {
                packet_stream_.toggle_pause();
                return true;
            }
            // E: export
            if (event == Event::Character('e') ||
                event == Event::Character('E')) {
                auto snapshots = snapshot_buffer_.snapshot();
                std::string model_name = engine_ && engine_->get_model()
                    ? engine_->get_model()->name : "unknown";
                export_capture("captures", model_name, snapshots,
                              config_.buffer_size);
                // Add info anomaly about export
                anomaly_buffer_.push({
                    now_timestamp(),
                    Severity::Info,
                    "Capture exported to captures/ directory"
                });
                return true;
            }
            return false;
        }
    );
}

// ─── Launch handler ────────────────────────────────────────────
void App::on_launch(const AppConfig& config) {
    config_ = config;
    snapshot_buffer_.resize(config.buffer_size);
    anomaly_buffer_.resize(100);

    if (config.replay_mode) {
        // Load capture file
        auto snapshots = import_capture(config.capture_path);
        snapshot_buffer_.load(snapshots);
        in_dashboard_ = true;
        screen_index_ = 1;
        return;
    }

    // Create inference engine
    engine_ = std::make_unique<InferenceEngine>(
        snapshot_buffer_, anomaly_buffer_);

    // Set refresh callback to wake up FTXUI
    engine_->set_refresh_callback([this] {
        screen_.PostEvent(Event::Custom);
    });

    // Wire up topology tree selection to attention visualization
    topology_tree_.set_on_select([this](const std::string& layer_name) {
        if (engine_) {
            // Add a filter so we specifically capture and focus on this layer
            // For now, we just log it, but we should tell hook_manager to prioritize it
            debug_log("Selected layer in topology: " + layer_name);
            attention_heatmap_.set_data({}, 0, 0); // clear until new data arrives
        }
    });

    // Start model loading
    engine_->load_model_async(config.model_path, config.n_ctx,
                               config.n_threads, config.use_gpu);

    in_dashboard_ = true;
    screen_index_ = 1;
}

// ─── Panel refresh ─────────────────────────────────────────────
void App::refresh_panels() {
    if (!engine_) return;

    // Update topology with active layer
    auto* model = engine_->get_model();
    if (model) {
        // Only set topology once
        static bool topology_set = false;
        if (!topology_set) {
            topology_tree_.set_topology(model->topology);
            topology_set = true;
        }

        // Update active layer marker
        topology_tree_.set_active_layer(
            engine_->get_hook_manager().get_active_layer());
    }

    // Update metrics panel with latest snapshot
    if (!snapshot_buffer_.empty()) {
        try {
            auto latest = snapshot_buffer_.latest();
            metrics_panel_.set_snapshot(latest);
            metrics_panel_.add_latency_sample(
                static_cast<float>(latest.latency_ms));
            metrics_panel_.add_stats_sample(
                latest.act_min, latest.act_mean, latest.act_max);
            metrics_panel_.set_throughput(
                engine_->get_tokens_per_second());

            // Update attention heatmap if attention data available
            if (!latest.attention_weights.empty()) {
                attention_heatmap_.set_data(
                    latest.attention_weights,
                    latest.attn_size,
                    latest.num_heads);
            }
        } catch (...) {
            // Buffer might be empty between checks
        }
    }
}

} // namespace neuralscope
