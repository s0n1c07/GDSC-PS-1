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
#include <chrono>

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
    // Auto-export if requested
    if (config_.auto_export && !snapshot_buffer_.empty()) {
        auto snapshots = snapshot_buffer_.snapshot();
        std::string model_name = engine_ && engine_->get_model()
            ? engine_->get_model()->name : "unknown";
        export_capture("captures", model_name, snapshots, config_.buffer_size);
    }
}

// ─── CLI-driven launch (skips startup menu) ───────────────────
void App::launch_with_config(const AppConfig& config,
                              const std::string& initial_prompt)
{
    cli_initial_prompt_ = initial_prompt;
    on_launch(config);
}

void App::run() {
    debug_log("App::run() started");

    debug_log("App::run(): building startup screen");
    auto startup = build_startup_screen();
    debug_log("App::run(): building dashboard");
    auto dashboard = build_dashboard();

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
        Renderer(panels, [this, topology_comp, stream_comp, heatmap_comp,
                          metrics_comp, anomaly_comp, prompt_input] {
            refresh_panels();

            auto border_for = [&](Component comp) {
                return borderStyled(ROUNDED) | (comp->Focused()
                    ? color(Color::Cyan)
                    : color(Color::GrayDark));
            };

            // Build state string
            std::string state_str;
            bool is_replay_mode = (engine_ == nullptr);
            if (is_replay_mode) {
                if (!replay_snapshots_.empty()) {
                    state_str = replay_paused_
                        ? "REPLAY [PAUSED]"
                        : "REPLAY [PLAYING]";
                } else {
                    state_str = "No Engine";
                }
            } else {
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
            }

            double tps = engine_ ? engine_->get_tokens_per_second() : 0.0;

            // Top Application Header
            auto header_bar = hbox({
                text(" NeuralScope ") | bold | bgcolor(Color::Blue) | color(Color::White),
                text(" | Non-Invasive Model Inspector") | dim,
                filler(),
                text(" " + state_str + " ") | bold |
                    color(state_str.find("Error") != std::string::npos
                        ? Color::Red
                        : (is_replay_mode ? Color::Yellow : Color::Green)),
                text(" | ") | dim,
                text(format_float(static_cast<float>(tps), 1) + " tok/s ") |
                    bold | color(Color::GreenLight),
            });

            // Row 1: Topology (35%) + Packet Stream (65%)
            auto row1 = hbox({
                window(text(" MODEL TOPOLOGY ") | bold | center, topology_comp->Render())
                    | size(WIDTH, EQUAL, 40) | border_for(topology_comp),
                window(text(" LIVE PACKET STREAM ") | bold | center, stream_comp->Render())
                    | flex | border_for(stream_comp),
            }) | size(HEIGHT, EQUAL, 12);

            // Row 2: Attention Heatmap (full width)
            auto row2 = window(text(" ATTENTION MATRIX VISUALIZER ") | bold | center, heatmap_comp->Render())
                | border_for(heatmap_comp) | size(HEIGHT, EQUAL, 12);

            // Row 3: Metrics (50%) + Anomaly Ledger (50%)
            auto row3 = hbox({
                window(text(" RUNTIME METRICS INSPECTOR ") | bold | center, metrics_comp->Render())
                    | flex | border_for(metrics_comp),
                window(text(" NUMERICAL ANOMALY LEDGER ") | bold | center, anomaly_comp->Render())
                    | flex | border_for(anomaly_comp),
            });

            // Generated text display
            std::string gen_text = engine_ ? engine_->get_generated_text() : "";
            auto gen_display = text("");
            if (!gen_text.empty()) {
                std::string display = gen_text;
                if (display.size() > 80) {
                    display = "..." + display.substr(display.size() - 77);
                }
                gen_display = text(" Output: " + display) |
                    color(Color::GreenLight) | dim;
            }

            // Prompt bar / Replay seek bar
            Element bottom_bar;
            if (is_replay_mode && !replay_snapshots_.empty()) {
                // ── Replay seek bar ────────────────────────────────
                size_t total = replay_snapshots_.size();
                size_t idx   = std::min(replay_index_, total - 1);
                int pct = (total > 1)
                    ? static_cast<int>((idx * 100) / (total - 1))
                    : 100;

                // Build a simple ASCII bar
                int bar_width = 40;
                int filled = (bar_width * pct) / 100;
                std::string bar = "[";
                for (int i = 0; i < bar_width; ++i)
                    bar += (i < filled) ? "=" : "-";
                bar += "]";

                std::string seek_text =
                    " REPLAY " + bar
                    + " " + std::to_string(idx + 1)
                    + "/" + std::to_string(total)
                    + "  Speed: " + std::to_string(replay_speed_ms_) + "ms ";

                bottom_bar = hbox({
                    text(replay_paused_ ? " [PAUSED] " : " [PLAYING] ")
                        | bold
                        | color(replay_paused_ ? Color::Red : Color::Green),
                    text(seek_text) | color(Color::Yellow),
                }) | borderStyled(ROUNDED) | color(Color::Yellow);
            } else {
                // ── Normal prompt bar ─────────────────────────────
                bottom_bar = hbox({
                    text(is_replay_mode ? " [REPLAY MODE] " : " > ")
                        | bold
                        | color(is_replay_mode ? Color::Red : Color::Cyan),
                    (is_replay_mode
                        ? text("Prompt disabled. Please load a .gguf model to run live inference.") | dim | flex
                        : prompt_input->Render() | flex),
                }) | borderStyled(ROUNDED)
                   | ((prompt_input->Focused() && !is_replay_mode)
                       ? color(Color::Cyan)
                       : color(Color::GrayDark));
            }

            // Bottom Hotkey Legend
            auto hotkey_legend = is_replay_mode
                ? hbox({
                    text(" [Tab]") | bold | color(Color::Cyan), text(": Focus  ") | dim,
                    text("[Q]") | bold | color(Color::Cyan), text(": Quit  ") | dim,
                    text("[Esc]") | bold | color(Color::Cyan), text(": Menu  ") | dim,
                    text("[Space]") | bold | color(Color::Yellow), text(": Play/Pause  ") | dim,
                    text("[n]") | bold | color(Color::Yellow), text(": Step  ") | dim,
                    text("[ ][ ]") | bold | color(Color::Yellow), text(": Speed") | dim,
                  }) | center
                : hbox({
                    text(" [Tab]") | bold | color(Color::Cyan), text(": Focus  ") | dim,
                    text("[Q]") | bold | color(Color::Cyan), text(": Quit  ") | dim,
                    text("[Esc]") | bold | color(Color::Cyan), text(": Menu  ") | dim,
                    text("[P]") | bold | color(Color::Yellow), text(": Pause  ") | dim,
                    text("[E]") | bold | color(Color::Yellow), text(": Export") | dim,
                  }) | center;

            return vbox({
                header_bar,
                row1,
                row2,
                row3,
                gen_display,
                bottom_bar,
                hotkey_legend,
            });
        }),
        [&, prompt_input, this](Event event) -> bool {
            bool is_replay_mode = (engine_ == nullptr);

            // ── Replay-mode keybindings ───────────────────────────
            if (is_replay_mode && !replay_snapshots_.empty()) {
                // Space: toggle play/pause
                if (event == Event::Character(' ')) {
                    replay_paused_ = !replay_paused_;
                    if (!replay_paused_)
                        replay_last_tick_ = std::chrono::steady_clock::now();
                    return true;
                }
                // n: step forward one snapshot
                if (event == Event::Character('n')) {
                    if (replay_index_ + 1 < replay_snapshots_.size()) {
                        ++replay_index_;
                        auto& snap = replay_snapshots_[replay_index_];
                        metrics_panel_.set_snapshot(snap);
                        metrics_panel_.add_latency_sample(
                            static_cast<float>(snap.latency_ms));
                        metrics_panel_.add_stats_sample(
                            snap.act_min, snap.act_mean, snap.act_max);
                        if (!snap.attention_weights.empty())
                            attention_heatmap_.set_data(
                                snap.attention_weights, snap.attn_size,
                                snap.num_heads);
                    }
                    return true;
                }
                // [: increase speed (lower delay, min 50ms)
                if (event == Event::Character('[')) {
                    replay_speed_ms_ = std::max(50, replay_speed_ms_ - 100);
                    return true;
                }
                // ]: decrease speed (higher delay, max 2000ms)
                if (event == Event::Character(']')) {
                    replay_speed_ms_ = std::min(2000, replay_speed_ms_ + 100);
                    return true;
                }
            }

            // Do not trigger live hotkeys if user is typing a prompt
            if (prompt_input->Focused()) {
                return false;
            }

            // ── Live-mode keybindings ─────────────────────────────
            // P: toggle pause on packet stream
            if (event == Event::Character('p') ||
                event == Event::Character('P')) {
                packet_stream_.toggle_pause();
                return true;
            }
            // E: export current capture to JSON
            if (event == Event::Character('e') ||
                event == Event::Character('E')) {
                auto snapshots = snapshot_buffer_.snapshot();
                std::string model_name = engine_ && engine_->get_model()
                    ? engine_->get_model()->name : "unknown";
                export_capture("captures", model_name, snapshots,
                               config_.buffer_size);
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
        // Load capture file into replay state
        replay_snapshots_ = import_capture(config.capture_path);
        replay_index_     = 0;
        replay_paused_    = true;
        // Seed the snapshot buffer with the first frame so the panels are
        // not completely empty on launch.
        if (!replay_snapshots_.empty()) {
            snapshot_buffer_.load(replay_snapshots_);
        }
        in_dashboard_ = true;
        screen_index_ = 1;
        return;
    }

    // Create inference engine
    engine_ = std::make_unique<InferenceEngine>(
        snapshot_buffer_, anomaly_buffer_);

    engine_->set_refresh_callback([this] {
        screen_.PostEvent(Event::Custom);
    });

    topology_tree_.set_on_select([this](const std::string& layer_name) {
        selected_layer_name_ = layer_name;
        if (engine_) {
            debug_log("Selected layer in topology: " + layer_name);
            // Don't clear attention heatmap here, let refresh_panels do it based on selected_layer_name_
        }
    });

    engine_->load_model_async(config.model_path, config.n_ctx,
                               config.n_threads, config.use_gpu);

    in_dashboard_ = true;
    screen_index_ = 1;

    // If an initial prompt was given via CLI (-p), fire it once the model loads.
    if (!cli_initial_prompt_.empty()) {
        std::string prompt_copy = cli_initial_prompt_;
        cli_initial_prompt_.clear();
        // Poll until model is ready, then run in a detached thread
        std::thread([this, prompt_copy] {
            using namespace std::chrono_literals;
            for (int i = 0; i < 120; ++i) {  // wait up to 60 s
                if (engine_ && engine_->get_state() == InferenceState::Ready) {
                    engine_->run_inference(prompt_copy);
                    screen_.PostEvent(Event::Custom);
                    return;
                }
                std::this_thread::sleep_for(500ms);
            }
        }).detach();
    }
}

// ─── Panel refresh ─────────────────────────────────────────────
void App::refresh_panels() {
    bool is_replay_mode = (engine_ == nullptr);

    // ── Replay mode: auto-advance playback ─────────────────────
    if (is_replay_mode && !replay_snapshots_.empty() && !replay_paused_) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - replay_last_tick_).count();

        if (elapsed >= replay_speed_ms_) {
            replay_last_tick_ = now;
            if (replay_index_ + 1 < replay_snapshots_.size()) {
                ++replay_index_;
            } else {
                // End of trace: pause automatically
                replay_paused_ = true;
            }
        }

        // Push current replay snapshot into panels
        auto& snap = replay_snapshots_[replay_index_];
        metrics_panel_.set_snapshot(snap);
        metrics_panel_.add_latency_sample(
            static_cast<float>(snap.latency_ms));
        metrics_panel_.add_stats_sample(
            snap.act_min, snap.act_mean, snap.act_max);
        if (!snap.attention_weights.empty()) {
            attention_heatmap_.set_data(
                snap.attention_weights, snap.attn_size, snap.num_heads);
        }
        return;
    }

    // ── Live mode ──────────────────────────────────────────────
    if (!engine_) return;

    auto* model = engine_->get_model();
    if (model) {
        static bool topology_set = false;
        if (!topology_set) {
            topology_tree_.set_topology(model->topology);
            topology_set = true;
        }
        topology_tree_.set_active_layer(
            engine_->get_hook_manager().get_active_layer());
    }

    if (!snapshot_buffer_.empty()) {
        try {
            // --- Metrics panel: show selected layer or latest ---
            LayerSnapshot metrics_snap;
            if (!selected_layer_name_.empty()) {
                auto sel_opt = snapshot_buffer_.find_latest([&](const LayerSnapshot& s) {
                    return s.layer_name == selected_layer_name_;
                });
                metrics_snap = sel_opt.value_or(snapshot_buffer_.latest());
            } else {
                metrics_snap = snapshot_buffer_.latest();
            }
            metrics_panel_.set_snapshot(metrics_snap);
            metrics_panel_.add_latency_sample(
                static_cast<float>(metrics_snap.latency_ms));
            metrics_panel_.add_stats_sample(
                metrics_snap.act_min, metrics_snap.act_mean, metrics_snap.act_max);
            metrics_panel_.set_throughput(
                engine_->get_tokens_per_second());

            // --- Attention heatmap: ALWAYS find most recent attention data ---
            auto attn_opt = snapshot_buffer_.find_latest([](const LayerSnapshot& s) {
                return !s.attention_weights.empty();
            });
            if (attn_opt) {
                auto labels = engine_->get_token_labels();
                attention_heatmap_.set_data(
                    attn_opt->attention_weights,
                    attn_opt->attn_size,
                    attn_opt->num_heads,
                    labels);
            }
        } catch (...) {}
    }
}

} // namespace neuralscope
