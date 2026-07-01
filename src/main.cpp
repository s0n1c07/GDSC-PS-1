/// ─────────────────────────────────────────────────────────────
/// NeuralScope — Non-Invasive LLM Inspection TUI
/// Built with llama.cpp + FTXUI + nlohmann/json + CLI11
///
/// Entry point: parses CLI arguments and runs the App.
/// ─────────────────────────────────────────────────────────────

#include "tui/app.hpp"
#include "core/types.hpp"
#include "core/utils.hpp"

#include <CLI/CLI.hpp>
#include <iostream>
#include <string>
#include <filesystem>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    neuralscope::debug_log("-----------------------------------------");
    neuralscope::debug_log("main() started");

    // ─── CLI Argument Parsing ─────────────────────────────────────
    CLI::App cli{"NeuralScope v" NEURALSCOPE_VERSION
                 " - Non-Invasive LLM Inspection TUI"};
    cli.set_version_flag("-v,--version", std::string(NEURALSCOPE_VERSION));

    neuralscope::AppConfig config;

    // Model path
    std::string model_path;
    cli.add_option("-m,--model", model_path,
                   "Path to the GGUF model file")
        ->check(CLI::ExistingFile);

    // Replay mode (load a previously exported capture)
    std::string replay_path;
    cli.add_option("--replay", replay_path,
                   "Load a JSON capture file for replay mode")
        ->check(CLI::ExistingFile);

    // Auto-run prompt
    std::string prompt;
    cli.add_option("-p,--prompt", prompt,
                   "Prompt to run automatically on launch (live mode only)");

    // Context size
    int ctx_size = 512;
    cli.add_option("-c,--ctx-size", ctx_size,
                   "KV-cache context size (default: 512)")
        ->check(CLI::PositiveNumber);

    // Number of tokens to predict
    int n_predict = 128;
    cli.add_option("-n,--n-predict", n_predict,
                   "Max tokens to generate (default: 128)")
        ->check(CLI::PositiveNumber);

    // Thread count
    int threads = 4;
    cli.add_option("-t,--threads", threads,
                   "Inference threads (default: 4)")
        ->check(CLI::PositiveNumber);

    // GPU layers
    int gpu_layers = 0;
    cli.add_option("--gpu-layers", gpu_layers,
                   "Number of layers to offload to GPU (default: 0)")
        ->check(CLI::NonNegativeNumber);

    // Ring-buffer size
    int buffer_size = 256;
    cli.add_option("--buffer-size", buffer_size,
                   "Snapshot ring-buffer size (default: 256)")
        ->check(CLI::PositiveNumber);

    // Trace output (enable trace writing to the captures/ directory)
    bool trace = false;
    cli.add_flag("--trace", trace,
                 "Auto-export a JSON capture on exit");

    CLI11_PARSE(cli, argc, argv);

    // ─── Validate & Build AppConfig ────────────────────────────────
    bool has_model   = !model_path.empty();
    bool has_replay  = !replay_path.empty();

    if (has_model && has_replay) {
        std::cerr << "Error: --model and --replay are mutually exclusive.\n";
        return 1;
    }
    if (!has_model && !has_replay) {
        // Neither flag given — launch the interactive startup menu (default)
        neuralscope::debug_log("No flags given - launching TUI startup menu");
    }

    config.n_ctx       = ctx_size;
    config.n_threads   = threads;
    config.buffer_size = static_cast<size_t>(buffer_size);
    config.use_gpu     = (gpu_layers > 0);

    if (has_model) {
        config.model_path   = model_path;
        config.replay_mode  = false;
    } else if (has_replay) {
        config.capture_path = replay_path;
        config.replay_mode  = true;
    }

    // ─── Launch App ────────────────────────────────────────────────
    try {
        neuralscope::debug_log("Creating neuralscope::App object");
        neuralscope::App app;
        neuralscope::debug_log("neuralscope::App object constructed successfully");

        // If CLI provided a model or replay path, skip the startup menu
        if (has_model || has_replay) {
            neuralscope::debug_log("CLI config provided - skipping startup menu");
            app.launch_with_config(config, prompt);
        }

        neuralscope::debug_log("Calling app.run()");
        app.run();
        neuralscope::debug_log("app.run() returned successfully");

    } catch (const std::exception& e) {
        neuralscope::debug_log("Fatal error (std::exception): " + std::string(e.what()));
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        neuralscope::debug_log("Fatal error (unknown exception)");
        std::cerr << "Fatal error: Unknown exception caught!" << std::endl;
        return 1;
    }

    neuralscope::debug_log("main() returning 0");
    return 0;
}
