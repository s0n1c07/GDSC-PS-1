/// ─────────────────────────────────────────────────────────────
/// NeuralScope — Non-Invasive LLM Inspection TUI
/// Built with llama.cpp + FTXUI + nlohmann/json
///
/// Entry point: creates and runs the App.
/// ─────────────────────────────────────────────────────────────

#include "tui/app.hpp"
#include "core/utils.hpp"

#include <iostream>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    neuralscope::debug_log("-----------------------------------------");
    neuralscope::debug_log("main() started");

    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--help" || arg == "-h") {
            neuralscope::debug_log("CLI --help option called");
            std::cout << "🔬 NeuralScope v1.0.0" << std::endl;
            std::cout << "Non-Invasive LLM Inspection TUI" << std::endl;
            std::cout << "Usage: neuralscope.exe [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -h, --help     Show this help message" << std::endl;
            std::cout << "  -v, --version  Show version info" << std::endl;
            return 0;
        }
        if (arg == "--version" || arg == "-v") {
            neuralscope::debug_log("CLI --version option called");
            std::cout << "1.0.0" << std::endl;
            return 0;
        }
    }

    try {
        neuralscope::debug_log("Creating neuralscope::App object");
        neuralscope::App app;
        neuralscope::debug_log("neuralscope::App object constructed successfully");
        
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
