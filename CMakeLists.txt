cmake_minimum_required(VERSION 3.20)
project(NeuralScope VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Need C for ggml
set(CMAKE_C_STANDARD 11)

# Force static libraries for all dependencies
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(LLAMA_STATIC ON CACHE BOOL "" FORCE)

include(FetchContent)

# ─── llama.cpp (fetched from GitHub) ────────────────────────────
# Using FetchContent instead of local path to avoid OneDrive sync issues
FetchContent_Declare(llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
# Disable llama.cpp examples and tests to speed up build
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(llama_cpp)

# ─── FTXUI (auto-fetched) ───────────────────────────────────────
FetchContent_Declare(ftxui
    GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI
    GIT_TAG        v5.0.0
    GIT_SHALLOW    TRUE
)
set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_DOCS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(ftxui)

# ─── nlohmann/json (auto-fetched, header-only) ──────────────────
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(json)

# ─── CLI11 (header-only CLI argument parser) ────────────────────
FetchContent_Declare(cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11
    GIT_TAG        v2.4.2
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(cli11)

# ─── NeuralScope executable ─────────────────────────────────────
set(NEURALSCOPE_SOURCES
    src/main.cpp
    src/engine/analyzer.cpp
    src/engine/hook_manager.cpp
    src/engine/model_loader.cpp
    src/engine/inference.cpp
    src/tui/app.cpp
    src/tui/startup_menu.cpp
    src/tui/topology_tree.cpp
    src/tui/packet_stream.cpp
    src/tui/attention_heatmap.cpp
    src/tui/metrics_panel.cpp
    src/tui/anomaly_ledger.cpp
    src/io/exporter.cpp
    src/io/importer.cpp
)

add_executable(neuralscope ${NEURALSCOPE_SOURCES})

target_include_directories(neuralscope PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/common
    ${llama_cpp_SOURCE_DIR}/ggml/include
)

target_link_libraries(neuralscope PRIVATE
    llama
    llama-common
    ggml
    ftxui::screen
    ftxui::dom
    ftxui::component
    nlohmann_json::nlohmann_json
    CLI11::CLI11
)

# On Windows, link against ws2_32 for potential networking needs
if(WIN32)
    target_link_libraries(neuralscope PRIVATE ws2_32)
endif()

target_compile_definitions(neuralscope PRIVATE
    NEURALSCOPE_VERSION="${PROJECT_VERSION}"
)
