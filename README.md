# 🔬 NeuralScope

**Non-Invasive LLM Inspection TUI** — A real-time, interactive Terminal User Interface for inspecting transformer model internals during inference, without modifying any model source code.

Built entirely in **C++17** using [llama.cpp](https://github.com/ggml-org/llama.cpp) + [FTXUI](https://github.com/ArthurSonzogni/FTXUI) + [nlohmann/json](https://github.com/nlohmann/json).

---

## ✨ Features

- **Non-invasive hooking** — Uses `ggml_backend_sched_eval_callback` to intercept tensor computations without modifying model code
- **5-panel interactive dashboard** — Model topology, live packet stream, attention heatmap, runtime metrics, anomaly ledger
- **Vim-like keybindings** — `j/k` navigation, `Tab` focus cycling, `h/j/k/l` panning
- **Real-time metrics** — Layer latency, sparsity gauges, activation sparklines, tokens/sec throughput
- **Attention matrix visualization** — Unicode block-character heatmap with 16x16 viewport and head cycling
- **Anomaly detection** — Flags NaN/Inf, clipping risks, dead layers, outlier activations
- **Hardware Acceleration** — Built-in support for VRAM offloading via `llama.cpp` for ultra-fast GPU-accelerated inference
- **Ring buffer** — Fixed-size memory-bounded storage (configurable 64-512 snapshots)
- **Export/Import Replay Mode** — Save captures to JSON and reload them via the Startup Menu for offline replay and historical analysis without needing to run inference

---

## 📋 Requirements

- **OS**: Windows 10/11 (also works on Linux/macOS)
- **CMake**: 3.20+
- **C++ Compiler**: MSVC 2022, GCC 10+, or Clang 14+
- **llama.cpp**: Cloned and available locally
- **RAM**: 4GB+ (8GB recommended)
- **Model**: Any GGUF-format model (TinyLlama Q4_K_M recommended for testing)

---

## 🚀 Build

```bash
# 1. Clone this repo
git clone https://github.com/s0n1c07/GDSC-PS-1.git
cd GDSC-PS-1

# 2. Place your GGUF model in the models/ directory
# Example: models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# 3. Create build directory
mkdir build
cd build

# 4. Configure (adjust LLAMA_CPP_DIR if needed)
cmake .. -DLLAMA_CPP_DIR="C:/Users/Rajve/OneDrive/Desktop/llama.cpp"

# 5. Build
cmake --build . --config Release

# 6. Run
./Release/neuralscope.exe
```

---

## 🎮 Keybindings

| Key | Action | Scope |
|-----|--------|-------|
| `Tab` | Cycle focus between panels | Global |
| `Q` | Quit application | Global |
| `Esc` | Return to startup menu | Global |
| `P` | Pause/resume packet stream | Global |
| `E` | Export current buffer to JSON | Global |
| `j` / `k` | Navigate up/down | Topology, Stream |
| `Space` | Expand/collapse node | Topology |
| `h/j/k/l` / Arrows | Pan attention viewport | Heatmap |
| `[` / `]` | Previous/next attention head | Heatmap |
| `+` / `-` | Adjust contrast | Heatmap |
| `Enter` | Submit prompt for inference | Prompt Bar |

---

## 🏗️ Architecture

```
src/
├── main.cpp              # Entry point
├── core/
│   ├── types.hpp         # LayerSnapshot, TopologyNode, enums
│   ├── ring_buffer.hpp   # Thread-safe fixed-capacity buffer
│   └── utils.hpp         # Timestamps, sparklines, gauges
├── engine/
│   ├── analyzer.*        # Activation stats + anomaly detection
│   ├── hook_manager.*    # ggml eval callback (non-invasive hooks)
│   ├── model_loader.*    # GGUF model loading + topology extraction
│   └── inference.*       # Background threaded inference
├── tui/
│   ├── app.*             # Main dashboard compositor
│   ├── startup_menu.*    # Model selection screen
│   ├── topology_tree.*   # Panel 1: Model architecture tree
│   ├── packet_stream.*   # Panel 2: Live event table
│   ├── attention_heatmap.* # Panel 3: Attention matrix visualizer
│   ├── metrics_panel.*   # Panel 4: Runtime metrics + sparklines
│   └── anomaly_ledger.*  # Panel 5: Numerical anomaly log
└── io/
    ├── exporter.*        # JSON/CSV capture export
    └── importer.*        # Capture file replay
```

---

## 📊 Metrics Captured

| Metric | Description |
|--------|-------------|
| Layer Latency | Per-tensor execution time (ms) |
| Sparsity | Fraction of near-zero activations |
| Activation Stats | Min, max, mean, std deviation |
| Tensor Shape | Output shape of each operation |
| Throughput | Tokens generated per second |
| NaN/Inf Detection | Flags numerical instabilities |
| Clipping Risk | Warns when values exceed fp16 range |

---

## 📝 License

MIT