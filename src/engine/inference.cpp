#include "engine/inference.hpp"
#include "core/utils.hpp"

#include "llama.h"
#include "ggml-backend.h"

#include <chrono>
#include <iostream>

namespace neuralscope {

InferenceEngine::InferenceEngine(
    RingBuffer<LayerSnapshot>& snapshot_buffer,
    RingBuffer<AnomalyEntry>&  anomaly_buffer)
    : snapshot_buffer_(snapshot_buffer)
    , anomaly_buffer_(anomaly_buffer)
    , hook_manager_(snapshot_buffer, anomaly_buffer)
{}

InferenceEngine::~InferenceEngine() {
    stop();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void InferenceEngine::load_model_async(const std::string& model_path,
                                       int n_ctx, int n_threads, bool use_gpu)
{
    if (state_.load() != InferenceState::Idle &&
        state_.load() != InferenceState::Error) {
        return;
    }
    state_.store(InferenceState::Loading);
    stop_flag_.store(false);
    load_progress_.store(0.0f);

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    worker_thread_ = std::thread(&InferenceEngine::worker_load, this,
                                  model_path, n_ctx, n_threads, use_gpu);
}

void InferenceEngine::run_inference(const std::string& prompt, int max_tokens) {
    if (state_.load() != InferenceState::Ready &&
        state_.load() != InferenceState::Finished) {
        return; // Model not loaded
    }

    // Wait for any existing worker to finish
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    state_.store(InferenceState::Running);
    stop_flag_.store(false);
    tokens_per_second_.store(0.0);

    {
        std::lock_guard<std::mutex> lock(text_mutex_);
        generated_text_.clear();
    }

    // Reset hook manager counters
    hook_manager_.reset();

    worker_thread_ = std::thread(&InferenceEngine::worker_inference, this,
                                  prompt, max_tokens);
}

void InferenceEngine::stop() {
    stop_flag_.store(true);
}

InferenceState InferenceEngine::get_state() const {
    return state_.load();
}

std::string InferenceEngine::get_generated_text() const {
    std::lock_guard<std::mutex> lock(text_mutex_);
    return generated_text_;
}

double InferenceEngine::get_tokens_per_second() const {
    return tokens_per_second_.load();
}

float InferenceEngine::get_load_progress() const {
    return load_progress_.load();
}

HookManager& InferenceEngine::get_hook_manager() {
    return hook_manager_;
}

ModelHandle* InferenceEngine::get_model() const {
    return model_.get();
}

void InferenceEngine::set_refresh_callback(std::function<void()> cb) {
    std::lock_guard<std::mutex> lock(refresh_mutex_);
    refresh_callback_ = std::move(cb);
}

std::string InferenceEngine::get_error() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return error_message_;
}

std::vector<std::string> InferenceEngine::get_token_labels() const {
    std::lock_guard<std::mutex> lock(token_labels_mutex_);
    return token_labels_;
}

void InferenceEngine::notify_refresh() {
    std::lock_guard<std::mutex> lock(refresh_mutex_);
    if (refresh_callback_) {
        refresh_callback_();
    }
}

// ─── Background worker: model loading ──────────────────────────
void InferenceEngine::worker_load(const std::string& model_path,
                                   int n_ctx, int n_threads, bool use_gpu)
{
    try {
        model_ = load_model(model_path, n_ctx, n_threads, use_gpu,
            [this](float progress) {
                load_progress_.store(progress);
                notify_refresh();
            },
            HookManager::eval_callback,
            &hook_manager_
        );

        if (!model_) {
            {
                std::lock_guard<std::mutex> lock(error_mutex_);
                error_message_ = "Failed to load model: " + model_path;
            }
            state_.store(InferenceState::Error);
            notify_refresh();
            return;
        }

        state_.store(InferenceState::Ready);
        notify_refresh();

    } catch (const std::exception& e) {
        {
            std::lock_guard<std::mutex> lock(error_mutex_);
            error_message_ = std::string("Exception loading model: ") + e.what();
        }
        state_.store(InferenceState::Error);
        notify_refresh();
    }
}

// ─── Background worker: inference ──────────────────────────────
void InferenceEngine::worker_inference(const std::string& prompt,
                                        int max_tokens)
{
    if (!model_ || !model_->ctx || !model_->model) {
        state_.store(InferenceState::Error);
        return;
    }

    try {
        const llama_vocab* vocab = llama_model_get_vocab(model_->model);

        // Tokenize the prompt
        const int max_prompt_tokens = 1024;
        std::vector<llama_token> tokens(max_prompt_tokens);
        int n_tokens = llama_tokenize(
            vocab,
            prompt.c_str(),
            static_cast<int32_t>(prompt.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            true,   // add_bos
            true    // special
        );

        if (n_tokens < 0) {
            tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(
                vocab,
                prompt.c_str(),
                static_cast<int32_t>(prompt.size()),
                tokens.data(),
                static_cast<int32_t>(tokens.size()),
                true, true
            );
        }
        tokens.resize(n_tokens);

        // Convert prompt tokens to string labels
        {
            std::lock_guard<std::mutex> lock(token_labels_mutex_);
            token_labels_.clear();
            for (int i = 0; i < n_tokens; ++i) {
                char buf[256];
                int len = llama_token_to_piece(vocab, tokens[i],
                                              buf, sizeof(buf), 0, true);
                if (len > 0) {
                    std::string piece(buf, len);
                    // Trim whitespace for cleaner display
                    while (!piece.empty() && piece[0] == ' ') piece.erase(0, 1);
                    if (piece.empty()) piece = "_";
                    token_labels_.push_back(piece);
                } else {
                    token_labels_.push_back("<" + std::to_string(i) + ">");
                }
            }
        }

        if (tokens.empty()) {
            {
                std::lock_guard<std::mutex> lock(error_mutex_);
                error_message_ = "Failed to tokenize prompt";
            }
            state_.store(InferenceState::Error);
            notify_refresh();
            return;
        }

        // Clear KV cache
        llama_memory_clear(llama_get_memory(model_->ctx), true);

        // ── Prefill: process all prompt tokens ─────────────────
        auto start_time = std::chrono::high_resolution_clock::now();
        int n_generated = 0;

        // Create batch for prefill
        llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);

        if (llama_decode(model_->ctx, batch) != 0) {
            {
                std::lock_guard<std::mutex> lock(error_mutex_);
                error_message_ = "llama_decode failed during prefill";
            }
            state_.store(InferenceState::Error);
            notify_refresh();
            return;
        }

        notify_refresh();

        // ── Autoregressive generation ──────────────────────────
        int n_cur = n_tokens;
        const int n_ctx = model_->n_ctx;

        for (int i = 0; i < max_tokens && !stop_flag_.load(); ++i) {
            // Sample the next token
            float* logits = llama_get_logits_ith(model_->ctx, -1);

            // Simple greedy sampling (argmax)
            llama_token new_token_id = 0;
            float max_logit = logits[0];
            int vocab_size = llama_vocab_n_tokens(vocab);
            for (int j = 1; j < vocab_size; ++j) {
                if (logits[j] > max_logit) {
                    max_logit = logits[j];
                    new_token_id = j;
                }
            }

            // Check for EOS
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            // Convert token to text
            char buf[256];
            int len = llama_token_to_piece(vocab, new_token_id,
                                            buf, sizeof(buf), 0, true);
            if (len > 0) {
                std::string piece(buf, len);
                {
                    std::lock_guard<std::mutex> lock(text_mutex_);
                    generated_text_ += piece;
                }
                // Add to token labels
                {
                    std::lock_guard<std::mutex> lock(token_labels_mutex_);
                    std::string label = piece;
                    while (!label.empty() && label[0] == ' ') label.erase(0, 1);
                    if (label.empty()) label = "_";
                    token_labels_.push_back(label);
                }
            }

            ++n_generated;
            ++n_cur;

            // Update tokens per second
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            if (elapsed > 0.0) {
                tokens_per_second_.store(
                    static_cast<double>(n_generated) / elapsed);
            }

            // Check context limit
            if (n_cur >= n_ctx) {
                break;
            }

            // Decode the new token
            llama_batch single = llama_batch_get_one(&new_token_id, 1);
            if (llama_decode(model_->ctx, single) != 0) {
                {
                    std::lock_guard<std::mutex> lock(error_mutex_);
                    error_message_ = "llama_decode failed during generation";
                }
                state_.store(InferenceState::Error);
                notify_refresh();
                return;
            }

            notify_refresh();
        }

        state_.store(InferenceState::Finished);
        notify_refresh();

    } catch (const std::exception& e) {
        {
            std::lock_guard<std::mutex> lock(error_mutex_);
            error_message_ = std::string("Inference error: ") + e.what();
        }
        state_.store(InferenceState::Error);
        notify_refresh();
    }
}

} // namespace neuralscope
