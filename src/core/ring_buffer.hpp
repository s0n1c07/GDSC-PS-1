#pragma once

#include <deque>
#include <mutex>
#include <vector>
#include <functional>
#include <optional>
#include <stdexcept>

namespace neuralscope {

/// Thread-safe, fixed-capacity ring buffer.
/// When full, the oldest item is evicted on push.
template<typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t max_size = 256)
        : max_size_(max_size) {}

    /// Push an item. If buffer is full, the oldest item is evicted.
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (buffer_.size() >= max_size_) {
            buffer_.pop_front();
        }
        buffer_.push_back(std::move(item));
        if (on_push_) on_push_();
    }

    /// Get a thread-safe copy of all items.
    std::vector<T> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return std::vector<T>(buffer_.begin(), buffer_.end());
    }

    /// Get item at index (0 = oldest). Throws if out of range.
    T get(size_t index) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= buffer_.size()) {
            throw std::out_of_range("RingBuffer::get index out of range");
        }
        return buffer_[index];
    }

    /// Get the most recent item. Throws if empty.
    T latest() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (buffer_.empty()) {
            throw std::out_of_range("RingBuffer::latest called on empty buffer");
        }
        return buffer_.back();
    }

    /// Get the last N items (most recent first).
    std::vector<T> last_n(size_t n) const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<T> result;
        size_t count = std::min(n, buffer_.size());
        for (size_t i = 0; i < count; ++i) {
            result.push_back(buffer_[buffer_.size() - 1 - i]);
        }
        return result;
    }

    /// Find the most recent item that matches a condition.
    std::optional<T> find_latest(std::function<bool(const T&)> predicate) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (buffer_.empty()) return std::nullopt;
        for (auto it = buffer_.rbegin(); it != buffer_.rend(); ++it) {
            if (predicate(*it)) {
                return *it;
            }
        }
        return std::nullopt;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.size();
    }

    size_t capacity() const {
        return max_size_;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.empty();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_.clear();
    }

    /// Set a callback that fires after each push (called under lock).
    void set_on_push(std::function<void()> callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        on_push_ = std::move(callback);
    }

    /// Replace buffer contents (for import/replay).
    void load(const std::vector<T>& items) {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_.clear();
        for (const auto& item : items) {
            if (buffer_.size() >= max_size_) {
                buffer_.pop_front();
            }
            buffer_.push_back(item);
        }
    }

    /// Resize the buffer and evict oldest items if new capacity is smaller.
    void resize(size_t max_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        max_size_ = max_size;
        while (buffer_.size() > max_size_) {
            buffer_.pop_front();
        }
    }

private:
    std::deque<T>             buffer_;
    size_t                    max_size_;
    mutable std::mutex        mutex_;
    std::function<void()>     on_push_;
};

} // namespace neuralscope
