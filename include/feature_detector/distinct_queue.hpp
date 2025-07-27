//
// OPTIMIZED Distinct Queue for Event Cameras
// High-performance version with memory layout and algorithm optimizations
//

#ifndef PROJECT_DISTINCT_QUEUE_HPP
#define PROJECT_DISTINCT_QUEUE_HPP

#include <vector>
#include <cstdint>
#include <memory>
#include <cstring>

// Optimized fixed-size circular buffer with spatial indexing
class FixedDistinctQueue {
public:
    FixedDistinctQueue(uint32_t window_size, uint32_t max_queue_size) :
            window_size_(window_size), queue_max_(max_queue_size),
            queue_size_(0), head_(0), tail_(0) {

        // Allocate contiguous memory for better cache performance
        const size_t window_elements = window_size * window_size;
        window_data_ = std::make_unique<int16_t[]>(window_elements);
        queue_data_ = std::make_unique<QueueEvent[]>(max_queue_size);

        // Initialize window to empty (-1)
        std::fill_n(window_data_.get(), window_elements, -1);
    }

    inline bool isFull() const {
        return queue_size_ >= queue_max_;
    }

    inline void addNew(uint32_t x, uint32_t y) {
        // Fast bounds check
        if (x >= window_size_ || y >= window_size_) {
            return;
        }

        const size_t window_idx = y * window_size_ + x;
        const int16_t existing_queue_idx = window_data_[window_idx];

        if (existing_queue_idx >= 0) {
            // Element exists - move to front (most recent)
            moveToFront(existing_queue_idx);
        } else {
            // New element
            if (queue_size_ < queue_max_) {
                // Queue not full - add new element
                insertNew(x, y, window_idx);
            } else {
                // Queue full - replace oldest element
                replaceOldest(x, y, window_idx);
            }
        }
    }

    // Fast binary representation of window state
    inline void getWindow(uint8_t* output) const {
        const size_t window_elements = window_size_ * window_size_;
        const size_t bytes_needed = (window_elements + 7) / 8;
        std::memset(output, 0, bytes_needed);

        for (size_t i = 0; i < window_elements; ++i) {
            if (window_data_[i] >= 0) {
                output[i / 8] |= (1 << (i % 8));
            }
        }
    }

    // Legacy interface for compatibility
    std::vector<std::vector<int>> getWindowMatrix() const {
        std::vector<std::vector<int>> result(window_size_, std::vector<int>(window_size_, 0));
        for (uint32_t y = 0; y < window_size_; ++y) {
            for (uint32_t x = 0; x < window_size_; ++x) {
                result[y][x] = (window_data_[y * window_size_ + x] >= 0) ? 1 : 0;
            }
        }
        return result;
    }

private:
    struct QueueEvent {
        uint16_t x, y;  // Using smaller types to improve cache efficiency
        int16_t prev, next;
    };

    // Memory-efficient storage
    std::unique_ptr<int16_t[]> window_data_;  // Maps window position to queue index
    std::unique_ptr<QueueEvent[]> queue_data_;

    const uint32_t window_size_;
    const uint32_t queue_max_;
    uint32_t queue_size_;
    int16_t head_;  // Most recent element
    int16_t tail_;  // Oldest element

    inline void insertNew(uint32_t x, uint32_t y, size_t window_idx) {
        const int16_t new_idx = static_cast<int16_t>(queue_size_);

        queue_data_[new_idx].x = static_cast<uint16_t>(x);
        queue_data_[new_idx].y = static_cast<uint16_t>(y);
        queue_data_[new_idx].prev = -1;
        queue_data_[new_idx].next = head_;

        if (head_ >= 0) {
            queue_data_[head_].prev = new_idx;
        } else {
            tail_ = new_idx;  // First element
        }

        head_ = new_idx;
        window_data_[window_idx] = new_idx;
        ++queue_size_;
    }

    inline void replaceOldest(uint32_t x, uint32_t y, size_t window_idx) {
        if (tail_ < 0) return;

        // Clear old window position
        const size_t old_window_idx = queue_data_[tail_].y * window_size_ + queue_data_[tail_].x;
        window_data_[old_window_idx] = -1;

        // Update queue element
        queue_data_[tail_].x = static_cast<uint16_t>(x);
        queue_data_[tail_].y = static_cast<uint16_t>(y);

        // Move to front
        const int16_t old_tail = tail_;
        moveToFront(old_tail);

        // Set new window position
        window_data_[window_idx] = head_;
    }

    inline void moveToFront(int16_t idx) {
        if (idx == head_) return;  // Already at front

        // Remove from current position
        if (queue_data_[idx].prev >= 0) {
            queue_data_[queue_data_[idx].prev].next = queue_data_[idx].next;
        }
        if (queue_data_[idx].next >= 0) {
            queue_data_[queue_data_[idx].next].prev = queue_data_[idx].prev;
        } else {
            tail_ = queue_data_[idx].prev;  // Was tail
        }

        // Insert at front
        queue_data_[idx].prev = -1;
        queue_data_[idx].next = head_;
        if (head_ >= 0) {
            queue_data_[head_].prev = idx;
        }
        head_ = idx;
    }
};

// Optimized main queue system
class DistinctQueue {
public:
    DistinctQueue(uint32_t window_size, uint32_t queue_size, bool use_polarity,
                  uint32_t sensor_width, uint32_t sensor_height)
            : window_size_(window_size), sensor_width_(sensor_width),
              sensor_height_(sensor_height), use_polarity_(use_polarity) {

        // Validate inputs
        if (sensor_width == 0 || sensor_height == 0 ||
            window_size > std::min(sensor_width, sensor_height) / 2) {
            throw std::invalid_argument("Invalid sensor dimensions or window size");
        }

        // Calculate total number of queues needed
        const size_t base_queues = static_cast<size_t>(sensor_width) * sensor_height;
        const size_t total_queues = use_polarity ? (base_queues * 2) : base_queues;

        // Check for overflow
        if (total_queues > SIZE_MAX / sizeof(std::unique_ptr<FixedDistinctQueue>)) {
            throw std::invalid_argument("Sensor dimensions too large");
        }

        // Pre-allocate all queues
        queues_.reserve(total_queues);
        for (size_t i = 0; i < total_queues; ++i) {
            queues_.emplace_back(std::make_unique<FixedDistinctQueue>(
                    2 * window_size + 1, queue_size));
        }

        // Pre-compute window bounds to avoid repeated calculations
        window_start_ = -static_cast<int32_t>(window_size);
        window_end_ = static_cast<int32_t>(window_size);
    }

    inline void newEvent(uint32_t x, uint32_t y, bool polarity = false) {
        // Fast boundary check
        if (x >= sensor_width_ || y >= sensor_height_) {
            return;
        }

        // Update all neighboring pixels in the window
        const int32_t x_int = static_cast<int32_t>(x);
        const int32_t y_int = static_cast<int32_t>(y);

        for (int32_t dx = window_start_; dx <= window_end_; ++dx) {
            const int32_t new_x = x_int + dx;
            if (new_x < 0 || new_x >= static_cast<int32_t>(sensor_width_)) {
                continue;
            }

            for (int32_t dy = window_start_; dy <= window_end_; ++dy) {
                const int32_t new_y = y_int + dy;
                if (new_y < 0 || new_y >= static_cast<int32_t>(sensor_height_)) {
                    continue;
                }

                // Get queue index and update
                const size_t queue_idx = getQueueIndex(static_cast<uint32_t>(new_x),
                                                       static_cast<uint32_t>(new_y), polarity);

                // Add event to the queue (relative coordinates)
                queues_[queue_idx]->addNew(
                        static_cast<uint32_t>(window_size_ + dx),
                        static_cast<uint32_t>(window_size_ + dy)
                );
            }
        }
    }

    inline bool isFull(uint32_t x, uint32_t y, bool polarity = false) const {
        if (x >= sensor_width_ || y >= sensor_height_) {
            return false;
        }
        const size_t idx = getQueueIndex(x, y, polarity);
        return queues_[idx]->isFull();
    }

    // Fast binary patch retrieval
    inline void getPatchBinary(uint32_t x, uint32_t y, uint8_t* output, bool polarity = false) const {
        if (x >= sensor_width_ || y >= sensor_height_) {
            const size_t window_elements = (2 * window_size_ + 1) * (2 * window_size_ + 1);
            const size_t bytes_needed = (window_elements + 7) / 8;
            std::memset(output, 0, bytes_needed);
            return;
        }
        const size_t idx = getQueueIndex(x, y, polarity);
        queues_[idx]->getWindow(output);
    }

    // Legacy interface for compatibility
    std::vector<std::vector<int>> getPatch(uint32_t x, uint32_t y, bool polarity = false) const {
        if (x >= sensor_width_ || y >= sensor_height_) {
            const size_t window_dim = 2 * window_size_ + 1;
            return std::vector<std::vector<int>>(window_dim, std::vector<int>(window_dim, 0));
        }
        const size_t idx = getQueueIndex(x, y, polarity);
        return queues_[idx]->getWindowMatrix();
    }

private:
    // Optimized queue storage
    std::vector<std::unique_ptr<FixedDistinctQueue>> queues_;

    // Pre-computed constants
    const uint32_t window_size_;
    const uint32_t sensor_width_;
    const uint32_t sensor_height_;
    const bool use_polarity_;
    int32_t window_start_;
    int32_t window_end_;

    // Fast index calculation with overflow protection
    inline size_t getQueueIndex(uint32_t x, uint32_t y, bool polarity) const {
        const size_t base_index = static_cast<size_t>(y) * sensor_width_ + x;
        if (use_polarity_ && polarity) {
            return base_index + (static_cast<size_t>(sensor_width_) * sensor_height_);
        }
        return base_index;
    }
};

#endif //PROJECT_DISTINCT_QUEUE_HPP